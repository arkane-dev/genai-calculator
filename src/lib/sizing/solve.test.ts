import { describe, expect, it } from 'vitest';
import { sizeDisaggregated, sizeForGpu, solveWorkload } from './solve';
import type { WorkloadSpec } from './types';

const base: WorkloadSpec = {
	modelId: 'llama31-8b',
	gpuId: 'h100-sxm',
	gpusPerNode: 8,
	fabricId: 'ib-ndr',
	weightFormatId: 'bf16',
	kvBits: 16,
	kvAllocation: 'paged',
	concurrency: 64,
	basis: 'peak',
	inputTokens: 1024,
	outputTokens: 256,
	ttftTargetMs: 2000,
	throughputTargetTps: 20,
	steps: 50,
	resolution: 1024,
	guidance: true,
	targetImagesPerSec: 5,
	maxSecPerImage: 5,
	frames: 64,
	targetClipsPerSec: 10,
	maxSecPerClip: 2,
	targetDocsPerSec: 500,
	maxSecPerDoc: 0.5,
	imagesPerRequest: 0,
	targetRobots: 5,
	maxSecPerControl: 1,
	targetStreams: 100,
	minRtf: 1,
	specDecode: false,
	draftAcceptRate: 0.7,
	specTokens: 4,
	cachedPrefixFrac: 0
};

describe('sizeForGpu', () => {
	it('finds a feasible cluster that meets the SLOs', () => {
		const r = sizeForGpu(base, 'h100-sxm');
		expect(r.feasible).toBe(true);
		expect(r.numGpus).toBeGreaterThanOrEqual(1);
		expect(r.perUserTps).toBeGreaterThanOrEqual(base.throughputTargetTps);
		expect(r.ttftMs).toBeLessThanOrEqual(base.ttftTargetMs);
		expect(r.batchPerReplica * r.dp).toBeGreaterThanOrEqual(r.provisionedConcurrency);
	});

	it('needs more GPUs as concurrency rises', () => {
		const small = sizeForGpu({ ...base, concurrency: 32 }, 'h100-sxm').numGpus;
		const big = sizeForGpu({ ...base, concurrency: 512 }, 'h100-sxm').numGpus;
		expect(big).toBeGreaterThan(small);
	});

	it('a tighter per-user throughput SLO needs at least as many GPUs', () => {
		const loose = sizeForGpu({ ...base, throughputTargetTps: 10 }, 'h100-sxm').numGpus;
		const tight = sizeForGpu({ ...base, throughputTargetTps: 60 }, 'h100-sxm').numGpus;
		expect(tight).toBeGreaterThanOrEqual(loose);
	});

	it('applies the basis headroom (average provisions more than peak)', () => {
		const peak = sizeForGpu({ ...base, basis: 'peak' }, 'h100-sxm').provisionedConcurrency;
		const avg = sizeForGpu({ ...base, basis: 'average' }, 'h100-sxm').provisionedConcurrency;
		expect(avg).toBeGreaterThan(peak);
	});

	it('reports infeasible with a reason when the model cannot fit', () => {
		// 1 GPU/node caps parallelism to pp<=8, so 405B fp16 (~810GB) can't fit a 24GB card
		const r = sizeForGpu({ ...base, modelId: 'llama31-405b', gpusPerNode: 1 }, 'rtx4090');
		expect(r.feasible).toBe(false);
		expect(r.reason).toMatch(/fit/);
	});

	it('speculative decoding lifts per-user throughput, so it needs no more GPUs (fewer for an aggressive SLO)', () => {
		// low concurrency + aggressive per-user SLO: the throughput SLO binds, where spec decode helps
		const spec = { ...base, modelId: 'llama31-8b', concurrency: 8, throughputTargetTps: 200 };
		const off = sizeForGpu({ ...spec, specDecode: false }, 'h100-sxm');
		const on = sizeForGpu({ ...spec, specDecode: true }, 'h100-sxm');
		expect(off.feasible && on.feasible).toBe(true);
		expect(on.numGpus).toBeLessThanOrEqual(off.numGpus);
	});

	it('an impossible per-user throughput SLO names the ceiling, not a cluster cap', () => {
		// 5000 tok/s/user is far above any single-sequence decode rate
		const r = sizeForGpu({ ...base, throughputTargetTps: 5000 }, 'h100-sxm');
		expect(r.feasible).toBe(false);
		expect(r.reason).toMatch(/exceeds the hardware ceiling/);
		expect(r.reason).toMatch(/not a cluster-size cap/);
	});

	it('an infeasible result carries a non-trivial config so the UI can still render the die/memory', () => {
		// max-parallel probe (not the trivial 1×1×1) → profileable for the overflow view
		const r = sizeForGpu({ ...base, modelId: 'llama31-405b', gpusPerNode: 1 }, 'rtx4090');
		expect(r.feasible).toBe(false);
		expect(r.config.tp * r.config.pp * r.config.ep).toBeGreaterThan(1);
	});
});

describe('JEPA sizing', () => {
	it('sizes a JEPA model by clips/sec and reports clip latency', () => {
		const r = sizeForGpu(
			{ ...base, modelId: 'vjepa2-vitl', targetClipsPerSec: 20, maxSecPerClip: 2 },
			'h100-sxm'
		);
		expect(r.feasible).toBe(true);
		expect(r.clipsPerSec).toBeGreaterThanOrEqual(20);
		expect(r.secPerClip).toBeLessThanOrEqual(2);
		expect(r.ep).toBe(1);
	});

	it('a higher clips/sec target needs at least as many GPUs', () => {
		const lo = sizeForGpu(
			{ ...base, modelId: 'vjepa2-vith', targetClipsPerSec: 10, maxSecPerClip: 3 },
			'h100-sxm'
		).numGpus;
		const hi = sizeForGpu(
			{ ...base, modelId: 'vjepa2-vith', targetClipsPerSec: 100, maxSecPerClip: 3 },
			'h100-sxm'
		).numGpus;
		expect(hi).toBeGreaterThanOrEqual(lo);
	});
});

describe('encoder sizing', () => {
	it('sizes an embedding cluster by docs/sec and reports doc latency', () => {
		const r = sizeForGpu(
			{ ...base, modelId: 'bge-m3', inputTokens: 512, targetDocsPerSec: 500, maxSecPerDoc: 0.5 },
			'h100-sxm'
		);
		expect(r.feasible).toBe(true);
		expect(r.docsPerSec).toBeGreaterThanOrEqual(500);
		expect(r.secPerDoc).toBeLessThanOrEqual(0.5);
	});

	it('a higher docs/sec target needs at least as many GPUs', () => {
		const lo = sizeForGpu(
			{ ...base, modelId: 'bge-m3', inputTokens: 512, targetDocsPerSec: 100, maxSecPerDoc: 0.5 },
			'h100-sxm'
		).numGpus;
		const hi = sizeForGpu(
			{ ...base, modelId: 'bge-m3', inputTokens: 512, targetDocsPerSec: 5000, maxSecPerDoc: 0.5 },
			'h100-sxm'
		).numGpus;
		expect(hi).toBeGreaterThanOrEqual(lo);
	});
});

describe('disaggregated prefill/decode', () => {
	it('sizes independent prefill and decode pools that jointly meet the SLOs', () => {
		const r = sizeDisaggregated({ ...base, modelId: 'llama31-70b', concurrency: 256 }, 'h100-sxm');
		expect(r.feasible).toBe(true);
		expect(r.prefill.gpus).toBeGreaterThan(0);
		expect(r.decode.gpus).toBeGreaterThan(0);
		expect(r.totalGpus).toBe(r.prefill.gpus + r.decode.gpus);
		expect(r.ttftMs).toBeLessThanOrEqual(base.ttftTargetMs);
	});

	it('KV transfer time is included in TTFT and scales with input length', () => {
		const short = sizeDisaggregated({ ...base, inputTokens: 512 }, 'h100-sxm');
		const long = sizeDisaggregated({ ...base, inputTokens: 8192 }, 'h100-sxm');
		expect(long.feasible && short.feasible).toBe(true);
		expect(long.kvTransferMs).toBeGreaterThan(short.kvTransferMs);
	});

	it('a tighter TTFT reports infeasible with a reason (or needs more prefill)', () => {
		const r = sizeDisaggregated({ ...base, ttftTargetMs: 50 }, 'h100-sxm');
		if (!r.feasible) expect(r.reason).toMatch(/prefill|TTFT|KV/);
		else expect(r.ttftMs).toBeLessThanOrEqual(50);
	});

	it('more provisioned concurrency needs at least as many decode replicas', () => {
		const small = sizeDisaggregated({ ...base, concurrency: 64 }, 'h100-sxm');
		const big = sizeDisaggregated({ ...base, concurrency: 512 }, 'h100-sxm');
		if (small.feasible && big.feasible)
			expect(big.decode.replicas).toBeGreaterThanOrEqual(small.decode.replicas);
	});
});

describe('solveWorkload', () => {
	it('ranks GPUs, feasible first then by cluster size', () => {
		const { comparison } = solveWorkload(base);
		expect(comparison.length).toBeGreaterThan(3);
		const feas = comparison.filter((r) => r.feasible);
		// feasible entries are sorted ascending by GPU count
		for (let i = 1; i < feas.length; i++) {
			expect(feas[i].numGpus).toBeGreaterThanOrEqual(feas[i - 1].numGpus);
		}
		// infeasible entries sink to the end
		const firstInfeasible = comparison.findIndex((r) => !r.feasible);
		if (firstInfeasible !== -1) {
			expect(comparison.slice(firstInfeasible).every((r) => !r.feasible)).toBe(true);
		}
	});
});
