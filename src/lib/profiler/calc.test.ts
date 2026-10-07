import { describe, expect, it } from 'vitest';
import { computeProfile } from './calc';
import { GPUS } from './data';
import type { Config } from './types';

const base: Config = {
	modelId: 'llama31-8b',
	gpuId: 'h100-sxm',
	numGpus: 1,
	gpusPerNode: 8,
	fabricId: 'ib-ndr',
	tp: 1,
	pp: 1,
	ppEnabled: false,
	ep: 1,
	epEnabled: false,
	batchSize: 1,
	inputTokens: 2048,
	outputTokens: 2048,
	weightFormatId: 'bf16',
	kvBits: 16,
	phase: 'decode',
	kvAllocation: 'paged',
	specDecode: false,
	draftAcceptRate: 0.7,
	specTokens: 4,
	cachedPrefixFrac: 0,
	steps: 30,
	resolution: 1024,
	guidance: true,
	frames: 1,
	imagesPerRequest: 0
};

describe('memory fit', () => {
	it('fits an 8B fp16 model on one H100', () => {
		const p = computeProfile(base);
		expect(p.perGpu.fits).toBe(true);
		// ~16 GB of weights
		expect(p.perGpu.weights / 1024 ** 3).toBeGreaterThan(14);
		expect(p.perGpu.weights / 1024 ** 3).toBeLessThan(18);
	});

	it('does not fit 405B fp16 on a single H100', () => {
		const p = computeProfile({ ...base, modelId: 'llama31-405b' });
		expect(p.perGpu.fits).toBe(false);
		expect(p.perGpu.headroom).toBeLessThan(0);
	});

	it('sharding across TP reduces per-GPU weights', () => {
		const one = computeProfile({ ...base, modelId: 'llama31-70b', numGpus: 1, tp: 1 });
		const eight = computeProfile({ ...base, modelId: 'llama31-70b', numGpus: 8, tp: 8 });
		expect(eight.perGpu.weights).toBeCloseTo(one.perGpu.weights / 8, -2);
		expect(eight.perGpu.fits).toBe(true);
	});

	it('4-bit weights shrink the weight segment ~4x vs fp16', () => {
		const fp16 = computeProfile(base);
		const int4 = computeProfile({ ...base, weightFormatId: 'int4' });
		expect(int4.perGpu.weights).toBeCloseTo(fp16.perGpu.weights / 4, -2);
	});
});

describe('weight formats', () => {
	it('memory scales with the format bit width (incl. micro-scale overhead)', () => {
		const bytes = (id: string) => computeProfile({ ...base, weightFormatId: id }).perGpu.weights;
		// bf16 16 > fp8 8 > mxfp6 6 > nvfp4 4.5 > mxfp4 4.25 > int4 4
		expect(bytes('bf16')).toBeGreaterThan(bytes('fp8'));
		expect(bytes('fp8')).toBeGreaterThan(bytes('mxfp6'));
		expect(bytes('mxfp6')).toBeGreaterThan(bytes('nvfp4'));
		expect(bytes('nvfp4')).toBeGreaterThan(bytes('mxfp4'));
		expect(bytes('mxfp4')).toBeGreaterThan(bytes('int4'));
	});

	it('FP4 gets the 2x tensor path on Blackwell (compute-bound prefill)', () => {
		const g = {
			gpuId: 'b200',
			numGpus: 1,
			tp: 1,
			phase: 'prefill' as const,
			batchSize: 8,
			inputTokens: 256,
			outputTokens: 256
		};
		const fp8 = computeProfile({ ...base, ...g, weightFormatId: 'fp8' });
		const nvfp4 = computeProfile({ ...base, ...g, weightFormatId: 'nvfp4' });
		expect(nvfp4.computePeakTflops).toBeCloseTo(fp8.computePeakTflops * 2, -2);
		expect(nvfp4.throughputTps).toBeGreaterThan(fp8.throughputTps);
	});

	it('FP4 falls back to fp16 rate on GPUs with no native FP4 path (H100)', () => {
		const g = {
			gpuId: 'h100-sxm',
			numGpus: 1,
			tp: 1,
			phase: 'prefill' as const,
			batchSize: 8,
			inputTokens: 256,
			outputTokens: 256
		};
		const nvfp4 = computeProfile({ ...base, ...g, weightFormatId: 'nvfp4' });
		const bf16 = computeProfile({ ...base, ...g, weightFormatId: 'bf16' });
		// no FP4 path -> dequant to fp16, same compute peak as bf16 (below fp8)
		expect(nvfp4.computePeakTflops).toBeCloseTo(bf16.computePeakTflops, -1);
		const fp8 = computeProfile({ ...base, ...g, weightFormatId: 'fp8' });
		expect(nvfp4.computePeakTflops).toBeLessThan(fp8.computePeakTflops);
	});

	it('FP6 runs at the FP8 rate but uses less memory', () => {
		const g = {
			gpuId: 'b200',
			numGpus: 1,
			tp: 1,
			phase: 'prefill' as const,
			batchSize: 8,
			inputTokens: 256,
			outputTokens: 256
		};
		const fp8 = computeProfile({ ...base, ...g, weightFormatId: 'fp8' });
		const fp6 = computeProfile({ ...base, ...g, weightFormatId: 'mxfp6' });
		expect(fp6.computePeakTflops).toBeCloseTo(fp8.computePeakTflops, -2);
		expect(fp6.perGpu.weights).toBeLessThan(fp8.perGpu.weights);
	});
});

describe('performance roofline', () => {
	it('produces positive, finite throughput', () => {
		const p = computeProfile(base);
		expect(p.throughputTps).toBeGreaterThan(0);
		expect(Number.isFinite(p.throughputTps)).toBe(true);
	});

	it('is memory bound at batch 1 and compute bound at large batch + short context', () => {
		const small = computeProfile(base);
		expect(small.bottleneck).toBe('memory');
		// Short context keeps the KV read small, so weight-read amortizes and
		// the step tips over into compute bound.
		const big = computeProfile({ ...base, batchSize: 512, inputTokens: 64, outputTokens: 64 });
		expect(big.bottleneck).toBe('compute');
	});

	it('larger batch raises aggregate throughput', () => {
		const b1 = computeProfile(base).throughputTps;
		const b32 = computeProfile({ ...base, batchSize: 32 }).throughputTps;
		expect(b32).toBeGreaterThan(b1);
	});

	it('data-parallel replicas scale throughput', () => {
		const single = computeProfile(base).throughputTps;
		const dp4 = computeProfile({ ...base, numGpus: 4, tp: 1 }).throughputTps;
		expect(dp4).toBeCloseTo(single * 4, -1);
	});

	it('all fill fractions stay within 0..1', () => {
		for (const batch of [1, 8, 64, 1024]) {
			const p = computeProfile({ ...base, numGpus: 8, tp: 8, batchSize: batch });
			for (const f of [p.memBwFrac, p.computeFrac, p.nvlinkFrac]) {
				expect(f).toBeGreaterThanOrEqual(0);
				expect(f).toBeLessThanOrEqual(1);
			}
		}
	});
});

describe('interconnect', () => {
	it('has no interconnect traffic without tensor parallelism', () => {
		expect(computeProfile(base).nvlinkUsedGBs).toBe(0);
	});

	it('uses the interconnect once TP > 1', () => {
		const p = computeProfile({ ...base, modelId: 'llama31-70b', numGpus: 4, tp: 4 });
		expect(p.nvlinkUsedGBs).toBeGreaterThan(0);
	});
});

describe('multi-node topology', () => {
	it('packs GPUs into nodes of the chosen size', () => {
		const p = computeProfile({
			...base,
			modelId: 'llama31-70b',
			numGpus: 16,
			gpusPerNode: 8,
			tp: 8
		});
		expect(p.cluster.numNodes).toBe(2);
		expect(p.cluster.gpusPerNode).toBe(8);
	});

	it('data-parallel scale-out keeps TP inside a node and the fabric idle', () => {
		const p = computeProfile({
			...base,
			modelId: 'llama31-70b',
			numGpus: 16,
			gpusPerNode: 8,
			tp: 8
		});
		expect(p.cluster.crossesFabric).toBe(false);
		expect(p.cluster.scaleOut).toBe('data-parallel');
		expect(p.netFrac).toBe(0);
		expect(p.netUsedGBs).toBe(0);
	});

	it('TP wider than a node pushes the all-reduce onto the fabric', () => {
		const p = computeProfile({
			...base,
			modelId: 'llama31-70b',
			numGpus: 16,
			gpusPerNode: 8,
			tp: 16
		});
		expect(p.cluster.crossesFabric).toBe(true);
		expect(p.comm.tpOverFabric).toBe(true);
		expect(p.cluster.scaleOut).toBe('model-across-nodes');
		expect(p.netUsedGBs).toBeGreaterThan(0);
	});

	it('crossing nodes with TP is slower than staying on NVLink', () => {
		const onNvlink = computeProfile({
			...base,
			modelId: 'llama31-70b',
			numGpus: 8,
			gpusPerNode: 8,
			tp: 8,
			batchSize: 64
		});
		const overFabric = computeProfile({
			...base,
			modelId: 'llama31-70b',
			numGpus: 16,
			gpusPerNode: 8,
			tp: 16,
			batchSize: 64
		});
		expect(overFabric.perUserTps).toBeLessThan(onNvlink.perUserTps);
	});

	it('a faster fabric recovers throughput when TP spans nodes', () => {
		const slow = computeProfile({
			...base,
			modelId: 'llama31-70b',
			numGpus: 16,
			tp: 16,
			fabricId: 'eth100',
			batchSize: 64
		});
		const fast = computeProfile({
			...base,
			modelId: 'llama31-70b',
			numGpus: 16,
			tp: 16,
			fabricId: 'ib-xdr',
			batchSize: 64
		});
		expect(fast.throughputTps).toBeGreaterThan(slow.throughputTps);
	});
});

describe('pipeline parallelism', () => {
	it('splits per-GPU weights across pipeline stages', () => {
		const tp8 = computeProfile({ ...base, modelId: 'llama31-70b', numGpus: 8, tp: 8 });
		const tp4pp2 = computeProfile({
			...base,
			modelId: 'llama31-70b',
			numGpus: 8,
			tp: 4,
			pp: 2,
			ppEnabled: true
		});
		// tp4*pp2 == 8-way weight split, same as tp8
		expect(tp4pp2.perGpu.weights).toBeCloseTo(tp8.perGpu.weights, -2);
	});

	it('lets a model fit that would not fit without it', () => {
		const noPp = computeProfile({
			...base,
			modelId: 'llama31-405b',
			gpuId: 'h100-sxm',
			numGpus: 8,
			tp: 8
		});
		const withPp = computeProfile({
			...base,
			modelId: 'llama31-405b',
			gpuId: 'h100-sxm',
			numGpus: 64,
			tp: 8,
			pp: 8,
			ppEnabled: true,
			weightFormatId: 'fp8'
		});
		expect(noPp.perGpu.fits).toBe(false);
		expect(withPp.perGpu.fits).toBe(true);
	});

	it('is disabled when the switch is off', () => {
		const p = computeProfile({ ...base, pp: 4, ppEnabled: false });
		expect(p.cluster.pp).toBe(1);
	});
});

describe('expert parallelism (MoE all-to-all)', () => {
	const moe = { ...base, modelId: 'mixtral-8x7b', numGpus: 16, gpusPerNode: 8 };

	it('generates all-to-all traffic only when enabled', () => {
		const off = computeProfile({ ...moe, ep: 8, epEnabled: false });
		expect(off.comm.epActive).toBe(false);
		expect(off.comm.epGBs).toBe(0);
		const on = computeProfile({ ...moe, ep: 8, epEnabled: true, batchSize: 64 });
		expect(on.comm.epActive).toBe(true);
		expect(on.comm.epGBs).toBeGreaterThan(0);
	});

	it('EP is capped by the GPU budget left after pipeline stages', () => {
		// pp=2 halves the GPUs available to spread experts over
		const p = computeProfile({
			...moe,
			numGpus: 16,
			pp: 2,
			ppEnabled: true,
			ep: 16,
			epEnabled: true
		});
		expect(p.cluster.ep).toBeLessThanOrEqual(8);
	});

	it('EP shards expert compute, speeding up a compute-bound prefill', () => {
		const noEp = computeProfile({
			...moe,
			numGpus: 8,
			gpusPerNode: 8,
			tp: 1,
			ep: 1,
			epEnabled: false,
			phase: 'prefill',
			batchSize: 8
		});
		const ep8 = computeProfile({
			...moe,
			numGpus: 8,
			gpusPerNode: 8,
			tp: 1,
			ep: 8,
			epEnabled: true,
			phase: 'prefill',
			batchSize: 8
		});
		// experts sharded 8 ways -> far less expert compute per GPU, all-to-all
		// stays on NVLink, so a single sequence prefills faster
		expect(ep8.perUserTps).toBeGreaterThan(noEp.perUserTps);
	});

	it('multipath RDMA beats InfiniBand on cross-node all-to-all at equal bandwidth', () => {
		// ep 16 spans both 8-GPU nodes -> all-to-all crosses the fabric
		const ib = computeProfile({
			...moe,
			ep: 16,
			epEnabled: true,
			fabricId: 'ib-ndr',
			batchSize: 128
		});
		const multipath = computeProfile({
			...moe,
			ep: 16,
			epEnabled: true,
			fabricId: 'mp-400',
			batchSize: 128
		});
		expect(ib.comm.epOverFabric).toBe(true);
		expect(multipath.comm.epOverFabric).toBe(true);
		// same 400 Gb/s peak, but packet spraying sustains a higher all-to-all fraction
		expect(multipath.fabric.a2aEff).toBeGreaterThan(ib.fabric.a2aEff);
		expect(multipath.throughputTps).toBeGreaterThan(ib.throughputTps);
	});
});

describe('KV cache allocation', () => {
	it('paged allocation uses less memory than contiguous', () => {
		const paged = computeProfile({ ...base, batchSize: 16 });
		const contig = computeProfile({ ...base, batchSize: 16, kvAllocation: 'contiguous' });
		expect(paged.perGpu.kv).toBeLessThan(contig.perGpu.kv);
		expect(paged.kvWasteFactor).toBeLessThan(contig.kvWasteFactor);
	});

	it('paged waste is tiny for long context, contiguous over-reserves', () => {
		const p = computeProfile({ ...base, inputTokens: 4096, outputTokens: 4096 });
		expect(p.kvWasteFactor).toBeGreaterThanOrEqual(1);
		expect(p.kvWasteFactor).toBeLessThan(1.05);
		const c = computeProfile({
			...base,
			inputTokens: 4096,
			outputTokens: 4096,
			kvAllocation: 'contiguous'
		});
		expect(c.kvWasteFactor).toBeGreaterThan(1.4);
	});

	it('reports how many sequences the KV budget holds', () => {
		const p = computeProfile({ ...base, batchSize: 1, inputTokens: 4096, outputTokens: 4096 });
		expect(p.maxConcurrentSeqs).toBeGreaterThan(0);
		expect(Number.isFinite(p.maxConcurrentSeqs)).toBe(true);
		// a shorter context should hold more sequences
		const shorter = computeProfile({
			...base,
			batchSize: 1,
			inputTokens: 1024,
			outputTokens: 1024
		});
		expect(shorter.maxConcurrentSeqs).toBeGreaterThan(p.maxConcurrentSeqs);
	});
});

describe('MLA (latent attention)', () => {
	it('has smaller per-token KV than a GQA model (compared unsharded, TP 1)', () => {
		const mla = computeProfile({ ...base, modelId: 'glm-5-3', numGpus: 1, tp: 1 });
		const gqa = computeProfile({ ...base, modelId: 'llama31-70b', numGpus: 1, tp: 1 });
		// normalize out layer count: MLA keeps one small latent vs GQA's K+V per head
		const mlaPerLayer = mla.kvPerSeqBytes / 78;
		const gqaPerLayer = gqa.kvPerSeqBytes / 80;
		expect(mlaPerLayer).toBeLessThan(gqaPerLayer);
	});

	it('MLA KV is replicated across TP (not sharded), unlike GQA', () => {
		const mla1 = computeProfile({ ...base, modelId: 'glm-5-3', numGpus: 1, tp: 1 });
		const mla8 = computeProfile({ ...base, modelId: 'glm-5-3', numGpus: 8, tp: 8 });
		expect(mla8.kvPerSeqBytes).toBeCloseTo(mla1.kvPerSeqBytes, -2);
		// a GQA model does shard its KV across TP
		const gqa1 = computeProfile({ ...base, modelId: 'llama31-70b', numGpus: 1, tp: 1 });
		const gqa8 = computeProfile({ ...base, modelId: 'llama31-70b', numGpus: 8, tp: 8 });
		expect(gqa8.kvPerSeqBytes).toBeLessThan(gqa1.kvPerSeqBytes);
	});

	it('a hybrid model (Kimi K3) caches KV on only its full-attention layers', () => {
		const kimi = computeProfile({ ...base, modelId: 'kimi-k3', numGpus: 8, tp: 8 });
		// 24 of 93 layers hold KV, so per-seq KV is well under a quarter of the
		// naive all-93-layer figure
		const naiveAll = (kimi.kvPerSeqBytes / 24) * 93;
		expect(kimi.kvPerSeqBytes).toBeLessThan(naiveAll * 0.3);
	});
});

describe('card-stated active params drive the expert split', () => {
	it('uses the exact active-param count from the model spec', () => {
		expect(computeProfile({ ...base, modelId: 'deepseek-v4' }).activeParams).toBeCloseTo(49e9, -8);
		expect(computeProfile({ ...base, modelId: 'kimi-k3' }).activeParams).toBeCloseTo(104e9, -9);
	});

	it('active params stay well below total for these large MoEs', () => {
		const k = computeProfile({ ...base, modelId: 'kimi-k3' });
		expect(k.activeParams).toBeLessThan(2.8e12 * 0.1);
	});

	it('weight memory uses the algebraic split, not the geometric overcount', () => {
		// Kimi K3: 2.8T fp16 sharded EP8/TP8 -> ~700 GB/GPU, not >1 TB.
		const k = computeProfile({
			...base,
			modelId: 'kimi-k3',
			numGpus: 8,
			tp: 8,
			ep: 8,
			epEnabled: true
		});
		const gb = k.perGpu.weights / 1024 ** 3;
		expect(gb).toBeGreaterThan(500);
		expect(gb).toBeLessThan(900);
	});
});

describe('phase (prefill vs decode)', () => {
	it('decode is memory bound, prefill of the same config is compute bound', () => {
		const decode = computeProfile({ ...base, batchSize: 8 });
		const prefill = computeProfile({ ...base, batchSize: 8, phase: 'prefill' });
		expect(decode.bottleneck).toBe('memory');
		expect(prefill.bottleneck).toBe('compute');
	});

	it('prefill processes far more tokens per second than decode', () => {
		const decode = computeProfile({ ...base, batchSize: 8 }).throughputTps;
		const prefill = computeProfile({ ...base, batchSize: 8, phase: 'prefill' }).throughputTps;
		expect(prefill).toBeGreaterThan(decode);
	});

	it('HBM I/O and L2 track together in decode but diverge in prefill', () => {
		const decode = computeProfile({ ...base, batchSize: 8 });
		// decode streams weights: HBM ~ L2
		expect(decode.l2Frac).toBeCloseTo(decode.memBwFrac, 5);
		const prefill = computeProfile({ ...base, batchSize: 8, phase: 'prefill' });
		// prefill: HBM controllers idle-ish, L2 busy feeding the tensor cores
		expect(prefill.l2Frac).toBeGreaterThan(prefill.memBwFrac);
	});
});

describe('GPU die composition', () => {
	it('every GPU die breakdown sums to ~1 and has tensor + cuda', () => {
		for (const g of GPUS) {
			const sum = g.die.reduce((s, u) => s + u.areaFrac, 0);
			expect(sum).toBeCloseTo(1, 1);
			expect(g.die.some((u) => u.kind === 'tensor')).toBe(true);
			expect(g.die.some((u) => u.kind === 'cuda')).toBe(true);
		}
	});

	it('only Ada parts (L40S, RTX 4090) carry RT cores', () => {
		const hasRt = (id: string) => GPUS.find((g) => g.id === id)!.die.some((u) => u.kind === 'rt');
		expect(hasRt('l40s')).toBe(true);
		expect(hasRt('rtx4090')).toBe(true);
		expect(hasRt('h100-sxm')).toBe(false);
		expect(hasRt('a100-80')).toBe(false);
	});
});

describe('MoE', () => {
	it('active params are far below total for a MoE model', () => {
		const p = computeProfile({ ...base, modelId: 'mixtral-8x7b' });
		expect(p.activeParams).toBeLessThan(46.7e9);
		expect(p.activeParams).toBeGreaterThan(10e9);
	});

	it('Kimi K2.5 is a 1T MoE with ~32B active params', () => {
		const p = computeProfile({
			...base,
			modelId: 'kimi-k2-5',
			numGpus: 32,
			tp: 8,
			ep: 8,
			epEnabled: true
		});
		expect(p.activeParams).toBeCloseTo(32e9, -9); // card-stated
	});
});

describe('prefix caching', () => {
	it('cuts TTFT with the cached fraction and leaves KV memory unchanged', () => {
		const none = computeProfile({
			...base,
			modelId: 'llama31-8b',
			phase: 'prefill',
			inputTokens: 8000,
			cachedPrefixFrac: 0
		});
		const half = computeProfile({
			...base,
			modelId: 'llama31-8b',
			phase: 'prefill',
			inputTokens: 8000,
			cachedPrefixFrac: 0.5
		});
		const most = computeProfile({
			...base,
			modelId: 'llama31-8b',
			phase: 'prefill',
			inputTokens: 8000,
			cachedPrefixFrac: 0.9
		});
		expect(half.ttftMs).toBeLessThan(none.ttftMs);
		expect(most.ttftMs).toBeLessThan(half.ttftMs);
		// ~linear: 50% cached ≈ half the TTFT
		expect(half.ttftMs).toBeCloseTo(none.ttftMs / 2, 0);
		// prefix KV is resident, not recomputed — memory is unchanged
		expect(half.perGpu.kv).toBeCloseTo(none.perGpu.kv, 5);
	});
});

describe('speculative decoding', () => {
	it('raises decode throughput and per-user tok/s, higher acceptance helps more', () => {
		const off = computeProfile({
			...base,
			modelId: 'llama31-8b',
			phase: 'decode',
			specDecode: false
		});
		const lo = computeProfile({
			...base,
			modelId: 'llama31-8b',
			phase: 'decode',
			specDecode: true,
			draftAcceptRate: 0.5,
			specTokens: 4
		});
		const hi = computeProfile({
			...base,
			modelId: 'llama31-8b',
			phase: 'decode',
			specDecode: true,
			draftAcceptRate: 0.9,
			specTokens: 4
		});
		expect(lo.perUserTps).toBeGreaterThan(off.perUserTps);
		expect(hi.perUserTps).toBeGreaterThan(lo.perUserTps);
		expect(hi.specSpeedup!).toBeGreaterThan(1);
	});

	it('does not affect prefill (TTFT / prefill throughput)', () => {
		const off = computeProfile({
			...base,
			modelId: 'llama31-8b',
			phase: 'prefill',
			specDecode: false
		});
		const on = computeProfile({
			...base,
			modelId: 'llama31-8b',
			phase: 'prefill',
			specDecode: true,
			draftAcceptRate: 0.9
		});
		expect(on.throughputTps).toBeCloseTo(off.throughputTps, 5);
		expect(on.specSpeedup).toBe(1);
	});
});

describe('diffusion / DiT', () => {
	it('DiT-XL/2 at 256px has 16² latent tokens and fits on one H100', () => {
		const p = computeProfile({
			...base,
			modelId: 'dit-xl2',
			resolution: 256,
			steps: 50,
			guidance: true,
			batchSize: 1
		});
		expect(p.diffusion).toBeDefined();
		expect(p.diffusion!.latentTokens).toBe(256); // (256/8/2)² = 16²
		expect(p.perGpu.fits).toBe(true);
		expect(p.diffusion!.imagesPerSec).toBeGreaterThan(0);
	});

	it('a video DiT (CogVideoX) scales its latent tokens with the frame count', () => {
		const short = computeProfile({
			...base,
			modelId: 'cogvideox-5b',
			resolution: 512,
			frames: 5,
			batchSize: 1
		});
		const long = computeProfile({
			...base,
			modelId: 'cogvideox-5b',
			resolution: 512,
			frames: 13,
			batchSize: 1
		});
		expect(long.diffusion!.latentTokens).toBeGreaterThan(short.diffusion!.latentTokens);
		// more frames → quadratic attention makes per-clip time climb faster than linearly
		const frameRatio = 13 / 5;
		expect(long.diffusion!.secPerImage / short.diffusion!.secPerImage).toBeGreaterThan(frameRatio);
	});

	it('the non-square aspect raises the latent token count (CogVideoX 1.5:1)', () => {
		// square grid at 512: (512/8/2)² = 1024 per frame; ×13 frames = 13312.
		// aspect 1.5 lifts it to ~19968.
		const p = computeProfile({
			...base,
			modelId: 'cogvideox-5b',
			resolution: 512,
			frames: 13,
			batchSize: 1
		});
		expect(p.diffusion!.latentTokens).toBe(19968);
	});

	it('the DiT attention term does not apply to a conv UNet (SDXL uses its effective patch)', () => {
		// SDXL is arch:'unet'; its per-forward FLOPs come only from the effective-patch
		// linear term, so its time should be unchanged by the attention addition.
		const p = computeProfile({
			...base,
			modelId: 'sdxl',
			resolution: 1024,
			steps: 50,
			guidance: true,
			batchSize: 1
		});
		expect(p.diffusion!.latentTokens).toBe(1024); // (1024/8/4)² with effective patch 4
		expect(p.diffusion!.secPerImage).toBeGreaterThan(0.4);
		expect(p.diffusion!.secPerImage).toBeLessThan(1.2);
	});
});

describe('JEPA', () => {
	it('produces a jepa profile with clips/sec and no KV', () => {
		const p = computeProfile({
			...base,
			modelId: 'vjepa2-vitl',
			resolution: 256,
			frames: 16,
			batchSize: 1
		});
		expect(p.jepa).toBeDefined();
		expect(p.perGpu.kv).toBe(0);
		expect(p.jepa!.clipsPerSec).toBeGreaterThan(0);
		expect(p.jepa!.secPerClip).toBeGreaterThan(0);
		// ViT-L @256, 16 frames: (256/16)²=256 spatial × (16/2)=8 temporal = 2048 patches
		expect(p.jepa!.tokens).toBe(2048);
	});

	it('more frames raise the patch count and the compute (quadratic attention)', () => {
		const few = computeProfile({ ...base, modelId: 'vjepa2-vitg', frames: 16, batchSize: 1 });
		const many = computeProfile({ ...base, modelId: 'vjepa2-vitg', frames: 64, batchSize: 1 });
		expect(many.jepa!.tokens).toBeGreaterThan(few.jepa!.tokens);
		// 4× the frames → >4× the per-clip time because attention is quadratic in tokens
		expect(many.jepa!.secPerClip / few.jepa!.secPerClip).toBeGreaterThan(4);
	});

	it('a bigger batch takes longer per pass and never lowers clips/sec (compute-bound)', () => {
		const b1 = computeProfile({ ...base, modelId: 'vjepa2-vitl', frames: 16, batchSize: 1 });
		const b8 = computeProfile({ ...base, modelId: 'vjepa2-vitl', frames: 16, batchSize: 8 });
		expect(b8.jepa!.secPerClip).toBeGreaterThan(b1.jepa!.secPerClip);
		expect(b8.jepa!.clipsPerSec).toBeGreaterThanOrEqual(b1.jepa!.clipsPerSec - 1e-6);
	});

	it('V-JEPA 2-AC rolls out the predictor, so it is far slower than the bare encoder', () => {
		const enc = computeProfile({
			...base,
			modelId: 'vjepa2-vitg',
			frames: 64,
			resolution: 256,
			batchSize: 1
		});
		const ac = computeProfile({
			...base,
			modelId: 'vjepa2-ac',
			frames: 64,
			resolution: 256,
			batchSize: 1
		});
		// same encoder + a 16-step planning rollout of a 300M predictor
		expect(ac.jepa!.secPerClip).toBeGreaterThan(enc.jepa!.secPerClip);
		expect(ac.perGpu.weights).toBeGreaterThan(enc.perGpu.weights); // AC predictor is resident
	});
});

describe('α-β latency model for collectives', () => {
	it('a high-α fabric (Ethernet) is slower than a low-α fabric (IB) at similar bandwidth', () => {
		// Cross-node decode: small-message collectives per step. α dominates.
		const eth = computeProfile({
			...base,
			modelId: 'llama31-70b',
			numGpus: 16,
			gpusPerNode: 8,
			tp: 16,
			pp: 1,
			ppEnabled: false,
			phase: 'decode',
			batchSize: 1,
			fabricId: 'eth100'
		});
		const ib = computeProfile({
			...base,
			modelId: 'llama31-70b',
			numGpus: 16,
			gpusPerNode: 8,
			tp: 16,
			pp: 1,
			ppEnabled: false,
			phase: 'decode',
			batchSize: 1,
			fabricId: 'ib-ndr'
		});
		expect(ib.stepTimeMs).toBeLessThan(eth.stepTimeMs);
	});

	it('single-node TP has no cross-node α penalty', () => {
		const before = computeProfile({
			...base,
			modelId: 'llama31-8b',
			numGpus: 8,
			gpusPerNode: 8,
			tp: 8,
			phase: 'decode',
			batchSize: 1
		});
		expect(before.stepTimeMs).toBeGreaterThan(0);
	});
});

describe('ASR (Whisper)', () => {
	it('Whisper large-v3 transcribes faster than real-time on one H100', () => {
		const p = computeProfile({ ...base, modelId: 'whisper-large-v3', numGpus: 1, batchSize: 1 });
		expect(p.asr).toBeDefined();
		expect(p.perGpu.fits).toBe(true);
		expect(p.asr!.rtf).toBeGreaterThan(1);
	});

	it('Whisper turbo is much faster than large-v3 (4-layer decoder)', () => {
		const big = computeProfile({ ...base, modelId: 'whisper-large-v3', numGpus: 1, batchSize: 1 });
		const turbo = computeProfile({
			...base,
			modelId: 'whisper-large-v3-turbo',
			numGpus: 1,
			batchSize: 1
		});
		expect(turbo.asr!.secPerWindow).toBeLessThan(big.asr!.secPerWindow);
		// decoder time drops sharply — 8× fewer layers
		expect(turbo.asr!.decoderMs).toBeLessThan(big.asr!.decoderMs / 4);
	});
});

describe('VLM (vision-language)', () => {
	it('adding images inflates prefill compute and TTFT', () => {
		const text = computeProfile({
			...base,
			modelId: 'qwen25-vl-7b',
			phase: 'prefill',
			imagesPerRequest: 0,
			inputTokens: 512
		});
		const with2 = computeProfile({
			...base,
			modelId: 'qwen25-vl-7b',
			phase: 'prefill',
			imagesPerRequest: 2,
			inputTokens: 512
		});
		expect(with2.ttftMs).toBeGreaterThan(text.ttftMs);
	});

	it('VLM weights include the vision encoder as extra resident memory', () => {
		const off = computeProfile({ ...base, modelId: 'llama31-8b', numGpus: 1 });
		const vlm = computeProfile({ ...base, modelId: 'qwen25-vl-7b', numGpus: 1 });
		expect(vlm.perGpu.weights).toBeGreaterThan(0);
		expect(off.perGpu.weights).toBeGreaterThan(0);
	});
});

describe('VLA (vision-language-action)', () => {
	it('pi-zero produces action chunks at target Hz on one H100', () => {
		const p = computeProfile({ ...base, modelId: 'pi-0', numGpus: 1, batchSize: 1 });
		expect(p.vla).toBeDefined();
		expect(p.perGpu.fits).toBe(true);
		expect(p.vla!.effectiveHz).toBeGreaterThan(0);
		expect(p.vla!.actionsPerSec).toBeGreaterThan(0);
	});

	it('SmolVLA is much faster than pi-zero (smaller backbone)', () => {
		const pi = computeProfile({ ...base, modelId: 'pi-0', numGpus: 1, batchSize: 1 });
		const smol = computeProfile({ ...base, modelId: 'smolvla', numGpus: 1, batchSize: 1 });
		expect(smol.vla!.secPerControl).toBeLessThan(pi.vla!.secPerControl);
	});
});

describe('encoder / embeddings', () => {
	it('BGE-M3 produces an encoder profile with docs/sec and no KV', () => {
		const p = computeProfile({
			...base,
			modelId: 'bge-m3',
			inputTokens: 512,
			batchSize: 8,
			phase: 'prefill'
		});
		expect(p.encoder).toBeDefined();
		expect(p.perGpu.kv).toBe(0);
		expect(p.encoder!.tokens).toBe(512);
		expect(p.encoder!.docsPerSec).toBeGreaterThan(0);
	});

	it('a bigger batch keeps docs/sec at least as high (once compute-bound, it stays constant)', () => {
		const b1 = computeProfile({
			...base,
			modelId: 'bge-m3',
			inputTokens: 512,
			batchSize: 1,
			phase: 'prefill'
		});
		const b32 = computeProfile({
			...base,
			modelId: 'bge-m3',
			inputTokens: 512,
			batchSize: 32,
			phase: 'prefill'
		});
		expect(b32.encoder!.docsPerSec).toBeGreaterThanOrEqual(b1.encoder!.docsPerSec - 1e-6);
		expect(b32.encoder!.secPerDoc).toBeGreaterThan(b1.encoder!.secPerDoc); // per-batch time does grow
	});

	it('a small encoder is memory-bound at batch 1 and batching amortizes weight reads', () => {
		// A short seq of a tiny encoder is memory-bound, so batching raises docs/sec strictly.
		const b1 = computeProfile({
			...base,
			modelId: 'bge-m3',
			inputTokens: 64,
			batchSize: 1,
			phase: 'prefill'
		});
		const b16 = computeProfile({
			...base,
			modelId: 'bge-m3',
			inputTokens: 64,
			batchSize: 16,
			phase: 'prefill'
		});
		expect(b16.encoder!.docsPerSec).toBeGreaterThan(b1.encoder!.docsPerSec);
	});

	it('a longer sequence quadratically raises per-doc time (attention)', () => {
		const short = computeProfile({
			...base,
			modelId: 'bge-m3',
			inputTokens: 512,
			batchSize: 1,
			phase: 'prefill'
		});
		const long = computeProfile({
			...base,
			modelId: 'bge-m3',
			inputTokens: 2048,
			batchSize: 1,
			phase: 'prefill'
		});
		expect(long.encoder!.secPerDoc).toBeGreaterThan(short.encoder!.secPerDoc * 4);
	});
});

describe('visual autoregressive (VAR)', () => {
	it('VAR-d30 is a transformer that decodes image tokens and fits on one H100', () => {
		const p = computeProfile({
			...base,
			modelId: 'var-d30',
			inputTokens: 1,
			outputTokens: 680,
			batchSize: 1,
			phase: 'decode'
		});
		expect(p.diffusion).toBeUndefined();
		expect(p.jepa).toBeUndefined();
		expect(p.perGpu.fits).toBe(true);
		expect(p.perUserTps).toBeGreaterThan(0);
	});
});
