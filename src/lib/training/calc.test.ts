import { describe, expect, it } from 'vitest';
import { computeTraining } from './calc';
import type { TrainConfig } from './types';

const base: TrainConfig = {
	modelId: 'llama31-8b',
	gpuId: 'h100-sxm',
	numGpus: 1,
	gpusPerNode: 8,
	fabricId: 'ib-ndr',
	method: 'full',
	optimizer: 'adamw',
	weightFormatId: 'bf16',
	tp: 1,
	pp: 1,
	ppEnabled: false,
	zeroStage: 0,
	activationCheckpointing: true,
	cpuOffload: false,
	microBatchSize: 1,
	gradAccum: 8,
	seqLen: 2048,
	loraRank: 16,
	datasetTokens: 1e9,
	epochs: 1
};

describe('training memory', () => {
	it('full fine-tuning an 8B (16 B/param) overflows one H100, but LoRA fits', () => {
		const full = computeTraining(base);
		expect(full.perGpu.fits).toBe(false);
		const lora = computeTraining({ ...base, method: 'lora' });
		expect(lora.perGpu.fits).toBe(true);
		expect(lora.trainableParams).toBeLessThan(0.05 * lora.baseParams);
	});

	it('QLoRA uses less memory than LoRA, which uses less than full', () => {
		const full = computeTraining(base).perGpu.used;
		const lora = computeTraining({ ...base, method: 'lora' }).perGpu.used;
		const qlora = computeTraining({ ...base, method: 'qlora' }).perGpu.used;
		expect(qlora).toBeLessThan(lora);
		expect(lora).toBeLessThan(full);
	});

	it('optimizer states dominate full-FT model states (12 of 16 bytes/param)', () => {
		const p = computeTraining(base);
		expect(p.perGpu.optimizer).toBeGreaterThan(p.perGpu.weights);
		expect(p.perGpu.optimizer).toBeGreaterThan(p.perGpu.gradients);
	});

	it('ZeRO-3 shards model states across DP, cutting per-GPU memory', () => {
		// 8 GPUs, all data-parallel (tp=pp=1 → dp=8)
		const z0 = computeTraining({ ...base, numGpus: 8 });
		const z3 = computeTraining({ ...base, numGpus: 8, zeroStage: 3 });
		expect(z3.perGpu.optimizer).toBeLessThan(z0.perGpu.optimizer);
		expect(z3.perGpu.used).toBeLessThan(z0.perGpu.used);
	});

	it('CPU offload drops optimizer memory on-device to zero and adds a PCIe step term', () => {
		const on = computeTraining({ ...base, cpuOffload: true });
		const off = computeTraining({ ...base, cpuOffload: false });
		expect(on.perGpu.optimizer).toBe(0);
		expect(off.perGpu.optimizer).toBeGreaterThan(0);
		expect(on.stepTimeMs).toBeGreaterThan(off.stepTimeMs); // PCIe shuttle adds time
	});

	it('activation checkpointing lowers activation memory', () => {
		const on = computeTraining({ ...base, activationCheckpointing: true }).perGpu.activations;
		const off = computeTraining({ ...base, activationCheckpointing: false }).perGpu.activations;
		expect(on).toBeLessThan(off);
	});
});

describe('training throughput', () => {
	it('more data-parallel GPUs raise tokens/sec and cut time-to-train', () => {
		const small = computeTraining({ ...base, method: 'lora', numGpus: 1 });
		const big = computeTraining({ ...base, method: 'lora', numGpus: 8, zeroStage: 1 });
		expect(big.dp).toBe(8);
		expect(big.tokensPerSec).toBeGreaterThan(small.tokensPerSec);
		expect(big.timeToTrainHours).toBeLessThan(small.timeToTrainHours);
	});

	it('reports a finite step time, MFU in range, and a global batch', () => {
		const p = computeTraining({ ...base, method: 'lora' });
		expect(p.stepTimeMs).toBeGreaterThan(0);
		expect(p.mfu).toBeGreaterThan(0);
		expect(p.mfu).toBeLessThanOrEqual(0.45);
		expect(p.globalBatchTokens).toBe(1 * 2048 * 8 * 1);
	});
});
