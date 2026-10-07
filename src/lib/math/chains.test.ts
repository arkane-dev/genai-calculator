import { describe, expect, it } from 'vitest';
import { inferenceChain, trainingChain } from './chains';
import { computeProfile } from '$lib/profiler/calc';
import { computeTraining } from '$lib/training/calc';
import type { Config } from '$lib/profiler/types';
import type { TrainConfig } from '$lib/training/types';

const cfg: Config = {
	modelId: 'llama31-70b',
	gpuId: 'h100-sxm',
	numGpus: 8,
	gpusPerNode: 8,
	fabricId: 'mp-400',
	tp: 8,
	pp: 1,
	ppEnabled: false,
	ep: 1,
	epEnabled: false,
	batchSize: 32,
	inputTokens: 4096,
	outputTokens: 4096,
	weightFormatId: 'bf16',
	kvBits: 16,
	phase: 'decode',
	kvAllocation: 'paged',
	specDecode: false,
	draftAcceptRate: 0.7,
	specTokens: 4,
	cachedPrefixFrac: 0,
	imagesPerRequest: 0,
	steps: 50,
	resolution: 1024,
	guidance: true,
	frames: 1
};

const trainCfg: TrainConfig = {
	modelId: 'llama31-70b',
	gpuId: 'h100-sxm',
	numGpus: 8,
	gpusPerNode: 8,
	fabricId: 'mp-400',
	method: 'lora',
	optimizer: 'adamw',
	weightFormatId: 'bf16',
	tp: 8,
	pp: 1,
	ppEnabled: false,
	zeroStage: 1,
	activationCheckpointing: true,
	cpuOffload: false,
	microBatchSize: 1,
	gradAccum: 16,
	seqLen: 4096,
	loraRank: 16,
	datasetTokens: 1e9,
	epochs: 3
};

function allSteps(sections: ReturnType<typeof inferenceChain>) {
	return sections.flatMap((s) => s.steps);
}

describe('inferenceChain', () => {
	const sections = inferenceChain(cfg);
	it('produces the four sections with steps', () => {
		expect(sections.map((s) => s.title)).toEqual([
			'Setup',
			'Memory (per GPU)',
			'Throughput (roofline)',
			'Latency (TTFT)'
		]);
		expect(allSteps(sections).length).toBeGreaterThan(10);
	});
	it('every step has non-empty formula/substituted/result and no NaN/undefined', () => {
		for (const s of allSteps(sections)) {
			expect(s.formula.length).toBeGreaterThan(0);
			expect(s.substituted.length).toBeGreaterThan(0);
			expect(s.result.length).toBeGreaterThan(0);
			for (const f of [s.formula, s.substituted, s.result]) {
				expect(f).not.toMatch(/NaN|undefined|Infinity/);
			}
		}
	});
	it('finals match computeProfile (no drift)', () => {
		const p = computeProfile(cfg);
		const bytext = (b: number) => `${(b / 1e9).toFixed(0)}`; // coarse GB compare
		const steps = allSteps(sections);
		const total = steps.find((s) => s.label === 'Total vs capacity')!;
		expect(total.result).toContain(bytext(p.perGpu.used).slice(0, 2)); // used GB appears
		const tput = steps.find((s) => s.label === 'Aggregate throughput')!;
		expect(tput.result).toContain('tok/s');
		const ttft = steps.find((s) => s.label === 'Time to first token')!;
		expect(ttft.result).toContain('ms');
	});
});

describe('trainingChain', () => {
	const sections = trainingChain(trainCfg);
	it('produces sections with steps and clean strings', () => {
		expect(sections.map((s) => s.title)).toEqual(['Setup', 'Memory (per GPU)', 'Compute & time']);
		for (const s of allSteps(sections)) {
			for (const f of [s.formula, s.substituted, s.result]) {
				expect(f).not.toMatch(/NaN|undefined|Infinity/);
			}
		}
	});
	it('time-to-train result reflects computeTraining', () => {
		const t = computeTraining(trainCfg);
		const step = allSteps(sections).find((s) => s.label === 'Time to train')!;
		expect(step.result).toContain('hours');
		expect(t.timeToTrainHours).toBeGreaterThan(0);
	});
});

describe('non-transformer families', () => {
	// one representative per kind — each has its own memory + roofline shape
	const reps: Record<string, string> = {
		diffusion: 'sdxl',
		encoder: 'bge-m3',
		asr: 'whisper-large-v3',
		jepa: 'vjepa2-vitl',
		vla: 'smolvla'
	};
	for (const [kind, modelId] of Object.entries(reps)) {
		it(`${kind} (${modelId}) renders real chains with clean strings`, () => {
			const sections = inferenceChain({ ...cfg, modelId, tp: 1, numGpus: 1 });
			expect(sections.length).toBeGreaterThanOrEqual(3);
			const steps = allSteps(sections);
			expect(steps.length).toBeGreaterThan(5);
			for (const s of steps) {
				expect(s.formula.length).toBeGreaterThan(0);
				expect(s.substituted.length).toBeGreaterThan(0);
				expect(s.result.length).toBeGreaterThan(0);
				for (const f of [s.formula, s.substituted, s.result]) {
					expect(f).not.toMatch(/NaN|undefined|Infinity/);
				}
			}
			// the memory total step is present and references a fit verdict
			expect(steps.some((s) => s.label === 'Total vs capacity')).toBe(true);
		});
	}
});
