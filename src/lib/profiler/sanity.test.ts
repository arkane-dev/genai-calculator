import { describe, it, expect } from 'vitest';
import { configWarnings } from './sanity';
import { GPUS_BY_ID } from './data';

const h100 = GPUS_BY_ID.get('h100-sxm')!;
const t4 = GPUS_BY_ID.get('t4')!;
const b200 = GPUS_BY_ID.get('b200')!;

describe('configWarnings', () => {
	it('flags tensor parallel that spans nodes', () => {
		const w = configWarnings({
			tp: 8,
			gpusPerNode: 4,
			fabricLabel: 'Cloud VPC Ethernet (no RDMA)',
			gpu: h100,
			formatTier: 'fp16'
		});
		expect(w.some((s) => s.includes('spans nodes'))).toBe(true);
	});

	it('flags FP8 on a GPU without native FP8 tensor cores', () => {
		const w = configWarnings({
			tp: 1,
			gpusPerNode: 8,
			fabricLabel: 'x',
			gpu: t4,
			formatTier: 'fp8'
		});
		expect(w.some((s) => s.includes('FP8'))).toBe(true);
	});

	it('flags FP4 on non-Blackwell (H100 has no native FP4 path)', () => {
		const w = configWarnings({
			tp: 8,
			gpusPerNode: 8,
			fabricLabel: 'x',
			gpu: h100,
			formatTier: 'fp4'
		});
		expect(w.some((s) => s.includes('FP4'))).toBe(true);
	});

	it('is silent on a sane config (B200, in-node TP, native FP4)', () => {
		const w = configWarnings({
			tp: 8,
			gpusPerNode: 8,
			fabricLabel: 'x',
			gpu: b200,
			formatTier: 'fp4'
		});
		expect(w).toEqual([]);
	});
});
