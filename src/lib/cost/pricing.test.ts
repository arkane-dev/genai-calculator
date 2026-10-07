import { describe, it, expect } from 'vitest';
import { estimateCost, marketRate, defaultPurchasing, DEFAULT_DISCOUNTS } from './pricing';
import { GPUS } from '$lib/profiler/data';

describe('cloud-agnostic GPU pricing', () => {
	it('bills per GPU at the market rate by default', () => {
		const c = estimateCost('h100-sxm', 3, 'on-demand');
		expect(c.billableGpus).toBe(3);
		expect(c.perGpuHour).toBeCloseTo(3.47, 2);
		expect(c.clusterPerHour).toBeCloseTo(3 * 3.47, 2);
	});

	it('rounds up to whole nodes when a node size is set', () => {
		const c = estimateCost('h100-sxm', 9, 'on-demand', { nodeSize: 8 });
		expect(c.billableGpus).toBe(16);
		expect(c.clusterPerHour).toBeCloseTo(16 * 3.47, 2);
	});

	it('applies purchasing discounts, and custom ones override the defaults', () => {
		const od = estimateCost('b200', 8, 'on-demand').clusterPerHour!;
		const committed = estimateCost('b200', 8, 'committed').clusterPerHour!;
		const spot = estimateCost('b200', 8, 'spot', { discounts: { spot: 0.5 } }).clusterPerHour!;
		expect(committed).toBeCloseTo(od * (1 - DEFAULT_DISCOUNTS.committed), 6);
		expect(spot).toBeCloseTo(od * 0.5, 6);
		expect(spot).toBeLessThan(od);
	});

	it('returns no price (with a reason) when there is no market data', () => {
		const c = estimateCost('rtx-pro-4500', 2, 'on-demand');
		expect(c.clusterPerHour).toBeNull();
		expect(c.warning).toMatch(/own rate/);
	});

	it('defaults every GPU to on-demand', () => {
		expect(defaultPurchasing('h100-sxm')).toBe('on-demand');
		expect(defaultPurchasing('t4')).toBe('on-demand');
	});

	it('knows a rate for every GPU in the catalog, or says it has none', () => {
		for (const g of GPUS) {
			const r = marketRate(g.id);
			expect(r === null || r > 0, g.id).toBe(true);
		}
		expect(GPUS.filter((g) => marketRate(g.id) === null).map((g) => g.id)).toEqual(['rtx-pro-4500']);
	});
});
