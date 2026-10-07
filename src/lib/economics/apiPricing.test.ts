import { describe, it, expect } from 'vitest';
import { apiPriceFor, apiTierPrice, priceSourceFor, MARKET_MEDIAN, BATCH_FRACTION, CACHE_READ_FRACTION } from './apiPricing';

describe('vendor-neutral API pricing', () => {
	it('seeds listed models from the snapshot', () => {
		expect(apiPriceFor('llama31-70b')).toEqual({ inPerM: 0.4, outPerM: 0.4 });
		expect(priceSourceFor('llama31-70b')).toBe('openrouter');
		expect(priceSourceFor('mistral-7b')).toBe('market');
	});

	it('falls back to the market median for unlisted models', () => {
		expect(priceSourceFor('nemotron-h-8b')).toBe('median');
		expect(apiPriceFor('nemotron-h-8b')).toEqual(MARKET_MEDIAN);
		expect(MARKET_MEDIAN.inPerM).toBeGreaterThan(0);
	});

	it('batch uses the listed batch price, else the generic fraction', () => {
		expect(apiTierPrice('gpt-oss-120b', 'batch')).toEqual({ inPerM: 0.03, outPerM: 0.136 });
		const std = apiTierPrice('llama31-70b', 'standard');
		const batch = apiTierPrice('llama31-70b', 'batch');
		expect(batch.inPerM).toBeCloseTo(std.inPerM * BATCH_FRACTION, 9);
		expect(batch.outPerM).toBeCloseTo(std.outPerM * BATCH_FRACTION, 9);
	});

	it('caching discounts input only, using the listed read rate when known', () => {
		const p = apiTierPrice('kimi-k2-5', 'standard', { cachedFrac: 1 });
		expect(p.inPerM).toBeCloseTo(0.07, 9);
		expect(p.outPerM).toBe(2.25);
		const generic = apiTierPrice('llama31-70b', 'standard', { cachedFrac: 0.5 });
		expect(generic.inPerM).toBeCloseTo(0.5 * 0.4 + 0.5 * 0.4 * CACHE_READ_FRACTION, 9);
	});
});
