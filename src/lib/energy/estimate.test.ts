import { describe, expect, it } from 'vitest';
import { GPUS_BY_ID } from '$lib/profiler/data';
import { co2ForRun, co2PerMillionTokens, estimateEnergy } from './estimate';

const h100 = GPUS_BY_ID.get('h100-sxm')!;

describe('estimateEnergy', () => {
	it('draws near TDP at high utilization, sits at ~30% idle at zero', () => {
		const idle = estimateEnergy({ gpu: h100, numGpus: 1, util: 0, region: 'us-virginia' });
		const busy = estimateEnergy({ gpu: h100, numGpus: 1, util: 1, region: 'us-virginia' });
		expect(idle.perGpuWatts).toBeCloseTo(h100.tdpWatts * 0.3, 3);
		expect(busy.perGpuWatts).toBeCloseTo(h100.tdpWatts, 3);
	});

	it('applies PUE and host overhead on top of GPU draw', () => {
		const e = estimateEnergy({
			gpu: h100,
			numGpus: 8,
			util: 0.8,
			region: 'us-virginia',
			pue: 1.2,
			hostOverheadFrac: 0.5
		});
		const perGpu = h100.tdpWatts * (0.3 + 0.7 * 0.8); // 3+.56 = 0.86 → 602W
		const raw = (perGpu * (1 + 0.5) * 8) / 1000;
		expect(e.facilityKw).toBeCloseTo(raw * 1.2, 3);
	});

	it('region intensity affects emissions but not power', () => {
		const va = estimateEnergy({ gpu: h100, numGpus: 8, util: 0.8, region: 'us-virginia' });
		const or = estimateEnergy({ gpu: h100, numGpus: 8, util: 0.8, region: 'us-oregon' });
		expect(or.facilityKw).toBeCloseTo(va.facilityKw, 5);
		expect(or.gCo2PerHour).toBeLessThan(va.gCo2PerHour); // Oregon is much cleaner
	});

	it('market-based emissions are far lower than location-based (renewable match)', () => {
		const loc = estimateEnergy({
			gpu: h100,
			numGpus: 8,
			util: 0.8,
			region: 'us-virginia',
			method: 'location'
		});
		const mkt = estimateEnergy({
			gpu: h100,
			numGpus: 8,
			util: 0.8,
			region: 'us-virginia',
			method: 'market'
		});
		expect(mkt.facilityKw).toBeCloseTo(loc.facilityKw, 5);
		expect(mkt.gCo2PerHour).toBeCloseTo(loc.gCo2PerHour * 0.1, 3); // ~90% renewable match
		expect(mkt.method).toBe('market');
	});

	it('cost-per-1M-tokens and per-run scale as expected', () => {
		const e = estimateEnergy({ gpu: h100, numGpus: 8, util: 0.8, region: 'us-virginia' });
		const perMTok = co2PerMillionTokens(e, 5000)!;
		const perRun = co2ForRun(e, 24);
		expect(perMTok).toBeGreaterThan(0);
		expect(perRun).toBeCloseTo((e.gCo2PerHour * 24) / 1000, 3);
	});
});

describe('cloud-agnostic grid inputs', () => {
	const h100 = GPUS_BY_ID.get('h100-sxm')!;
	it('uses a custom grid intensity', () => {
		const e = estimateEnergy({ gpu: h100, numGpus: 1, util: 1, region: 'custom', customGCo2PerKwh: 100 });
		expect(e.gCo2PerKwh).toBe(100);
		expect(e.regionLabel).toBe('Custom grid');
	});
	it('market-based scales with the renewable match you enter', () => {
		const loc = estimateEnergy({ gpu: h100, numGpus: 1, util: 1, region: 'germany' });
		const half = estimateEnergy({ gpu: h100, numGpus: 1, util: 1, region: 'germany', method: 'market', renewableMatch: 0.5 });
		expect(half.gCo2PerKwh).toBeCloseTo(loc.gCo2PerKwh * 0.5, 6);
	});
	it('defaults to the world-average grid and a PUE of 1.2', () => {
		const e = estimateEnergy({ gpu: h100, numGpus: 1, util: 1, region: 'nowhere' });
		expect(e.regionLabel).toBe('World average');
		expect(e.pue).toBe(1.2);
	});
});
