import { describe, expect, it } from 'vitest';
import { breakeven, type BreakevenInput } from './breakeven';

const base: BreakevenInput = {
	clusterPerHourUsd: 40, // ~ one 8-GPU H100-class node
	clusterTps: 4000, // aggregate output tok/s at peak
	inputTokens: 1000,
	outputTokens: 1000,
	apiInPerM: 3, // $/1M in
	apiOutPerM: 15, // $/1M out
	dutyCycle: 0.5
};

describe('breakeven', () => {
	it('self-host cost is fixed regardless of duty cycle', () => {
		const lo = breakeven({ ...base, dutyCycle: 0.1 });
		const hi = breakeven({ ...base, dutyCycle: 0.9 });
		expect(lo.selfHostMonthly).toBeCloseTo(hi.selfHostMonthly, 6);
		expect(lo.selfHostMonthly).toBeCloseTo(40 * 730, 6);
	});

	it('API cost scales linearly with duty cycle (volume)', () => {
		const lo = breakeven({ ...base, dutyCycle: 0.2 });
		const hi = breakeven({ ...base, dutyCycle: 0.4 });
		expect(hi.apiMonthly).toBeCloseTo(2 * lo.apiMonthly, 4);
	});

	it('effective self-host $/1M falls as utilisation rises; API $/1M is constant', () => {
		const lo = breakeven({ ...base, dutyCycle: 0.2 });
		const hi = breakeven({ ...base, dutyCycle: 0.8 });
		expect(hi.selfHostPerMTokens!).toBeLessThan(lo.selfHostPerMTokens!);
		expect(hi.apiPerMTokens).toBeCloseTo(lo.apiPerMTokens, 6);
		// blended rate of a 50/50 in/out mix at $3 / $15
		expect(hi.apiPerMTokens).toBeCloseTo(9, 6);
	});

	it('finds a crossover and reports which side wins around it', () => {
		const r = breakeven(base);
		expect(r.regime).toBe('crossover');
		expect(r.breakevenReqPerMonth).toBeGreaterThan(0);
		// below the crossover duty cycle the API wins, above it self-host wins
		const below = breakeven({ ...base, dutyCycle: r.breakevenDutyCycle! * 0.5 });
		const above = breakeven({ ...base, dutyCycle: Math.min(1, r.breakevenDutyCycle! * 1.5) });
		expect(below.cheaper).toBe('api');
		expect(above.cheaper).toBe('self-host');
	});

	it('at the crossover duty cycle the two costs are equal', () => {
		const r = breakeven(base);
		const at = breakeven({ ...base, dutyCycle: r.breakevenDutyCycle! });
		expect(at.apiMonthly).toBeCloseTo(at.selfHostMonthly, 2);
	});

	it('flags api-always when even flat-out self-hosting is dearer', () => {
		// tiny cluster throughput but pricey box vs a cheap API → API wins at any utilisation
		const r = breakeven({ ...base, apiInPerM: 0.05, apiOutPerM: 0.1, clusterPerHourUsd: 100 });
		expect(r.regime).toBe('api-always');
		expect(r.breakevenDutyCycle!).toBeGreaterThan(1);
	});

	it('handles a free API (never worth self-hosting)', () => {
		const r = breakeven({ ...base, apiInPerM: 0, apiOutPerM: 0 });
		expect(r.regime).toBe('api-always');
		expect(r.apiMonthly).toBe(0);
	});
});
