// Break-even economics: when does it make financial sense to self-host a model
// on your own GPU cluster versus consuming the same model through a per-token API?
//
// The core asymmetry:
//   • Self-hosting buys FIXED capacity. You size a cluster for peak demand and pay
//     for it around the clock, whether it is 5% or 95% used. Cost is flat vs volume.
//   • An API bills per token. Cost is a straight line through the origin — you pay
//     only for what you use, but every token carries the provider's margin.
//
// So the two cost curves cross exactly once. Below the crossover the API is cheaper
// (you would be paying for idle GPUs); above it self-hosting is cheaper (the fixed
// cost is amortised over enough tokens to beat the per-token price). This module
// finds that crossover and evaluates both options at the user's actual utilisation.

const HOURS_PER_MONTH = 730; // 365 × 24 / 12

export interface BreakevenInput {
	clusterPerHourUsd: number; // self-host fixed hourly rate for the sized cluster (billed 24/7)
	clusterTps: number; // peak aggregate output tokens/sec the cluster sustains at full load
	inputTokens: number; // avg input (prompt) tokens per request
	outputTokens: number; // avg output (generation) tokens per request
	apiInPerM: number; // API price, USD per 1M input tokens
	apiOutPerM: number; // API price, USD per 1M output tokens
	dutyCycle: number; // 0..1 — fraction of the month the cluster runs at peak load
	hoursPerMonth?: number;
}

export interface BreakevenResult {
	// costs at the given duty cycle
	selfHostMonthly: number; // fixed: clusterPerHour × hours (independent of duty cycle)
	apiMonthly: number; // linear in volume
	monthlySavings: number; // selfHostMonthly − apiMonthly, signed (positive = self-host costs more)
	cheaper: 'self-host' | 'api' | 'equal';

	// volumes served at the given duty cycle
	reqPerMonth: number;
	inTokensPerMonth: number;
	outTokensPerMonth: number;

	// effective blended $/1M tokens (input + output combined)
	selfHostPerMTokens: number | null; // falls as utilisation rises; null at zero volume
	apiPerMTokens: number; // constant — the blended API rate for this in/out mix
	apiCostPerRequest: number;

	// the crossover
	peakReqPerSec: number;
	breakevenReqPerMonth: number | null; // requests/month where the two costs are equal
	breakevenOutTokensPerMonth: number | null;
	breakevenDutyCycle: number | null; // duty cycle at the crossover; >1 means API wins even flat-out
	// 'api-always' → API cheaper even at 100% duty cycle; 'self-host-always' → self-host cheaper at any volume>0
	regime: 'crossover' | 'api-always' | 'self-host-always';
}

export function breakeven(inp: BreakevenInput): BreakevenResult {
	const hours = inp.hoursPerMonth ?? HOURS_PER_MONTH;
	const duty = Math.max(0, Math.min(1, inp.dutyCycle));
	const secPerMonth = hours * 3600;

	const selfHostMonthly = inp.clusterPerHourUsd * hours;

	// peak request rate the cluster can emit: one request produces `outputTokens`.
	const peakReqPerSec = inp.outputTokens > 0 ? inp.clusterTps / inp.outputTokens : 0;
	const reqPerMonth = peakReqPerSec * secPerMonth * duty;
	const inTokensPerMonth = reqPerMonth * inp.inputTokens;
	const outTokensPerMonth = reqPerMonth * inp.outputTokens;
	const totalTokensPerMonth = inTokensPerMonth + outTokensPerMonth;

	const apiCostPerRequest =
		(inp.inputTokens * inp.apiInPerM + inp.outputTokens * inp.apiOutPerM) / 1e6;
	const apiMonthly = apiCostPerRequest * reqPerMonth;

	const monthlySavings = selfHostMonthly - apiMonthly;
	const cheaper: BreakevenResult['cheaper'] =
		Math.abs(monthlySavings) < 1e-6 ? 'equal' : monthlySavings < 0 ? 'self-host' : 'api';

	// blended API $/1M is constant (weighted by the in/out token mix)
	const apiPerMTokens =
		inp.inputTokens + inp.outputTokens > 0
			? (inp.inputTokens * inp.apiInPerM + inp.outputTokens * inp.apiOutPerM) /
				(inp.inputTokens + inp.outputTokens)
			: 0;
	const selfHostPerMTokens =
		totalTokensPerMonth > 0 ? (selfHostMonthly / totalTokensPerMonth) * 1e6 : null;

	// crossover: selfHostMonthly = apiCostPerRequest × reqPerMonth_be
	let breakevenReqPerMonth: number | null = null;
	let breakevenOutTokensPerMonth: number | null = null;
	let breakevenDutyCycle: number | null = null;
	let regime: BreakevenResult['regime'] = 'crossover';

	if (apiCostPerRequest <= 0) {
		// a free API always wins
		regime = 'api-always';
	} else if (selfHostMonthly <= 0) {
		// free capacity always wins (degenerate)
		regime = 'self-host-always';
	} else {
		breakevenReqPerMonth = selfHostMonthly / apiCostPerRequest;
		breakevenOutTokensPerMonth = breakevenReqPerMonth * inp.outputTokens;
		const maxReqPerMonth = peakReqPerSec * secPerMonth; // duty cycle = 1
		breakevenDutyCycle = maxReqPerMonth > 0 ? breakevenReqPerMonth / maxReqPerMonth : null;
		if (breakevenDutyCycle !== null && breakevenDutyCycle > 1) regime = 'api-always';
	}

	return {
		selfHostMonthly,
		apiMonthly,
		monthlySavings,
		cheaper,
		reqPerMonth,
		inTokensPerMonth,
		outTokensPerMonth,
		selfHostPerMTokens,
		apiPerMTokens,
		apiCostPerRequest,
		peakReqPerSec,
		breakevenReqPerMonth,
		breakevenOutTokensPerMonth,
		breakevenDutyCycle,
		regime
	};
}
