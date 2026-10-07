// Per-model API $/1M-token prices, used to seed the API side of the self-host vs API
// comparison. Vendor-neutral sources, in preference order:
//   - OpenRouter market price (openrouter.ai/api/v1/models), snapshot 2026-10-07.
//     OpenRouter routes to many providers, so this is a fair "what an API costs" figure.
//   - Typical market price (provider ranges, midpoint) for models OpenRouter no longer
//     lists.
// Models on neither fall back to the median of the listed prices.
// Prices change often. Every field in the tool is editable, so enter a real quote
// when you have one.

export const PRICE_SNAPSHOT = '2026-10-07';

export type PriceSource = 'openrouter' | 'market' | 'median';

// Two ways to buy the same tokens:
//   standard: the normal per-token rate
//   batch:    asynchronous bulk jobs. Most large API providers charge about 50% of
//             standard; where OpenRouter lists a batch price we use it.
// Prompt caching is a modifier, not a tier: cached input tokens are billed at a
// lower read rate. Where OpenRouter lists that rate we use it, else 10% of input,
// a common figure across providers.
export type ApiTier = 'standard' | 'batch';
export const API_TIERS: ApiTier[] = ['standard', 'batch'];
export const BATCH_FRACTION = 0.5;
export const CACHE_READ_FRACTION = 0.1;

interface Entry {
	inPerM: number;
	outPerM: number;
	src: PriceSource;
	batchInPerM?: number;
	batchOutPerM?: number;
	cacheReadPerM?: number; // price for a cached (re-read) input token
}

const PRICES: Record<string, Entry> = {
	// OpenRouter market price, 2026-10-07
	'llama32-3b': { inPerM: 0.05, outPerM: 0.33, src: 'openrouter' },
	'llama31-8b': { inPerM: 0.05, outPerM: 0.08, cacheReadPerM: 0.025, src: 'openrouter' },
	'llama31-70b': { inPerM: 0.4, outPerM: 0.4, src: 'openrouter' },
	'llama31-405b': { inPerM: 1.0, outPerM: 1.0, src: 'openrouter' }, // 405B-class (Hermes-3 405B)
	'gpt-oss-120b': { inPerM: 0.037, outPerM: 0.17, batchInPerM: 0.03, batchOutPerM: 0.136, src: 'openrouter' },
	'qwen25-72b': { inPerM: 0.36, outPerM: 0.4, src: 'openrouter' },
	'qwen38-flash': { inPerM: 0.117, outPerM: 0.455, src: 'openrouter' }, // Qwen3 8B
	'deepseek-v4': { inPerM: 0.209, outPerM: 0.418, cacheReadPerM: 0.017, src: 'openrouter' }, // V4 Pro
	'deepseek-v4-flash': { inPerM: 0.03, outPerM: 1.28, cacheReadPerM: 0.03, src: 'openrouter' },
	'deepseek-v4-1-flash': { inPerM: 0.05, outPerM: 1.2, cacheReadPerM: 0.02, src: 'openrouter' },
	'minimax-m3': { inPerM: 0.3, outPerM: 1.2, cacheReadPerM: 0.06, src: 'openrouter' },
	'kimi-k3': { inPerM: 0.62, outPerM: 15.0, cacheReadPerM: 0.43, src: 'openrouter' },
	'kimi-k2-5': { inPerM: 0.45, outPerM: 2.25, cacheReadPerM: 0.07, src: 'openrouter' },
	'glm-5-2': { inPerM: 0.171, outPerM: 7.2, cacheReadPerM: 0.162, src: 'openrouter' },
	'glm-5-3': { inPerM: 0.07, outPerM: 7.0, cacheReadPerM: 0.065, src: 'openrouter' },
	'glm-5-3-flash': { inPerM: 0.15, outPerM: 0.5, batchInPerM: 0.06, batchOutPerM: 0.2, cacheReadPerM: 0.03, src: 'openrouter' },
	'mimo-v2-5': { inPerM: 0.14, outPerM: 0.28, cacheReadPerM: 0.003, src: 'openrouter' },
	// Typical market prices (provider ranges; midpoint where a range was given)
	'mistral-7b': { inPerM: 0.15, outPerM: 0.2, src: 'market' },
	'mixtral-8x7b': { inPerM: 0.45, outPerM: 0.7, src: 'market' },
	'qwen25-14b': { inPerM: 0.35, outPerM: 1.4, src: 'market' },
	'r1-distill-qwen-1_5b': { inPerM: 0.225, outPerM: 0.225, src: 'market' },
	'r1-distill-qwen-7b': { inPerM: 0.175, outPerM: 0.175, src: 'market' },
	'r1-distill-qwen-32b': { inPerM: 0.525, outPerM: 0.525, src: 'market' },
	'qwq-32b': { inPerM: 0.675, outPerM: 0.825, src: 'market' },
	hy3: { inPerM: 0.18, outPerM: 0.6, src: 'market' },
	'nemotron-nano-8b': { inPerM: 0.04, outPerM: 0.16, src: 'market' },
	'nemotron-nano-9b-v2': { inPerM: 0.05, outPerM: 0.205, src: 'market' },
	'nemotron-nano-12b-v2': { inPerM: 0.2, outPerM: 0.2, src: 'market' },
	'nemotron-h-47b': { inPerM: 0.2, outPerM: 0.6, src: 'market' },
	'nemotron-h-56b': { inPerM: 0.275, outPerM: 1.2, src: 'market' },
	'nemotron-super-49b': { inPerM: 0.1, outPerM: 0.4, src: 'market' },
	'nemotron-ultra-253b': { inPerM: 0.6, outPerM: 1.8, src: 'market' },
	'qwen25-vl-7b': { inPerM: 0.2, outPerM: 0.2, src: 'market' },
	'pixtral-12b': { inPerM: 0.15, outPerM: 0.15, src: 'market' }
	// Not offered via any API (nemotron-h-8b, llava-ov-7b) → median fallback
};

const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	const m = Math.floor(s.length / 2);
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Median input/output price across the listed models: the fallback for unlisted ones. */
export const MARKET_MEDIAN = {
	inPerM: median(Object.values(PRICES).map((e) => e.inPerM)),
	outPerM: median(Object.values(PRICES).map((e) => e.outPerM))
};

/** Where a model's seeded price comes from: 'median' when it isn't individually listed. */
export function priceSourceFor(modelId: string): PriceSource {
	return PRICES[modelId]?.src ?? 'median';
}

/** Seed price for a model, or the market median when it isn't individually listed. */
export function apiPriceFor(modelId: string): { inPerM: number; outPerM: number } {
	const e = PRICES[modelId];
	return e ? { inPerM: e.inPerM, outPerM: e.outPerM } : { ...MARKET_MEDIAN };
}

/**
 * Effective $/1M for a model at a tier, optionally with prompt caching over a
 * `cachedFrac` of the input. Batch = listed batch rate if known, else BATCH_FRACTION of
 * standard. Caching bills the cached input fraction at the read rate (listed, else
 * CACHE_READ_FRACTION of the input price).
 */
export function apiTierPrice(
	modelId: string,
	tier: ApiTier,
	opts: { cachedFrac?: number } = {}
): { inPerM: number; outPerM: number } {
	const e: Entry = PRICES[modelId] ?? { ...MARKET_MEDIAN, src: 'median' };
	let inPerM = e.inPerM;
	let outPerM = e.outPerM;
	if (tier === 'batch') {
		inPerM = e.batchInPerM ?? e.inPerM * BATCH_FRACTION;
		outPerM = e.batchOutPerM ?? e.outPerM * BATCH_FRACTION;
	}
	const cachedFrac = Math.max(0, Math.min(1, opts.cachedFrac ?? 0));
	if (cachedFrac > 0) {
		const cacheRead = e.cacheReadPerM ?? e.inPerM * CACHE_READ_FRACTION;
		const read = tier === 'batch' ? Math.min(cacheRead, inPerM) : cacheRead;
		inPerM = (1 - cachedFrac) * inPerM + cachedFrac * read;
	}
	return { inPerM, outPerM };
}

/** Typical API latency, used only as the editable starting value on the economics tab. */
export const DEFAULT_API_PERF = { ttftMs: 500, tps: 80 };
