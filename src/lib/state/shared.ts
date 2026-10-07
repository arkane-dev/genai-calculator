// Cross-tab sync: the common config fields persist in sessionStorage so switching tabs
// (Workload / Modelling / Self-host vs API / Training) keeps the same model, GPU, tokens,
// quant, and parallelism. Each tab holds its own state shape; we only read/write the keys
// it actually has (applyShared already copies only matching keys). Survives navigation
// within the browsing session, not across a full browser restart.

import type { Purchasing } from '$lib/cost/pricing';

const KEY = 'gwp.shared';

// Union of fields shared across the tabs' Config / WorkloadSpec / TrainConfig.
const SHARED_KEYS = [
	'modelId',
	'gpuId',
	'gpusPerNode',
	'fabricId',
	'weightFormatId',
	'kvBits',
	'kvAllocation',
	'inputTokens',
	'outputTokens',
	'cachedPrefixFrac',
	'specDecode',
	'draftAcceptRate',
	'specTokens',
	'imagesPerRequest',
	'tp',
	'pp',
	'ppEnabled',
	'ep',
	'epEnabled',
	'numGpus',
	'batchSize',
	'phase',
	// Cost/pricing controls, shared so a custom $/GPU-hour set on one tab (via the cost
	// panel's "Use estimated pricing" toggle) carries to Self-host vs API and back.
	'purchasing',
	'useEstimatedPricing',
	'manualPerGpuHour'
] as const;

// Shared pricing state, owned by each page and bound into <CostPanel>. Kept as one object
// so it round-trips through saveShared({ ...config, ...pricing }) and readPricing().
const PURCHASING_VALUES: readonly string[] = ['on-demand', 'committed', 'spot'];
export interface PricingState {
	purchasing: Purchasing;
	useEstimatedPricing: boolean;
	manualPerGpuHour: number;
}

/** Copy any pricing fields present on a shared record into `p` (in place). */
export function readPricing(
	rec: Record<string, unknown> | null | undefined,
	p: PricingState
): void {
	if (!rec) return;
	if (typeof rec.purchasing === 'string' && PURCHASING_VALUES.includes(rec.purchasing))
		p.purchasing = rec.purchasing as Purchasing;
	if (typeof rec.useEstimatedPricing === 'boolean') p.useEstimatedPricing = rec.useEstimatedPricing;
	if (typeof rec.manualPerGpuHour === 'number') p.manualPerGpuHour = rec.manualPerGpuHour;
}

export function loadShared(): Record<string, unknown> | null {
	if (typeof sessionStorage === 'undefined') return null;
	try {
		const s = sessionStorage.getItem(KEY);
		return s ? (JSON.parse(s) as Record<string, unknown>) : null;
	} catch {
		return null;
	}
}

/** Persist the shared keys present on `state`. Reading state[k] here registers the
 *  fields as dependencies when called inside a Svelte $effect, so it re-saves on change. */
export function saveShared(state: Record<string, unknown>): void {
	if (typeof sessionStorage === 'undefined') return;
	const out: Record<string, unknown> = {};
	for (const k of SHARED_KEYS) if (k in state) out[k] = state[k];
	try {
		sessionStorage.setItem(KEY, JSON.stringify(out));
	} catch {
		/* quota / disabled storage — sync is best-effort */
	}
}
