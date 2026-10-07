// One shared scenario for every tab, held in Svelte state.
//
// Each tab keeps its own config shape (Workload's spec, Modelling's config, Training's
// cfg), but they overlap: model, GPU, fabric, tokens, quant, parallelism, pricing and
// more. Every tab writes all of its fields into `scenario` on every change, and reads
// the ones it has from it when it opens. So whichever tab you switch to, it starts from
// the latest values set anywhere. Writes merge: a tab never erases fields it doesn't
// have (Training doesn't wipe Workload's latency targets or the pricing choice).
//
// The scenario lives in this module, so it survives client-side navigation. It is also
// mirrored to sessionStorage, so a reload keeps it for the rest of the browser session.

import type { Purchasing } from '$lib/cost/pricing';

const KEY = 'gwp.shared';

function restore(): Record<string, unknown> {
	if (typeof sessionStorage === 'undefined') return {};
	try {
		const s = sessionStorage.getItem(KEY);
		const v = s ? JSON.parse(s) : null;
		return v && typeof v === 'object' && !Array.isArray(v) ? v : {};
	} catch {
		return {};
	}
}

/** The shared scenario: the latest value of every field any tab has set. */
export const scenario = $state<Record<string, unknown>>(restore());

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

/** The current scenario, for a tab to read the fields it has when it opens. */
export function loadShared(): Record<string, unknown> {
	return scenario;
}

/**
 * Merge a tab's fields into the scenario. Only plain values (string, number, boolean)
 * are shared. `except` keeps fields this tab only holds locally (e.g. a fallback model).
 * Reading state[k] here registers the fields as dependencies when called inside a
 * Svelte $effect, so it re-saves on every change.
 */
export function saveShared(state: Record<string, unknown>, opts: { except?: string[] } = {}): void {
	let changed = false;
	for (const [k, v] of Object.entries(state)) {
		if (opts.except?.includes(k)) continue;
		if (typeof v !== 'string' && typeof v !== 'number' && typeof v !== 'boolean') continue;
		if (scenario[k] !== v) {
			scenario[k] = v;
			changed = true;
		}
	}
	if (!changed || typeof sessionStorage === 'undefined') return;
	try {
		sessionStorage.setItem(KEY, JSON.stringify(scenario));
	} catch {
		/* quota / disabled storage — the in-memory scenario still works */
	}
}

/**
 * For a tab that only supports some models (Self-host vs API: text only; Training:
 * transformers only). If the shared model isn't one it supports, the tab shows a
 * fallback, but keeps it to itself so other tabs keep the model you chose there.
 * Once you pick a model on this tab, it is shared like everything else.
 */
export class LocalModelFallback {
	#fallback: string;
	active = $state(false);
	constructor(fallback: string) {
		this.#fallback = fallback;
	}
	/** Call after reading the shared scenario. Swaps in the fallback if needed. */
	apply(cfg: { modelId: string }, supported: (id: string) => boolean): void {
		if (!supported(cfg.modelId)) {
			cfg.modelId = this.#fallback;
			this.active = true;
		}
	}
	/** Call from an $effect: the fallback ends once the user picks another model. */
	track(modelId: string): void {
		if (this.active && modelId !== this.#fallback) this.active = false;
	}
	/** Fields to leave out of saveShared while the fallback is showing. */
	get except(): string[] {
		return this.active ? ['modelId'] : [];
	}
}
