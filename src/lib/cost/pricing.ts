// GPU cost, cloud-agnostic: dollars per GPU-hour.
//
// Each GPU has a default on-demand rate: the median across GPU cloud providers
// (GetDeploying GPU price index, week of 2026-10-07). It is a starting point, not
// a quote. Real rates vary 2-3x between providers, so the cost panel lets you
// enter your own rate.
//
// Purchasing changes the rate by a discount:
//   on-demand  no commitment, the median rate as listed
//   committed  1-3 year reservation or committed-use contract
//   spot       interruptible / preemptible capacity
// The discounts are typical market figures and are editable. They are not tied to
// any one provider.
//
// Billing is per GPU. Set `nodeSize` to bill whole nodes instead (e.g. 8 for the
// HGX-style parts many providers only rent as full 8-GPU servers).

export type Purchasing = 'on-demand' | 'committed' | 'spot';

export const PURCHASING_LABELS: Record<Purchasing, string> = {
	'on-demand': 'On-demand',
	committed: 'Committed',
	spot: 'Spot'
};

/** Typical discounts off on-demand. Editable in the cost panel. */
export const DEFAULT_DISCOUNTS: Record<Purchasing, number> = {
	'on-demand': 0,
	committed: 0.35,
	spot: 0.6
};

export const PRICE_SOURCE = 'GetDeploying GPU price index';
export const PRICE_AS_OF = '2026-10-07';

export const PRICING_NOTE = `Default rates are the median on-demand price across GPU cloud providers (${PRICE_SOURCE}, ${PRICE_AS_OF}). Providers vary 2-3x around it, so enter your own quote when you have one. Committed and spot apply a typical discount.`;

interface GpuRate {
	perGpuHour: number | null; // median on-demand $/GPU-hour; null = no market data
	note?: string;
}

// Median on-demand $/GPU-hour, week of 2026-10-07.
const RATES: Record<string, GpuRate> = {
	b300: { perGpuHour: 8.82 },
	b200: { perGpuHour: 8.55 },
	h200: { perGpuHour: 5.4 },
	'h100-sxm': { perGpuHour: 3.47 },
	'a100-80': { perGpuHour: 1.83 },
	'a100-40': {
		perGpuHour: 1.83,
		note: 'No separate 40GB median; this is the A100 80GB figure. 40GB usually rents for less.'
	},
	l40s: { perGpuHour: 1.57 },
	'rtx-pro-6000': { perGpuHour: 2.15 },
	'rtx-pro-4500': { perGpuHour: null, note: 'No market data yet. Enter your own rate.' },
	rtx4090: { perGpuHour: 0.45 },
	l4: { perGpuHour: 0.87 },
	a10g: { perGpuHour: 1.55, note: 'Median for the A10 class (A10 / A10G).' },
	t4: { perGpuHour: 0.64 }
};

/** On-demand market rate for a GPU, or null when there is no market data. */
export function marketRate(gpuId: string): number | null {
	return RATES[gpuId]?.perGpuHour ?? null;
}

export function rateNote(gpuId: string): string | null {
	return RATES[gpuId]?.note ?? null;
}

/** Every GPU defaults to on-demand. Kept as a function so callers don't hard-code it. */
export function defaultPurchasing(_gpuId: string): Purchasing {
	return 'on-demand';
}

export interface CostOptions {
	discounts?: Partial<Record<Purchasing, number>>;
	nodeSize?: number; // bill whole nodes of this many GPUs (default 1 = per GPU)
}

export interface CostEstimate {
	billableGpus: number; // GPUs you pay for (numGpus rounded up to whole nodes)
	perGpuHour: number | null; // after the purchasing discount
	clusterPerHour: number | null;
	warning: string | null;
}

export function estimateCost(
	gpuId: string,
	numGpus: number,
	purchasing: Purchasing,
	opts: CostOptions = {}
): CostEstimate {
	const nodeSize = Math.max(1, Math.round(opts.nodeSize ?? 1));
	const n = Math.max(1, numGpus);
	const billableGpus = Math.ceil(n / nodeSize) * nodeSize;
	const base = marketRate(gpuId);
	if (base == null) {
		return {
			billableGpus,
			perGpuHour: null,
			clusterPerHour: null,
			warning: rateNote(gpuId) ?? 'No market rate for this GPU. Enter your own.'
		};
	}
	const discount = Math.min(0.95, Math.max(0, opts.discounts?.[purchasing] ?? DEFAULT_DISCOUNTS[purchasing]));
	const perGpuHour = base * (1 - discount);
	return {
		billableGpus,
		perGpuHour,
		clusterPerHour: perGpuHour * billableGpus,
		warning: rateNote(gpuId)
	};
}
