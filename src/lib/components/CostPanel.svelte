<script lang="ts">
	import InfoTip from './InfoTip.svelte';
	import Segmented from './Segmented.svelte';
	import Toggle from './Toggle.svelte';
	import {
		defaultPurchasing,
		estimateCost,
		PRICING_NOTE,
		PURCHASING_LABELS,
		type Purchasing
	} from '$lib/cost/pricing';

	// A rate produces cost-per-unit-of-work; a run produces a total for the job.
	type CostMetric =
		| { kind: 'rate'; perSec: number; unit: 'tokens' | 'images' | 'clips' | 'docs' | 'audio-hours' }
		| { kind: 'run'; hours: number };

	interface Props {
		gpuId: string;
		numGpus: number;
		metric?: CostMetric;
		// Pricing controls are bindable so the parent page can own them and sync them
		// across tabs (a manual $/GPU-hour set here carries to Self-host vs API).
		useEstimated?: boolean;
		manualPerGpuHour?: number;
		purchasing?: Purchasing;
	}
	let {
		gpuId,
		numGpus,
		metric,
		useEstimated = $bindable(true),
		manualPerGpuHour = $bindable(0),
		purchasing = $bindable<Purchasing>('on-demand')
	}: Props = $props();

	// Estimated pricing on by default; when off, the user supplies a per-GPU-hour rate.
	let manualSeeded = $state(false);
	// Many providers rent the big HGX parts only as whole 8-GPU servers.
	let wholeNodes = $state(false);

	// Reset purchasing when the GPU changes. A manual pick sticks while the GPU doesn't change.
	let seededFor = $state('');
	$effect(() => {
		if (gpuId !== seededFor) {
			purchasing = defaultPurchasing(gpuId);
			seededFor = gpuId;
		}
	});

	const est = $derived(estimateCost(gpuId, numGpus, purchasing, { nodeSize: wholeNodes ? 8 : 1 }));

	// seed the manual price from the estimate the first time the user turns estimates off,
	// unless the parent already supplied one (e.g. synced from another tab).
	$effect(() => {
		if (!useEstimated && !manualSeeded && !(manualPerGpuHour > 0)) {
			manualPerGpuHour = est.perGpuHour != null ? Math.round(est.perGpuHour * 100) / 100 : 4;
			manualSeeded = true;
		}
	});

	// unified cluster $/hr: from the estimate, or the user's per-GPU rate × GPU count
	const clusterPerHour = $derived(
		useEstimated ? est.clusterPerHour : manualPerGpuHour > 0 ? manualPerGpuHour * numGpus : null
	);

	const fmtUsd = (n: number) =>
		n >= 100 ? `$${n.toLocaleString('en-US', { maximumFractionDigits: 0 })}` : `$${n.toFixed(2)}`;
	const fmtUsdSmall = (n: number) =>
		n >= 1 ? `$${n.toFixed(2)}` : n >= 0.01 ? `$${n.toFixed(3)}` : `$${n.toFixed(5)}`;
	const fmtHours = (h: number) =>
		h < 1
			? `${(h * 60).toFixed(0)} min`
			: h < 48
				? `${h.toFixed(1)} h`
				: `${(h / 24).toFixed(1)} days`;

	// derived cost-per-work-unit
	const perUnit = $derived.by(() => {
		if (!metric || clusterPerHour == null) return null;
		const perSecDollars = clusterPerHour / 3600;
		if (metric.kind === 'run') {
			return {
				label: 'Total for the run',
				value: fmtUsd(clusterPerHour * metric.hours),
				sub: `${fmtHours(metric.hours)} × ${fmtUsd(clusterPerHour)}/hr`
			};
		}
		if (metric.perSec <= 0) return null;
		if (metric.unit === 'tokens') {
			const per1M = (perSecDollars / metric.perSec) * 1e6;
			return { label: 'Cost / 1M tokens', value: fmtUsdSmall(per1M), sub: 'output tokens' };
		}
		const per1k = (perSecDollars / metric.perSec) * 1e3;
		return { label: `Cost / 1k ${metric.unit}`, value: fmtUsdSmall(per1k), sub: metric.unit };
	});

	const purchasingOptions = (['on-demand', 'committed', 'spot'] as const).map((value) => ({
		value,
		label: PURCHASING_LABELS[value]
	}));
</script>

<section class="rounded-xl border border-slate-700 bg-slate-900/60 p-5">
	<div class="mb-3 flex flex-wrap items-center justify-between gap-3">
		<div class="flex items-center gap-1.5 text-sm font-medium text-slate-200">
			Cost estimate
			<InfoTip
				text="Estimated pricing uses the median on-demand rate for this GPU across GPU cloud providers, with a typical discount for committed or spot capacity. Turn estimates off to enter your own $/GPU-hour, and every figure below is recomputed from it."
			/>
		</div>
		<Toggle
			label="Use estimated pricing"
			bind:checked={useEstimated}
			info="On: the median market rate for this GPU. Off: enter your own $/GPU-hour, such as a quote or a negotiated rate."
		/>
	</div>

	{#if useEstimated}
		<div class="mb-3 rounded-lg border border-amber-600/40 bg-amber-500/5 p-3 text-xs text-amber-200/90">
			<span class="font-semibold text-amber-300">Estimates only.</span>
			Rates vary 2-3x between providers, and committed contracts or private pricing can change the
			bill a lot. Get a real quote before you decide.
		</div>
		<div class="mb-4 flex flex-wrap items-end gap-4">
			<Segmented
				label="Purchasing"
				bind:value={purchasing}
				options={purchasingOptions}
				info="On-demand: no commitment, the median market rate. Committed: a 1-3 year reservation, about 35% less. Spot: interruptible capacity, about 60% less, but it can be taken away."
			/>
			<Toggle
				label="Bill whole 8-GPU nodes"
				bind:checked={wholeNodes}
				info="Many providers rent H100/H200/B200-class GPUs only as full 8-GPU servers. On: the GPU count rounds up to whole nodes."
			/>
		</div>
	{:else}
		<div class="mb-3 rounded-lg border border-teal-700/40 bg-teal-500/[0.04] p-3">
			<label class="block">
				<span class="flex items-center gap-1.5 text-xs text-slate-300">
					$ / GPU-hour
					<InfoTip
						text="Your own price for one GPU for one hour. The whole-cluster cost is this × the {numGpus} GPU(s) sized here, so it scales as the workload changes."
					/>
				</span>
				<input
					type="number"
					bind:value={manualPerGpuHour}
					min="0"
					step="0.01"
					class="mt-1 w-full max-w-[12rem] rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100"
				/>
			</label>
			<p class="mt-1.5 text-[11px] text-slate-500">
				Cluster / hour = {fmtUsd(manualPerGpuHour || 0)} × {numGpus} GPU = {clusterPerHour == null
					? '—'
					: fmtUsd(clusterPerHour)}/hr. Your figure, not a market estimate.
			</p>
		</div>
	{/if}

	<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
		<div class="rounded-lg border border-slate-700 bg-slate-800/40 p-3">
			<div class="text-[11px] tracking-wide text-slate-500 uppercase">GPUs billed</div>
			<div class="mt-1 font-mono text-lg text-slate-100">
				{useEstimated ? est.billableGpus : numGpus} ×
			</div>
			<div class="mt-0.5 text-xs text-slate-400">
				{useEstimated
					? est.perGpuHour == null
						? 'no market rate'
						: `${fmtUsdSmall(est.perGpuHour)}/GPU-hr`
					: 'your rate'}
			</div>
		</div>
		<div class="rounded-lg border border-slate-700 bg-slate-800/40 p-3">
			<div class="text-[11px] tracking-wide text-slate-500 uppercase">Cluster / hour</div>
			<div class="mt-1 font-mono text-lg text-slate-100">
				{clusterPerHour == null ? '—' : fmtUsd(clusterPerHour)}
			</div>
			<div class="mt-0.5 text-xs text-slate-400">
				{useEstimated ? `${PURCHASING_LABELS[purchasing]}${wholeNodes ? ' · whole nodes' : ''}` : 'your price'}
			</div>
		</div>
		<div class="rounded-lg border border-slate-700 bg-slate-800/40 p-3">
			<div class="text-[11px] tracking-wide text-slate-500 uppercase">Cluster / day</div>
			<div class="mt-1 font-mono text-lg text-slate-100">
				{clusterPerHour == null ? '—' : fmtUsd(clusterPerHour * 24)}
			</div>
			<div class="mt-0.5 text-xs text-slate-400">24 × hourly</div>
		</div>
		{#if perUnit}
			<div class="rounded-lg border border-emerald-700/50 bg-emerald-500/5 p-3">
				<div class="text-[11px] tracking-wide text-emerald-400/80 uppercase">{perUnit.label}</div>
				<div class="mt-1 font-mono text-lg text-emerald-300">{perUnit.value}</div>
				<div class="mt-0.5 text-xs text-slate-400">{perUnit.sub}</div>
			</div>
		{:else}
			<div class="rounded-lg border border-slate-700 bg-slate-800/40 p-3">
				<div class="text-[11px] tracking-wide text-slate-500 uppercase">Cluster / month</div>
				<div class="mt-1 font-mono text-lg text-slate-100">
					{clusterPerHour == null ? '—' : fmtUsd(clusterPerHour * 730)}
				</div>
				<div class="mt-0.5 text-xs text-slate-400">730 hr</div>
			</div>
		{/if}
	</div>

	{#if useEstimated && est.warning}
		<p class="mt-3 rounded-lg border border-amber-600/40 bg-amber-500/5 p-2.5 text-xs text-amber-200/90">
			{est.warning}
		</p>
	{/if}
	{#if useEstimated}
		<p class="mt-2 text-[10px] text-slate-500">{PRICING_NOTE}</p>
	{/if}
</section>
