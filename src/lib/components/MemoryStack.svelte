<script lang="ts">
	import { fmtBytes } from '$lib/profiler/calc';
	import type { MemorySegment } from '$lib/profiler/types';

	interface Props {
		segments: MemorySegment[];
		capacity: number;
		used: number;
		fits: boolean;
		warnNearFull?: boolean; // opt-in: flag high memory pressure (red bar + fill %) even when it fits
	}
	let { segments, capacity, used, fits, warnNearFull = false }: Props = $props();

	// The box represents whichever is larger: the capacity, or the demand when
	// it overflows. That keeps the capacity line meaningful in both cases.
	const total = $derived(Math.max(capacity, used));
	const capacityPct = $derived((capacity / total) * 100);
	const pct = (bytes: number) => (bytes / total) * 100;

	const fillPct = $derived(capacity > 0 ? (used / capacity) * 100 : 0);
	// "hot" = overflowing, or (when opted in) running close to the capacity line
	const hot = $derived(!fits || (warnNearFull && fillPct >= 85));
</script>

<div class="flex h-full flex-col">
	<div class="mb-2 flex items-baseline justify-between">
		<span class="text-sm font-medium text-slate-200">HBM Memory</span>
		<span class="font-mono text-xs {hot ? 'text-red-400' : 'text-slate-400'}">
			{fmtBytes(used)} / {fmtBytes(capacity)}
			{#if hot}<span class="ml-1">· {fillPct.toFixed(0)}% full</span>{/if}
		</span>
	</div>

	<div
		class="relative flex-1 overflow-hidden rounded-lg border bg-slate-900 {hot
			? 'border-red-400/70'
			: 'border-slate-600'}"
	>
		<!-- stacked segments, top to bottom -->
		<div class="flex h-full flex-col">
			{#each segments as seg (seg.key)}
				<div
					class="relative flex items-center justify-between overflow-hidden px-2 transition-[height] duration-300 ease-out"
					style:height="{pct(seg.bytes)}%"
					style:background-color={seg.key === 'free' ? 'transparent' : seg.color}
					style:border-top={seg.key === 'free' ? 'none' : '1px solid rgba(15,23,42,0.5)'}
				>
					{#if pct(seg.bytes) > 7 && seg.key !== 'free'}
						<span class="truncate text-xs font-medium text-slate-900">{seg.label}</span>
						<span class="font-mono text-xs text-slate-900">{fmtBytes(seg.bytes)}</span>
					{/if}
				</div>
			{/each}
		</div>

		<!-- capacity line, shown when demand overflows the device -->
		{#if !fits}
			<div
				class="pointer-events-none absolute inset-x-0 border-t-2 border-dashed border-red-300"
				style:top="{capacityPct}%"
			>
				<span class="absolute -top-4 right-1 rounded bg-red-500 px-1 text-[10px] text-white">
					capacity
				</span>
			</div>
		{/if}
	</div>
</div>
