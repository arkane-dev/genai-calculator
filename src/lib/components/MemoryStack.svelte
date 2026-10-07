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

<div class="mem">
	<div class="kv head">
		<span class="title">HBM Memory</span>
		<span class="num small" class:bad={hot} class:dim={!hot}>
			{fmtBytes(used)} / {fmtBytes(capacity)}
			{#if hot}<span>· {fillPct.toFixed(0)}% full</span>{/if}
		</span>
	</div>

	<div class="stack" class:hot>
		<!-- stacked segments, top to bottom -->
		<div class="segs">
			{#each segments as seg (seg.key)}
				<div
					class="seg"
					style:height="{pct(seg.bytes)}%"
					style:background-color={seg.key === 'free' ? 'transparent' : seg.color}
					style:border-top={seg.key === 'free' ? 'none' : '1px solid rgba(3, 4, 12, 0.5)'}
				>
					{#if pct(seg.bytes) > 7 && seg.key !== 'free'}
						<span class="seg-label">{seg.label}</span>
						<span class="num seg-label">{fmtBytes(seg.bytes)}</span>
					{/if}
				</div>
			{/each}
		</div>

		<!-- capacity line, shown when demand overflows the device -->
		{#if !fits}
			<div class="cap" style:top="{capacityPct}%">
				<span>capacity</span>
			</div>
		{/if}
	</div>
</div>

<style>
	.mem { display: flex; flex-direction: column; height: 100%; }
	.head { margin-bottom: var(--nd-space-2); }
	.title { color: var(--nd-text); font-size: var(--nd-text-sm); font-weight: 500; }
	.stack {
		position: relative;
		flex: 1;
		overflow: hidden;
		border: 1px solid var(--nd-line-strong);
		background: var(--nd-surface-1);
	}
	.stack.hot { border-color: var(--nd-red); }
	.segs { display: flex; flex-direction: column; height: 100%; }
	.seg {
		position: relative;
		display: flex;
		align-items: center;
		justify-content: space-between;
		overflow: hidden;
		padding: 0 var(--nd-space-2);
		transition: height 300ms var(--nd-ease);
	}
	/* Dark text on the bright segment colours. */
	.seg-label {
		overflow: hidden;
		color: var(--nd-text-on-neon);
		font-size: var(--nd-text-xs);
		font-weight: 500;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.cap {
		position: absolute;
		right: 0;
		left: 0;
		border-top: 2px dashed var(--nd-red);
		pointer-events: none;
	}
	.cap span {
		position: absolute;
		top: -1rem;
		right: var(--nd-space-1);
		padding: 0 var(--nd-space-1);
		background: var(--nd-red);
		color: var(--nd-text-on-neon);
		font-size: 0.625rem;
	}
	@media (prefers-reduced-motion: reduce) { .seg { transition: none; } }
</style>
