<script lang="ts">
	interface Props {
		frac: number; // 0..1 fill
		color: string;
		label?: string;
		detail?: string;
	}
	let { frac, color, label = '', detail = '' }: Props = $props();

	const pct = $derived(Math.max(0, Math.min(1, frac)) * 100);
	// Flow appears only when there is throughput, and moves faster as it fills.
	const flowDur = $derived(frac > 0.02 ? `${Math.max(0.35, 1.8 - frac * 1.5)}s` : '0s');
</script>

<div class="flex flex-col items-center gap-1">
	{#if label}
		<div class="flex w-full items-baseline justify-between text-xs">
			<span class="text-slate-300">{label}</span>
			<span class="font-mono text-slate-400">{detail}</span>
		</div>
	{/if}
	<div
		class="relative h-6 w-full overflow-hidden rounded-full border border-slate-600 bg-slate-900"
	>
		<!-- fill -->
		<div
			class="absolute inset-y-0 left-0 transition-[width] duration-300 ease-out"
			style:width="{pct}%"
			style:background-color={color}
		></div>
		<!-- animated flow stripes over the filled portion -->
		<div
			class="pipe-flow absolute inset-y-0 left-0 transition-[width] duration-300 ease-out"
			style:width="{pct}%"
			style:--flow-dur={flowDur}
			style:opacity={frac > 0.02 ? 0.5 : 0}
		></div>
	</div>
</div>

<style>
	.pipe-flow {
		background-image: repeating-linear-gradient(
			-60deg,
			rgba(255, 255, 255, 0.35) 0 8px,
			transparent 8px 20px
		);
		background-size: 200% 100%;
		animation: flow var(--flow-dur, 1s) linear infinite;
	}
	@keyframes flow {
		to {
			background-position: -40px 0;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.pipe-flow {
			animation: none;
		}
	}
</style>
