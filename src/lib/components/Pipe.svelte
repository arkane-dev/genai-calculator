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

<div class="pipe">
	{#if label}
		<div class="kv small">
			<span class="dim">{label}</span>
			<span class="num mute">{detail}</span>
		</div>
	{/if}
	<div class="tube">
		<!-- fill -->
		<div class="fill" style:width="{pct}%" style:background-color={color}></div>
		<!-- animated flow stripes over the filled portion -->
		<div
			class="fill pipe-flow"
			style:width="{pct}%"
			style:--flow-dur={flowDur}
			style:opacity={frac > 0.02 ? 0.5 : 0}
		></div>
	</div>
</div>

<style>
	.pipe { display: flex; flex-direction: column; align-items: center; gap: var(--nd-space-1); }
	.pipe .kv { width: 100%; }
	.tube {
		position: relative;
		width: 100%;
		height: 1.5rem;
		overflow: hidden;
		border: 1px solid var(--nd-line-strong);
		background: var(--nd-surface-1);
	}
	.fill {
		position: absolute;
		top: 0;
		bottom: 0;
		left: 0;
		transition: width 300ms var(--nd-ease);
	}
	.pipe-flow {
		background-image: repeating-linear-gradient(
			-60deg,
			color-mix(in srgb, var(--nd-text) 35%, transparent) 0 8px,
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
		.pipe-flow { animation: none; }
		.fill { transition: none; }
	}
</style>
