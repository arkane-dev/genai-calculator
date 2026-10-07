<script lang="ts">
	import InfoTip from './InfoTip.svelte';
	interface Props {
		label: string;
		value: string;
		unit?: string;
		sub?: string;
		accent?: string; // css color for the value
		info?: string; // plain-language explanation of what the number means + why it matters
	}
	let { label, value, unit = '', sub = '', accent = '#ecebff', info }: Props = $props();
</script>

<div class="stat">
	<div class="label">
		<span>{label}</span>{#if info}<InfoTip text={info} {label} />{/if}
	</div>
	<div class="value-row">
		<span class="value" style:color={accent}>{value}</span>
		{#if unit}<span class="unit">{unit}</span>{/if}
	</div>
	{#if sub}<div class="sub">{sub}</div>{/if}
</div>

<style>
	.stat {
		border: 1px solid var(--nd-line-strong);
		background: color-mix(in srgb, var(--nd-surface-2) 50%, transparent);
		padding: var(--nd-space-4);
		min-width: 0;
	}
	.label {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		color: var(--nd-text-dim);
		font-size: var(--nd-text-xs);
		letter-spacing: 0.05em;
		text-transform: uppercase;
	}
	.value-row { display: flex; align-items: baseline; gap: var(--nd-space-1); margin-top: var(--nd-space-1); }
	.value { font-size: var(--nd-text-3xl); font-weight: 600; line-height: 1.1; }
	.unit { color: var(--nd-text-dim); font-size: var(--nd-text-sm); }
	.sub { margin-top: var(--nd-space-1); color: var(--nd-text-mute); font-size: var(--nd-text-xs); }
</style>
