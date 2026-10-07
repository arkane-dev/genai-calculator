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
	let { label, value, unit = '', sub = '', accent = '#e2e8f0', info }: Props = $props();
</script>

<div class="rounded-xl border border-slate-700 bg-slate-800/50 p-4">
	<div class="flex items-center gap-1.5 text-xs tracking-wide text-slate-400 uppercase">
		<span>{label}</span>{#if info}<InfoTip text={info} {label} />{/if}
	</div>
	<div class="mt-1 flex items-baseline gap-1">
		<span class="text-3xl font-semibold" style:color={accent}>{value}</span>
		{#if unit}<span class="text-sm text-slate-400">{unit}</span>{/if}
	</div>
	{#if sub}<div class="mt-1 text-xs text-slate-500">{sub}</div>{/if}
</div>
