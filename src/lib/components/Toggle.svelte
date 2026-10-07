<script lang="ts">
	import InfoTip from './InfoTip.svelte';
	interface Props {
		label: string;
		checked: boolean;
		hint?: string;
		info?: string;
	}
	let { label, checked = $bindable(), hint = '', info }: Props = $props();
</script>

<label class="flex cursor-pointer items-center justify-between gap-3">
	<span>
		<span class="flex items-center gap-1.5 text-sm text-slate-300">
			{label}{#if info}<InfoTip text={info} {label} />{/if}
		</span>
		{#if hint}<span class="block text-xs text-slate-500">{hint}</span>{/if}
	</span>
	<button
		type="button"
		role="switch"
		aria-checked={checked}
		aria-label={label}
		onclick={() => (checked = !checked)}
		class="relative h-6 w-11 shrink-0 rounded-full transition-colors {checked
			? 'bg-teal-500'
			: 'bg-slate-700'}"
	>
		<span
			class="absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white transition-transform {checked
				? 'translate-x-5'
				: ''}"
		></span>
	</button>
</label>
