<script lang="ts">
	import InfoTip from './InfoTip.svelte';
	interface Option {
		value: string | number;
		label: string;
		disabled?: boolean; // visible but not selectable (e.g. format the GPU can't run)
	}
	interface Props {
		label: string;
		value: string | number;
		options: Option[];
		info?: string;
	}
	let { label, value = $bindable(), options, info }: Props = $props();
</script>

<div>
	{#if label || info}
		<div class="mb-1 flex items-center gap-1.5 text-sm text-slate-300">
			{label}{#if info}<InfoTip text={info} {label} />{/if}
		</div>
	{/if}
	<div class="flex flex-wrap gap-1 rounded-lg bg-slate-800/60 p-1">
		{#each options as opt (opt.value)}
			<button
				type="button"
				disabled={opt.disabled}
				title={opt.disabled ? 'Not supported on the selected GPU' : undefined}
				onclick={() => !opt.disabled && (value = opt.value)}
				class="flex-1 rounded-md px-2 py-1 text-sm whitespace-nowrap transition-colors
					{opt.disabled
					? 'cursor-not-allowed text-slate-600'
					: value === opt.value
						? 'bg-teal-500 font-medium text-slate-900'
						: 'text-slate-300 hover:bg-slate-700'}"
			>
				{opt.label}
			</button>
		{/each}
	</div>
</div>
