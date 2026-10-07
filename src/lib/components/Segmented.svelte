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
		<div class="field-label">
			{label}{#if info}<InfoTip text={info} {label} />{/if}
		</div>
	{/if}
	<div class="seg">
		{#each options as opt (opt.value)}
			<button
				type="button"
				disabled={opt.disabled}
				title={opt.disabled ? 'Not supported on the selected GPU' : undefined}
				aria-pressed={value === opt.value}
				onclick={() => !opt.disabled && (value = opt.value)}
				class:on={value === opt.value}
			>
				{opt.label}
			</button>
		{/each}
	</div>
</div>

<style>
	.seg {
		display: flex;
		flex-wrap: wrap;
		gap: var(--nd-space-1);
		padding: var(--nd-space-1);
		background: color-mix(in srgb, var(--nd-surface-2) 60%, transparent);
	}
	button {
		flex: 1;
		padding: var(--nd-space-1) var(--nd-space-2);
		border: 0;
		background: transparent;
		color: var(--nd-text-dim);
		font-size: var(--nd-text-sm);
		white-space: nowrap;
		cursor: pointer;
		transition: background var(--nd-dur-fast) var(--nd-ease), color var(--nd-dur-fast) var(--nd-ease);
	}
	button:hover:not(:disabled):not(.on) { background: var(--nd-line-strong); color: var(--nd-text); }
	button.on { background: var(--nd-accent); color: var(--nd-text-on-neon); font-weight: 500; }
	button:disabled { color: var(--nd-text-mute); cursor: not-allowed; text-decoration: line-through; }
</style>
