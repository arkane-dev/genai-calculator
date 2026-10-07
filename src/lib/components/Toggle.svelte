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

<label class="toggle">
	<span>
		<span class="field-label">
			{label}{#if info}<InfoTip text={info} {label} />{/if}
		</span>
		{#if hint}<span class="hint">{hint}</span>{/if}
	</span>
	<button
		type="button"
		role="switch"
		aria-checked={checked}
		aria-label={label}
		onclick={() => (checked = !checked)}
		class="switch"
		class:on={checked}
	>
		<span class="knob"></span>
	</button>
</label>

<style>
	.toggle { display: flex; align-items: center; justify-content: space-between; gap: var(--nd-space-3); cursor: pointer; }
	.toggle .field-label { margin-bottom: 0; }
	.hint { display: block; }
	/* Square switch: NEONDECK has no round controls. */
	.switch {
		position: relative;
		flex: none;
		width: 2.75rem;
		height: 1.5rem;
		padding: 0;
		border: 1px solid var(--nd-line-strong);
		background: var(--nd-surface-2);
		cursor: pointer;
		transition: background var(--nd-dur-fast) var(--nd-ease), border-color var(--nd-dur-fast) var(--nd-ease);
	}
	.switch.on { border-color: var(--nd-accent); background: color-mix(in srgb, var(--nd-accent) 25%, transparent); }
	.knob {
		position: absolute;
		top: 0.1875rem;
		left: 0.1875rem;
		width: 1rem;
		height: 1rem;
		background: var(--nd-text-dim);
		transition: transform var(--nd-dur-fast) var(--nd-ease), background var(--nd-dur-fast) var(--nd-ease);
	}
	.switch.on .knob { transform: translateX(1.25rem); background: var(--nd-accent); }
	.switch:focus-visible { outline: 2px solid var(--nd-focus); outline-offset: 2px; }
	@media (prefers-reduced-motion: reduce) { .knob { transition: none; } }
</style>
