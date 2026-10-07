<script lang="ts">
	import { untrack } from 'svelte';
	import InfoTip from './InfoTip.svelte';
	interface Props {
		label: string;
		value: number;
		min: number;
		max: number;
		step?: number;
		display?: string; // formatted value to show; falls back to the raw value
		info?: string; // plain-language explanation shown via an "i" button
		log?: boolean; // map the slider position to [min,max] logarithmically
	}
	let {
		label,
		value = $bindable(),
		min,
		max,
		step = 1,
		display,
		info,
		log = false
	}: Props = $props();

	// Log mode drives a 0..STEPS position that maps exponentially onto [min,max], so each
	// pixel covers a constant ratio instead of a constant amount. Values snap to 2 significant
	// figures so the picked number reads cleanly (47M, not 46 813 271).
	const STEPS = 1000;
	const clampV = (v: number) => Math.min(max, Math.max(min, v));
	const roundNice = (x: number) => {
		const mag = Math.pow(10, Math.floor(Math.log10(x)) - 1);
		return Math.round(x / mag) * mag;
	};
	const toPos = (v: number) =>
		Math.round((STEPS * Math.log(clampV(v) / min)) / Math.log(max / min));
	const fromPos = (p: number) => clampV(roundNice(min * Math.pow(max / min, p / STEPS)));

	let pos = $state(untrack(() => (log ? toPos(value) : 0)));
	let lastValue = value;
	// External changes (defaults, shared-link restore) re-derive the position.
	$effect(() => {
		if (log && value !== lastValue) {
			pos = toPos(value);
			lastValue = value;
		}
	});
	function onLogInput() {
		const v = fromPos(pos);
		value = v;
		lastValue = v;
	}
</script>

<label class="slider">
	<div class="head">
		<span class="field-label">
			{label}{#if info}<InfoTip text={info} {label} />{/if}
		</span>
		<span class="num bright">{display ?? value}</span>
	</div>
	{#if log}
		<input type="range" min={0} max={STEPS} step={1} bind:value={pos} oninput={onLogInput} />
	{:else}
		<input type="range" {min} {max} {step} bind:value />
	{/if}
</label>

<style>
	.slider { display: block; }
	.head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: var(--nd-space-1); }
	.head .field-label { margin-bottom: 0; }
	.num { font-size: var(--nd-text-sm); }
	/* Hairline track, square accent thumb. */
	input {
		width: 100%;
		height: 1.25rem;
		appearance: none;
		background: transparent;
		cursor: pointer;
	}
	input::-webkit-slider-runnable-track { height: 2px; background: var(--nd-line-strong); }
	input::-moz-range-track { height: 2px; background: var(--nd-line-strong); }
	input::-webkit-slider-thumb {
		appearance: none;
		width: 0.875rem;
		height: 0.875rem;
		margin-top: calc(-0.4375rem + 1px);
		border: 0;
		background: var(--nd-accent);
	}
	input::-moz-range-thumb { width: 0.875rem; height: 0.875rem; border: 0; border-radius: 0; background: var(--nd-accent); }
	input:focus-visible { outline: 2px solid var(--nd-focus); outline-offset: 2px; }
</style>
