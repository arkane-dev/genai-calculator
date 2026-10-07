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

<label class="block">
	<div class="mb-1 flex items-baseline justify-between">
		<span class="flex items-center gap-1.5 text-sm text-slate-300">
			{label}{#if info}<InfoTip text={info} {label} />{/if}
		</span>
		<span class="font-mono text-sm text-slate-100">{display ?? value}</span>
	</div>
	{#if log}
		<input
			type="range"
			min={0}
			max={STEPS}
			step={1}
			bind:value={pos}
			oninput={onLogInput}
			class="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-700 accent-teal-400"
		/>
	{:else}
		<input
			type="range"
			{min}
			{max}
			{step}
			bind:value
			class="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-700 accent-teal-400"
		/>
	{/if}
</label>
