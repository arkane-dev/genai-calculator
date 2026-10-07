<script lang="ts">
	import InfoTip from './InfoTip.svelte';

	export interface CompareRow {
		label: string;
		a: string; // formatted value for scenario A (pinned)
		b: string; // formatted value for scenario B (current)
		aNum?: number; // raw values, for the delta + winner highlight
		bNum?: number;
		better?: 'higher' | 'lower'; // which direction is "good"; omit for info-only rows
	}
	interface Props {
		labelA: string;
		labelB: string;
		rows: CompareRow[];
		onclear?: () => void;
	}
	let { labelA, labelB, rows, onclear }: Props = $props();

	// winner + delta for a row, when both sides are numeric and a direction is set
	function verdict(r: CompareRow): { win: 'a' | 'b' | null; delta: string } {
		if (r.aNum == null || r.bNum == null || !r.better || r.aNum === r.bNum)
			return { win: null, delta: '' };
		const better =
			r.better === 'higher' ? (r.bNum > r.aNum ? 'b' : 'a') : r.bNum < r.aNum ? 'b' : 'a';
		const pct = r.aNum !== 0 ? ((r.bNum - r.aNum) / Math.abs(r.aNum)) * 100 : null;
		const delta = pct == null ? '' : `${pct > 0 ? '+' : ''}${pct.toFixed(0)}%`;
		return { win: better, delta };
	}
</script>

<section class="panel">
	<div class="panel-head">
		<h3 class="panel-title">
			Compare scenarios
			<InfoTip
				text="Scenario A is the config you pinned; B is what's on screen now. Change any control and the B column updates live, so you can weigh two hardware or config choices side by side. Green marks the better side per row (faster / cheaper); Δ is B relative to A."
			/>
		</h3>
		{#if onclear}
			<button type="button" onclick={onclear} class="tool-btn">Clear</button>
		{/if}
	</div>

	<div class="scroll">
		<table class="data-table">
			<thead>
				<tr>
					<th><span class="sr">Metric</span></th>
					<th>A · {labelA}</th>
					<th>B · {labelB}</th>
					<th>Δ (B vs A)</th>
				</tr>
			</thead>
			<tbody>
				{#each rows as r (r.label)}
					{@const v = verdict(r)}
					<tr>
						<td class="dim">{r.label}</td>
						<td class="num" class:ok={v.win === 'a'} class:bright={v.win !== 'a'}>{r.a}</td>
						<td class="num" class:ok={v.win === 'b'} class:bright={v.win !== 'b'}>{r.b}</td>
						<td class="num" class:ok={v.win === 'b'} class:warn={v.win === 'a'} class:mute={!v.win}
							>{v.delta || '—'}</td
						>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
</section>

<style>
	.scroll { overflow-x: auto; }
	.data-table { font-size: var(--nd-text-sm); }
	.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
</style>
