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

<section class="rounded-xl border border-slate-700 bg-slate-900/60 p-5">
	<div class="mb-3 flex items-center justify-between gap-2">
		<div class="flex items-center gap-1.5 text-sm font-medium text-slate-200">
			Compare scenarios
			<InfoTip
				text="Scenario A is the config you pinned; B is what's on screen now. Change any control and the B column updates live, so you can weigh two hardware or config choices side by side. Green marks the better side per row (faster / cheaper); Δ is B relative to A."
			/>
		</div>
		{#if onclear}
			<button
				type="button"
				onclick={onclear}
				class="rounded-md border border-slate-600 px-2 py-1 text-xs text-slate-400 hover:bg-slate-800"
				>Clear</button
			>
		{/if}
	</div>

	<div class="overflow-x-auto">
		<table class="w-full text-sm">
			<thead class="text-left text-xs text-slate-500">
				<tr>
					<th class="py-1 pr-4"></th>
					<th class="py-1 pr-4">A · {labelA}</th>
					<th class="py-1 pr-4">B · {labelB}</th>
					<th class="py-1">Δ (B vs A)</th>
				</tr>
			</thead>
			<tbody>
				{#each rows as r (r.label)}
					{@const v = verdict(r)}
					<tr class="border-t border-slate-800">
						<td class="py-1.5 pr-4 text-slate-400">{r.label}</td>
						<td
							class="py-1.5 pr-4 font-mono {v.win === 'a' ? 'text-emerald-300' : 'text-slate-200'}"
							>{r.a}</td
						>
						<td
							class="py-1.5 pr-4 font-mono {v.win === 'b' ? 'text-emerald-300' : 'text-slate-200'}"
							>{r.b}</td
						>
						<td
							class="py-1.5 font-mono text-xs {v.win === 'b'
								? 'text-emerald-400'
								: v.win === 'a'
									? 'text-amber-400'
									: 'text-slate-500'}">{v.delta || '—'}</td
						>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
</section>
