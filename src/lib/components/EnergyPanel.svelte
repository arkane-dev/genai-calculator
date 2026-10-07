<script lang="ts">
	import InfoTip from './InfoTip.svelte';
	import { GPUS_BY_ID } from '$lib/profiler/data';
	import Segmented from './Segmented.svelte';
	import {
		co2ForRun,
		co2Per1kUnits,
		co2PerMillionTokens,
		estimateEnergy,
		GRIDS,
		DEFAULT_GRID,
		DEFAULT_PUE,
		DEFAULT_RENEWABLE_MATCH,
		PUE_PRESETS,
		type EmissionsMethod
	} from '$lib/energy/estimate';

	// mirrors the CostPanel metric shape
	type EnergyMetric =
		| { kind: 'rate'; perSec: number; unit: 'tokens' | 'images' | 'clips' | 'docs' | 'audio-hours' }
		| { kind: 'run'; hours: number };

	interface Props {
		gpuId: string;
		numGpus: number;
		util: number; // 0..1 utilization (typically max of memBw and compute fill)
		metric?: EnergyMetric;
		region?: string; // bindable: shared with the page so one selector drives cost + energy
		method?: EmissionsMethod; // bindable
	}
	let {
		gpuId,
		numGpus,
		util,
		metric,
		region = $bindable(DEFAULT_GRID),
		method = $bindable<EmissionsMethod>('location')
	}: Props = $props();
	let pue = $state(DEFAULT_PUE);
	let customGCo2 = $state(300);
	let renewablePct = $state(Math.round(DEFAULT_RENEWABLE_MATCH * 100));
	const gpu = $derived(GPUS_BY_ID.get(gpuId));
	const e = $derived(
		gpu
			? estimateEnergy({
					gpu,
					numGpus,
					util,
					region,
					method,
					pue,
					customGCo2PerKwh: customGCo2,
					renewableMatch: renewablePct / 100
				})
			: null
	);

	const fmtKw = (kw: number) =>
		kw >= 100 ? `${kw.toFixed(0)} kW` : kw >= 10 ? `${kw.toFixed(1)} kW` : `${kw.toFixed(2)} kW`;
	const fmtKwh = (kwh: number) =>
		kwh >= 1e6
			? `${(kwh / 1e6).toFixed(1)} GWh`
			: kwh >= 1e3
				? `${(kwh / 1e3).toFixed(1)} MWh`
				: `${kwh.toFixed(0)} kWh`;
	const fmtCo2 = (kg: number) =>
		kg >= 1e6
			? `${(kg / 1e6).toFixed(1)} kt`
			: kg >= 1e3
				? `${(kg / 1e3).toFixed(1)} t`
				: kg >= 1
					? `${kg.toFixed(0)} kg`
					: `${(kg * 1000).toFixed(0)} g`;

	// small-number formatter for per-work-unit CO2
	const fmtCo2Small = (kg: number | null) => {
		if (kg == null) return '—';
		const g = kg * 1000;
		if (g >= 1000) return `${(g / 1000).toFixed(2)} kg`;
		if (g >= 1) return `${g.toFixed(1)} g`;
		return `${(g * 1000).toFixed(1)} mg`;
	};

	const perUnit = $derived.by(() => {
		if (!e || !metric) return null;
		if (metric.kind === 'run')
			return {
				label: 'CO₂e for the run',
				value: fmtCo2(co2ForRun(e, metric.hours)),
				sub: `${metric.hours.toFixed(1)} h × ${fmtKw(e.facilityKw)}`
			};
		if (metric.perSec <= 0) return null;
		if (metric.unit === 'tokens')
			return {
				label: 'CO₂e / 1M tokens',
				value: fmtCo2Small(co2PerMillionTokens(e, metric.perSec)),
				sub: 'output tokens'
			};
		return {
			label: `CO₂e / 1k ${metric.unit}`,
			value: fmtCo2Small(co2Per1kUnits(e, metric.perSec)),
			sub: metric.unit
		};
	});
</script>

<section class="panel">
	<div class="panel-head">
		<div class="panel-title">
			Power &amp; energy
			<InfoTip
				text="Estimated power at the wall (GPU draw + host overhead × datacenter PUE). GPU draw scales with utilization: idle sits near 30% of TDP, sustained work near TDP. Emissions use the grid's carbon intensity: a hydro grid like Sweden's is more than 10× cleaner than India's coal-heavy grid. First-order estimate."
			/>
		</div>
		<label class="inline-field">
			Grid
			<select
				bind:value={region}
				class="input compact"
			>
				{#each GRIDS as g (g.id)}<option value={g.id}>{g.label}</option>{/each}
				<option value="custom">Custom…</option>
			</select>
		</label>
	</div>

	<div class="notice spaced">
		<span class="notice-title">Estimates only.</span>
		First-order approximations, not audited figures. Grid intensities are directional, and TDP-based
		power under-counts real host draw. For anything you report externally, use measured figures from
		your provider or your own meters.
	</div>

	{#if !e || !gpu}
		<p class="size-sm dim">No power figure for this GPU.</p>
	{:else}
		<div class="field-row spaced">
			{#if region === 'custom'}
				<label class="field small dim">
					Grid intensity (gCO₂e/kWh)
					<input
						type="number"
						bind:value={customGCo2}
						min="0"
						step="10"
						class="input compact num-7"
					/>
				</label>
			{/if}
			<label class="field small dim">
				<span class="row"
					>PUE <InfoTip
						text="Power Usage Effectiveness: total facility power ÷ IT power. Large cloud datacenters report about 1.1-1.2, colocation about 1.4; the industry average is about 1.56."
					/></span
				>
				<input
					type="number"
					bind:value={pue}
					min="1"
					max="3"
					step="0.01"
					class="input compact num-6"
				/>
			</label>
			<div class="row">
				{#each PUE_PRESETS as preset (preset.label)}
					<button
						type="button"
						class="preset"
						class:on={pue === preset.pue}
						aria-pressed={pue === preset.pue}
						onclick={() => (pue = preset.pue)}>{preset.label} {preset.pue}</button
					>
				{/each}
			</div>
		</div>
		<div class="field-row spaced">
			<Segmented
				label="Emissions method"
				bind:value={method}
				options={[
					{ value: 'location', label: 'Location-based (grid mix)' },
					{ value: 'market', label: 'Market-based (renewables)' }
				]}
				info="Location-based counts the actual grid mix: what a physical measurement would show. Market-based subtracts the renewable energy your provider buys. Many large providers report matching most or all of their annual use. Both are legitimate; they answer different questions."
			/>
			{#if method === 'market'}
				<label class="field small dim">
					Renewable match (%)
					<input
						type="number"
						bind:value={renewablePct}
						min="0"
						max="100"
						step="5"
						class="input compact num-6"
					/>
				</label>
			{/if}
		</div>
		<div class="tiles">
			<div class="tile">
				<div class="tile-label">Power draw</div>
				<div class="tile-value">{fmtKw(e.facilityKw)}</div>
				<div class="tile-sub">
					{e.perGpuWatts.toFixed(0)} W / GPU · PUE {e.pue.toFixed(2)}
				</div>
			</div>
			<div class="tile">
				<div class="tile-label">Energy / day</div>
				<div class="tile-value">{fmtKwh(e.kwhPerDay)}</div>
				<div class="tile-sub">{fmtKwh(e.kwhPerYear)} / year</div>
			</div>
			<div class="tile">
				<div class="tile-label">CO₂e / year</div>
				<div class="tile-value">{fmtCo2(e.gCo2PerYear / 1000)}</div>
				<div class="tile-sub">
					{e.gCo2PerKwh.toFixed(e.gCo2PerKwh < 10 ? 1 : 0)} g/kWh · {e.method === 'market'
						? 'market-based'
						: 'location-based'}
				</div>
			</div>
			{#if perUnit}
				<div class="tile good">
					<div class="tile-label">{perUnit.label}</div>
					<div class="tile-value">{perUnit.value}</div>
					<div class="tile-sub">{perUnit.sub}</div>
				</div>
			{:else}
				<div class="tile">
					<div class="tile-label">CO₂e / hour</div>
					<div class="tile-value">{fmtCo2(e.gCo2PerHour / 1000)}</div>
					<div class="tile-sub">
						{e.method === 'market' ? 'market-based' : 'location-based'}, {e.regionLabel}
					</div>
				</div>
			{/if}
		</div>
		<p class="note">
			First-order: GPU draw = idle 30% of TDP + linear to TDP with utilization. Host overhead 30% of
			GPU. PUE {e.pue.toFixed(2)}. Grid intensities are directional estimates anchored to public 2024
			grid-mix data.{e.method === 'market'
				? ` Market-based applies a ${Math.round(e.renewableMatch * 100)}% renewable match.`
				: ''}
		</p>
	{/if}
</section>
