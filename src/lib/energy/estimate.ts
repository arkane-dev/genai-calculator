// Power / energy / emissions estimates, cloud-agnostic.
//
// GPU draw is bounded by the board TDP but scales with utilization. A simple
// linear model works well for a first-order sizing tool:
//   watts = idleFrac × TDP + (1 − idleFrac) × TDP × util
// where `util` blends memory-bandwidth and compute pressure (whichever is
// higher — one or the other saturates in most inference/training). Real cards
// show ~30% idle draw and near-TDP under sustained work (per NVIDIA nvsmi
// telemetry across H100/A100).
//
// Host overhead adds the rest of the server: CPUs, RAM, NICs, storage, fans,
// PSU losses. As a fraction of GPU power on an 8-GPU HGX-class node it lands
// around 30%; we model it as a multiplier.
//
// PUE (Power Usage Effectiveness) adds cooling and facility losses. It depends on
// where the servers run, so it's an input with presets: large cloud datacenters
// report around 1.1-1.2, colocation around 1.4, and the industry-wide average is
// about 1.56 (Uptime Institute, 2024). Default 1.2.
//
// Emissions intensity depends on the electricity grid, reported two ways:
//   Location-based: the actual grid mix. The figures below are FIRST-ORDER
//     ESTIMATES anchored to public 2024 grid-mix data (Electricity Maps / country
//     averages). The ranking is real (Sweden and Oregon far below Virginia; India
//     far above) and magnitudes are within ~25% of published values, but they are
//     estimates, not audited figures. Pick "Custom" to enter your own.
//   Market-based: counts renewable energy your provider buys. Modelled as
//     location × (1 − renewable match). Many large providers report matching most
//     or all of their annual use with renewables; enter your provider's figure.
// Both are legitimate; they answer different questions.

import type { GpuSpec } from '$lib/profiler/types';

export type EmissionsMethod = 'location' | 'market';

export interface EnergyInputs {
	gpu: GpuSpec;
	numGpus: number;
	util: number; // 0..1 blended utilization (max of memBwFrac and computeFrac is a good pick)
	region: string; // a GRIDS id, or 'custom'
	customGCo2PerKwh?: number; // used when region === 'custom'
	method?: EmissionsMethod; // default 'location'
	renewableMatch?: number; // 0..1, market-based only; default 0.9
	pue?: number; // default 1.2
	idleFrac?: number; // default 0.3
	hostOverheadFrac?: number; // default 0.3
}

export interface EnergyResult {
	perGpuWatts: number; // GPU alone, under the modelled utilization
	hostWatts: number; // per GPU-equivalent host overhead
	facilityKw: number; // whole cluster kW at the wall (× PUE)
	kwhPerHour: number; // = facilityKw
	kwhPerDay: number;
	kwhPerYear: number;
	gCo2PerHour: number;
	gCo2PerYear: number;
	regionLabel: string;
	gCo2PerKwh: number; // the intensity applied under the chosen method
	method: EmissionsMethod;
	renewableMatch: number;
	pue: number;
}

export interface GridMix {
	id: string;
	label: string;
	gCo2PerKwh: number;
}

// Grid intensities, gCO2e/kWh. Anchors: Electricity Maps 2024 annual averages and
// national averages. First-order estimates; see the header.
export const GRIDS: GridMix[] = [
	{ id: 'world', label: 'World average', gCo2PerKwh: 475 },
	{ id: 'us-virginia', label: 'US · Virginia', gCo2PerKwh: 370 },
	{ id: 'us-ohio', label: 'US · Ohio', gCo2PerKwh: 480 },
	{ id: 'us-oregon', label: 'US · Oregon', gCo2PerKwh: 100 },
	{ id: 'us-california', label: 'US · California', gCo2PerKwh: 220 },
	{ id: 'ireland', label: 'Ireland', gCo2PerKwh: 260 },
	{ id: 'sweden', label: 'Sweden', gCo2PerKwh: 30 },
	{ id: 'germany', label: 'Germany', gCo2PerKwh: 340 },
	{ id: 'uk', label: 'United Kingdom', gCo2PerKwh: 180 },
	{ id: 'japan', label: 'Japan', gCo2PerKwh: 470 },
	{ id: 'singapore', label: 'Singapore', gCo2PerKwh: 490 },
	{ id: 'india', label: 'India', gCo2PerKwh: 700 },
	{ id: 'south-korea', label: 'South Korea', gCo2PerKwh: 440 }
];

export const GRIDS_BY_ID = new Map(GRIDS.map((g) => [g.id, g]));
export const DEFAULT_GRID = 'world';

export const PUE_PRESETS: { label: string; pue: number }[] = [
	{ label: 'Large cloud', pue: 1.15 },
	{ label: 'Colocation', pue: 1.4 },
	{ label: 'Industry average', pue: 1.56 }
];

export const DEFAULT_PUE = 1.2;
export const DEFAULT_RENEWABLE_MATCH = 0.9;
const DEFAULT_IDLE_FRAC = 0.3;
const DEFAULT_HOST_OVERHEAD_FRAC = 0.3;

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

/** Label and grid intensity for a region id ('custom' uses the given figure). */
export function gridFor(region: string, customGCo2PerKwh?: number): GridMix {
	if (region === 'custom')
		return { id: 'custom', label: 'Custom grid', gCo2PerKwh: Math.max(0, customGCo2PerKwh ?? 0) };
	return GRIDS_BY_ID.get(region) ?? GRIDS[0];
}

export function estimateEnergy(i: EnergyInputs): EnergyResult {
	const pue = Math.max(1, i.pue ?? DEFAULT_PUE);
	const idleFrac = i.idleFrac ?? DEFAULT_IDLE_FRAC;
	const hostOverhead = i.hostOverheadFrac ?? DEFAULT_HOST_OVERHEAD_FRAC;
	const method: EmissionsMethod = i.method ?? 'location';
	const renewableMatch = clamp01(i.renewableMatch ?? DEFAULT_RENEWABLE_MATCH);
	const util = clamp01(i.util);
	const grid = gridFor(i.region, i.customGCo2PerKwh);

	// linear scale between idle and TDP
	const perGpuWatts = i.gpu.tdpWatts * (idleFrac + (1 - idleFrac) * util);
	const hostWatts = perGpuWatts * hostOverhead;
	const rawKw = ((perGpuWatts + hostWatts) * i.numGpus) / 1000;
	const facilityKw = rawKw * pue;

	const kwhPerHour = facilityKw;
	const kwhPerDay = kwhPerHour * 24;
	const kwhPerYear = kwhPerHour * 8760;

	const intensity = method === 'market' ? grid.gCo2PerKwh * (1 - renewableMatch) : grid.gCo2PerKwh;

	return {
		perGpuWatts,
		hostWatts,
		facilityKw,
		kwhPerHour,
		kwhPerDay,
		kwhPerYear,
		gCo2PerHour: kwhPerHour * intensity,
		gCo2PerYear: kwhPerYear * intensity,
		regionLabel: grid.label,
		gCo2PerKwh: intensity,
		method,
		renewableMatch,
		pue
	};
}

/** kg CO2e per 1M tokens, given a tokens/sec throughput. */
export function co2PerMillionTokens(e: EnergyResult, tokensPerSec: number): number | null {
	if (tokensPerSec <= 0) return null;
	const gPerSec = e.gCo2PerHour / 3600;
	return ((gPerSec / tokensPerSec) * 1e6) / 1000; // g → kg
}
/** kg CO2e per 1k images/clips, given a rate. */
export function co2Per1kUnits(e: EnergyResult, perSec: number): number | null {
	if (perSec <= 0) return null;
	const gPerSec = e.gCo2PerHour / 3600;
	return ((gPerSec / perSec) * 1e3) / 1000;
}
/** kg CO2e for a training run of a given wall-clock duration. */
export function co2ForRun(e: EnergyResult, hours: number): number {
	return (e.gCo2PerHour * hours) / 1000;
}
