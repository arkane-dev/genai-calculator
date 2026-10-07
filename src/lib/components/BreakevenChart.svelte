<script lang="ts">
	import InfoTip from './InfoTip.svelte';

	interface Props {
		unitCostMonthly: number; // $/mo for ONE sized cluster (the step height)
		capReqPerMonth: number; // requests/month one cluster serves flat-out (100% duty) — the step width
		apiCostPerRequest: number; // slope of the API line
		breakevenReqPerMonth: number | null; // tread-1 crossover, only when self-host can win
		currentReqPerMonth: number;
		regime: 'crossover' | 'api-always' | 'self-host-always';
		fmtUsd: (n: number) => string;
		fmtReq: (n: number) => string;
		apiViolatesSlo?: boolean; // API can't meet the per-user throughput SLO
		apiSloNote?: string; // what specifically it fails (shown in the overlay)
	}
	let {
		unitCostMonthly,
		capReqPerMonth,
		apiCostPerRequest,
		breakevenReqPerMonth,
		currentReqPerMonth,
		regime,
		fmtUsd,
		fmtReq,
		apiViolatesSlo = false,
		apiSloNote = ''
	}: Props = $props();

	// API turns red when it can't actually deliver the SLO — cost is moot if it can't serve it.
	const apiColor = $derived(apiViolatesSlo ? '#ff3b52' : '#3ff0b8');

	const W = 640;
	const H = 300;
	const PAD = { l: 64, r: 16, t: 16, b: 40 };
	const plotW = W - PAD.l - PAD.r;
	const plotH = H - PAD.t - PAD.b;

	// Show enough of the x-axis to make the staircase legible: a few cluster steps,
	// past the current demand and (when self-host can win) past the crossover.
	const xMax = $derived(
		Math.max(currentReqPerMonth * 1.4, (breakevenReqPerMonth ?? 0) * 1.4, capReqPerMonth * 3.2, 1)
	);
	// clusters needed at the far right → tallest the staircase reaches
	const maxClusters = $derived(Math.max(1, Math.ceil(xMax / Math.max(capReqPerMonth, 1))));
	const yMax = $derived(
		Math.max(maxClusters * unitCostMonthly, apiCostPerRequest * xMax, 1) * 1.08
	);

	const xOf = (req: number) => PAD.l + (req / xMax) * plotW;
	const yOf = (usd: number) => PAD.t + plotH - (usd / yMax) * plotH;

	// Self-host staircase: to serve volume V you need ceil(V / cap) clusters, each
	// costing unitCostMonthly. Cost jumps every time volume fills another cluster.
	const stairPts = $derived.by(() => {
		const cap = Math.max(capReqPerMonth, 1);
		const pts: string[] = [`${xOf(0)},${yOf(unitCostMonthly)}`];
		for (let k = 1; k <= maxClusters; k++) {
			const edge = Math.min(k * cap, xMax);
			pts.push(`${xOf(edge)},${yOf(k * unitCostMonthly)}`); // across the tread
			if (k * cap < xMax) pts.push(`${xOf(edge)},${yOf((k + 1) * unitCostMonthly)}`); // step up
		}
		return pts.join(' ');
	});

	const apiLine = $derived(`${xOf(0)},${yOf(0)} ${xOf(xMax)},${yOf(apiCostPerRequest * xMax)}`);
	const beX = $derived(
		breakevenReqPerMonth != null && regime === 'crossover' ? xOf(breakevenReqPerMonth) : null
	);
	const curX = $derived(xOf(Math.min(currentReqPerMonth, xMax)));
	const curSelfY = $derived(yOf(unitCostMonthly)); // current demand is always within the first cluster
	const curApiY = $derived(yOf(Math.min(apiCostPerRequest * currentReqPerMonth, yMax)));

	const xTicks = $derived([0, 0.25, 0.5, 0.75, 1].map((f) => f * xMax));
	const yTicks = $derived([0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax));
</script>

<div class="panel">
	<div class="panel-title spaced row wrap">
		Cost vs volume
		{#if apiViolatesSlo}
			<span class="pill bad"
				>⚠ API can’t meet your SLO</span
			>
		{/if}
		<InfoTip
			text="The API (green) is a straight line up from zero — you pay per token. Self-hosting (blue) is a staircase: one cluster serves only so much traffic, so as volume grows you add servers and the cost jumps a step. Each step is another cluster. Where the API line sits below the staircase, the API is cheaper; where it rises above, self-hosting wins. The dot is your demand today."
		/>
	</div>
	<svg
		viewBox="0 0 {W} {H}"
		class="chart"
		role="img"
		aria-label="Monthly cost versus request volume"
	>
		<!-- gridlines -->
		{#each yTicks as ty (ty)}
			<line x1={PAD.l} y1={yOf(ty)} x2={W - PAD.r} y2={yOf(ty)} stroke="#12163a" stroke-width="1" />
			<text x={PAD.l - 8} y={yOf(ty) + 3} text-anchor="end" class="t-mute"
				>{fmtUsd(ty)}</text
			>
		{/each}
		{#each xTicks as tx (tx)}
			<text x={xOf(tx)} y={H - PAD.b + 16} text-anchor="middle" class="t-mute"
				>{fmtReq(tx)}</text
			>
		{/each}

		<!-- crossover marker (only when self-host can actually win) -->
		{#if beX !== null}
			<line
				x1={beX}
				y1={PAD.t}
				x2={beX}
				y2={PAD.t + plotH}
				stroke="#f6bd6a"
				stroke-width="1.5"
				stroke-dasharray="4 3"
			/>
			<text x={beX} y={PAD.t + 10} text-anchor="middle" class="t-warn"
				>break-even</text
			>
		{/if}

		<!-- self-host: staircase (each step = +1 cluster) -->
		<polyline points={stairPts} fill="none" stroke="#22f2f7" stroke-width="2" />
		<!-- API: linear from origin (red + dashed when it can't meet the SLO) -->
		<polyline
			points={apiLine}
			fill="none"
			stroke={apiColor}
			stroke-width={apiViolatesSlo ? 2.5 : 2}
			stroke-dasharray={apiViolatesSlo ? '6 4' : 'none'}
		/>

		<!-- current-demand marker -->
		<line
			x1={curX}
			y1={PAD.t}
			x2={curX}
			y2={PAD.t + plotH}
			stroke="#ecebff"
			stroke-width="1"
			stroke-dasharray="2 3"
			opacity="0.5"
		/>
		<circle cx={curX} cy={curSelfY} r="4" fill="#22f2f7" />
		<circle cx={curX} cy={curApiY} r="4" fill={apiColor} />

		<!-- can't-serve overlay: cost is irrelevant if the API can't hit the SLO -->
		{#if apiViolatesSlo}
			<rect x={PAD.l} y={PAD.t} width={plotW} height="24" fill="#4a1020" opacity="0.4" />
			<text
				x={PAD.l + plotW / 2}
				y={PAD.t + 16}
				text-anchor="middle"
				class="t-bad label-strong"
			>
				{apiSloNote || 'API can’t meet your throughput SLO — the price comparison is moot'}
			</text>
		{/if}

		<!-- axes -->
		<line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={PAD.t + plotH} stroke="#5659a4" stroke-width="1" />
		<line
			x1={PAD.l}
			y1={PAD.t + plotH}
			x2={W - PAD.r}
			y2={PAD.t + plotH}
			stroke="#5659a4"
			stroke-width="1"
		/>
	</svg>

	<div class="legend-bar">
		<div class="legend">
			<span
				><span class="swatch line" style="background:#22f2f7"></span>Self-host
				(steps = +1 cluster)</span
			>
			<span class:bad={apiViolatesSlo}
				><span class="swatch line" style:background={apiColor}></span>API (per
				token){apiViolatesSlo ? ' — below SLO' : ''}</span
			>
			<span class="row dim"
				><span class="swatch demand"></span>your demand</span
			>
		</div>
		<span class="mute">x: requests / month · y: $ / month</span>
	</div>
</div>

<style>
	.demand { width: 0.625rem; height: 0.625rem; background: color-mix(in srgb, var(--nd-text) 70%, transparent); }
</style>
