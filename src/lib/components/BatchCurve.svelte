<script lang="ts">
	import InfoTip from './InfoTip.svelte';

	export interface BatchPoint {
		batch: number;
		throughputTps: number; // aggregate across the deployment
		perUserTps: number; // single-sequence decode speed
		ttftMs: number;
		per1M: number | null; // $ / 1M output tokens at this batch (estimate), null if no price
	}
	interface Props {
		points: BatchPoint[];
		currentBatch: number;
		fmtTps: (n: number) => string;
		fmtUsdSmall: (n: number) => string;
	}
	let { points, currentBatch, fmtTps, fmtUsdSmall }: Props = $props();

	const W = 640;
	const H = 300;
	const PAD = { l: 60, r: 56, t: 16, b: 40 };
	const plotW = W - PAD.l - PAD.r;
	const plotH = H - PAD.t - PAD.b;

	const n = $derived(points.length);
	const tpMax = $derived(Math.max(...points.map((p) => p.throughputTps), 1));
	const puMax = $derived(Math.max(...points.map((p) => p.perUserTps), 1));

	// x is the point index (batches aren't linearly spaced; even spacing keeps the curve readable)
	const xOf = (i: number) => PAD.l + (n <= 1 ? 0 : (i / (n - 1)) * plotW);
	const yTp = (v: number) => PAD.t + plotH - (v / tpMax) * plotH;
	const yPu = (v: number) => PAD.t + plotH - (v / puMax) * plotH;

	const tpLine = $derived(points.map((p, i) => `${xOf(i)},${yTp(p.throughputTps)}`).join(' '));
	const puLine = $derived(points.map((p, i) => `${xOf(i)},${yPu(p.perUserTps)}`).join(' '));

	const curIdx = $derived(points.findIndex((p) => p.batch === currentBatch));
	const cur = $derived(curIdx >= 0 ? points[curIdx] : null);
	// sweet spot: the batch capturing most of the throughput gain — first point at ≥90% of peak
	const kneeIdx = $derived(points.findIndex((p) => p.throughputTps >= 0.9 * tpMax));
	const knee = $derived(kneeIdx >= 0 ? points[kneeIdx] : null);

	// x tick labels: a handful of batch values across the range
	const tickIdxs = $derived(
		n <= 6
			? points.map((_, i) => i)
			: [0, Math.round(n / 4), Math.round(n / 2), Math.round((3 * n) / 4), n - 1]
	);
</script>

<section class="rounded-xl border border-slate-700 bg-slate-900/60 p-5">
	<div class="mb-2 flex items-center gap-1.5 text-sm font-medium text-slate-200">
		Batch sweep — throughput vs per-user speed
		<InfoTip
			text="Holds this exact config and sweeps the batch size (how many requests share the GPU at once). Bigger batches pack the GPU more fully, so aggregate throughput (teal) climbs then saturates — and cost per token falls with it. But each user's decode speed (amber) drops, because they share the same memory reads. The 'knee' is where throughput stops improving much: past it you lose per-user speed for little extra throughput. The dashed line is your current batch."
		/>
	</div>

	{#if n < 2}
		<p class="text-sm text-slate-400">Not enough fitting batch sizes to plot a curve.</p>
	{:else}
		<svg
			viewBox="0 0 {W} {H}"
			class="w-full"
			role="img"
			aria-label="Throughput and per-user speed versus batch size"
		>
			<!-- horizontal gridlines -->
			{#each [0, 0.25, 0.5, 0.75, 1] as f (f)}
				<line
					x1={PAD.l}
					y1={PAD.t + plotH - f * plotH}
					x2={W - PAD.r}
					y2={PAD.t + plotH - f * plotH}
					stroke="#12163a"
					stroke-width="1"
				/>
			{/each}

			<!-- knee marker -->
			{#if knee}
				<line
					x1={xOf(kneeIdx)}
					y1={PAD.t}
					x2={xOf(kneeIdx)}
					y2={PAD.t + plotH}
					stroke="#3ff0b8"
					stroke-width="1.5"
					stroke-dasharray="3 3"
					opacity="0.6"
				/>
				<text
					x={xOf(kneeIdx)}
					y={PAD.t + 10}
					text-anchor="middle"
					class="fill-emerald-400 text-[10px]">sweet spot</text
				>
			{/if}

			<!-- current-batch marker -->
			<line
				x1={xOf(curIdx < 0 ? 0 : curIdx)}
				y1={PAD.t}
				x2={xOf(curIdx < 0 ? 0 : curIdx)}
				y2={PAD.t + plotH}
				stroke="#ecebff"
				stroke-width="1"
				stroke-dasharray="2 3"
				opacity="0.5"
			/>

			<!-- throughput (left axis) -->
			<polyline points={tpLine} fill="none" stroke="#ff2bd6" stroke-width="2" />
			<!-- per-user (right axis) -->
			<polyline points={puLine} fill="none" stroke="#f5ec58" stroke-width="2" />

			{#if cur}
				<circle cx={xOf(curIdx)} cy={yTp(cur.throughputTps)} r="4" fill="#ff2bd6" />
				<circle cx={xOf(curIdx)} cy={yPu(cur.perUserTps)} r="4" fill="#f5ec58" />
			{/if}

			<!-- axis labels -->
			<text x={PAD.l - 8} y={PAD.t + 4} text-anchor="end" class="fill-teal-400 text-[10px]"
				>{fmtTps(tpMax)}</text
			>
			<text x={PAD.l - 8} y={PAD.t + plotH} text-anchor="end" class="fill-teal-400 text-[10px]"
				>0</text
			>
			<text x={W - PAD.r + 8} y={PAD.t + 4} text-anchor="start" class="fill-amber-400 text-[10px]"
				>{puMax.toFixed(0)}</text
			>
			<text
				x={W - PAD.r + 8}
				y={PAD.t + plotH}
				text-anchor="start"
				class="fill-amber-400 text-[10px]">0</text
			>
			{#each tickIdxs as i (i)}
				<text x={xOf(i)} y={H - PAD.b + 16} text-anchor="middle" class="fill-slate-500 text-[10px]"
					>{points[i].batch}</text
				>
			{/each}

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
			<line
				x1={W - PAD.r}
				y1={PAD.t}
				x2={W - PAD.r}
				y2={PAD.t + plotH}
				stroke="#5659a4"
				stroke-width="1"
			/>
		</svg>

		<div class="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs">
			<div class="flex items-center gap-4">
				<span class="flex items-center gap-1.5 text-slate-300"
					><span class="inline-block h-2 w-4 rounded" style="background:#ff2bd6"></span>Aggregate
					throughput</span
				>
				<span class="flex items-center gap-1.5 text-slate-300"
					><span class="inline-block h-2 w-4 rounded" style="background:#f5ec58"></span>Per-user
					speed</span
				>
			</div>
			<span class="text-slate-500">x: batch size (requests in flight)</span>
		</div>

		{#if cur}
			<div class="mt-3 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
				<div class="rounded border border-slate-800 bg-slate-900/40 p-2">
					<div class="text-slate-500">At batch {cur.batch}</div>
					<div class="font-mono text-slate-200">{fmtTps(cur.throughputTps)} tok/s</div>
				</div>
				<div class="rounded border border-slate-800 bg-slate-900/40 p-2">
					<div class="text-slate-500">Per user</div>
					<div class="font-mono text-slate-200">{cur.perUserTps.toFixed(1)} tok/s</div>
				</div>
				<div class="rounded border border-slate-800 bg-slate-900/40 p-2">
					<div class="text-slate-500">TTFT</div>
					<div class="font-mono text-slate-200">{cur.ttftMs.toFixed(0)} ms</div>
				</div>
				<div class="rounded border border-slate-800 bg-slate-900/40 p-2">
					<div class="text-slate-500">Cost / 1M tok</div>
					<div class="font-mono text-slate-200">
						{cur.per1M == null ? '—' : fmtUsdSmall(cur.per1M)}
					</div>
				</div>
			</div>
		{/if}
		{#if knee}
			<p class="mt-2 text-[10px] text-slate-500">
				Sweet spot ≈ batch {knee.batch}: {fmtTps(knee.throughputTps)} tok/s aggregate at
				{knee.perUserTps.toFixed(0)} tok/s per user. Beyond it, throughput is within 10% of peak while
				per-user speed keeps falling. Estimate, single replica of this config.
			</p>
		{/if}
	{/if}
</section>
