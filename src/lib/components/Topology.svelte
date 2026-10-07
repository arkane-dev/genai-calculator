<script lang="ts">
	import { COLORS } from '$lib/profiler/data';
	import type { Profile } from '$lib/profiler/types';
	import Pipe from './Pipe.svelte';

	interface Props {
		p: Profile;
		hasNvlink: boolean;
	}
	let { p, hasNvlink }: Props = $props();

	const MAX_NODES = 6;
	const MAX_GPN = 8;

	const c = $derived(p.cluster);
	const shownNodes = $derived(Math.min(c.numNodes, MAX_NODES));
	const gpn = $derived(Math.min(c.gpusPerNode, MAX_GPN));

	const gpusInNode = (n: number) =>
		Math.max(0, Math.min(c.gpusPerNode, c.numGpus - n * c.gpusPerNode));
	const sameTpGroup = (a: number, b: number) => Math.floor(a / c.tp) === Math.floor(b / c.tp);

	const badge = $derived(
		c.scaleOut === 'single'
			? { text: 'Single node', tone: 'text-slate-300' }
			: c.scaleOut === 'data-parallel'
				? { text: 'Data-parallel scale-out', tone: 'text-emerald-400' }
				: { text: 'Model-parallel across nodes', tone: 'text-amber-400' }
	);
	const parallelLabel = $derived(
		`TP ${c.tp}` +
			(c.pp > 1 ? ` · PP ${c.pp}` : '') +
			(c.ep > 1 ? ` · EP ${c.ep}` : '') +
			` · DP ${c.dp}`
	);
</script>

<div class="rounded-lg border border-slate-600 bg-slate-900 p-4">
	<div class="mb-3 flex flex-wrap items-baseline justify-between gap-2">
		<span class="text-sm font-medium text-slate-200">Cluster topology</span>
		<span class="font-mono text-xs text-slate-400">
			{c.numNodes} node × {c.gpusPerNode} GPU · {parallelLabel}
			<span class={badge.tone}> · {badge.text}</span>
		</span>
	</div>

	{#if c.numNodes > 1}
		<!-- shared inter-node fabric spine -->
		<div class="mb-3" style:opacity={c.crossesFabric ? 1 : 0.4}>
			<Pipe
				frac={p.netFrac}
				color={COLORS.network}
				label="Inter-node fabric · {p.fabric.label}"
				detail={c.crossesFabric
					? `${p.netUsedGBs.toFixed(1)} / ${p.netPeakGBs.toFixed(0)} GB/s`
					: 'idle'}
			/>
		</div>
	{/if}

	<!-- nodes: 2 wide, then wrap down -->
	<div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
		{#each Array(shownNodes) as _, n (n)}
			<div class="overflow-hidden rounded-lg border border-slate-600 bg-slate-800/40 p-2">
				<div
					class="mb-2 flex items-center justify-between text-[11px] tracking-wide text-slate-400 uppercase"
				>
					<span>Node {n}</span>
					{#if c.numNodes > 1}
						<span
							class="h-1.5 w-1.5 rounded-full"
							style:background-color={c.crossesFabric ? COLORS.network : '#5659a4'}
						></span>
					{/if}
				</div>
				<div class="flex items-center gap-0.5">
					{#each Array(Math.min(gpn, gpusInNode(n))) as _, col (col)}
						{@const g = n * c.gpusPerNode + col}
						<div
							class="flex h-9 min-w-0 flex-1 items-center justify-center truncate rounded-md border px-0.5 text-[10px] font-medium"
							style:border-color="{COLORS.nvlink}66"
							style:color={COLORS.nvlink}
							style:background-color="{COLORS.nvlink}18"
						>
							g{g}
						</div>
						{#if col < Math.min(gpn, gpusInNode(n)) - 1}
							{@const live = c.tp > 1 && sameTpGroup(g, g + 1)}
							<div class="w-2.5 shrink-0" style:opacity={live ? 1 : 0.2}>
								<Pipe frac={live ? p.nvlinkFrac : 0} color={COLORS.nvlink} />
							</div>
						{/if}
					{/each}
				</div>
			</div>
		{/each}
	</div>
	{#if c.numNodes > MAX_NODES}
		<div class="mt-2 text-xs text-slate-500">+{c.numNodes - MAX_NODES} more nodes</div>
	{/if}

	<!-- legend -->
	<div class="mt-3 grid gap-1 text-xs">
		{#if !p.comm.tpActive && !p.comm.epActive && !p.comm.ppActive}
			<p class="text-slate-500">No model parallelism — GPUs run independent replicas.</p>
		{/if}

		{#if p.comm.tpActive}
			<div class="flex items-start gap-2">
				<span
					class="mt-1 inline-block h-2 w-4 shrink-0 rounded-full"
					style:background-color={p.comm.tpOverFabric ? COLORS.network : COLORS.nvlink}
				></span>
				<span class="text-slate-400">
					TP all-reduce over {p.comm.tpOverFabric
						? p.fabric.label + ' fabric'
						: hasNvlink
							? 'NVLink'
							: 'PCIe'}
					· <span class="font-mono">{p.comm.tpGBs.toFixed(1)} GB/s</span>
					{#if p.comm.tpOverFabric}<span class="text-amber-400"> — crosses nodes</span>{/if}
				</span>
			</div>
		{/if}

		{#if p.comm.epActive}
			<div class="flex items-start gap-2">
				<span
					class="mt-1 inline-block h-2 w-4 shrink-0 rounded-full"
					style:background-color={p.comm.epOverFabric ? COLORS.network : COLORS.nvlink}
				></span>
				<span class="text-slate-400">
					EP all-to-all over {p.comm.epOverFabric
						? p.fabric.label + ' fabric'
						: hasNvlink
							? 'NVLink'
							: 'PCIe'}
					· <span class="font-mono">{p.comm.epGBs.toFixed(1)} GB/s</span>
					{#if p.comm.epOverFabric}
						<span class={p.fabric.kind === 'multipath' ? 'text-teal-300' : 'text-amber-400'}>
							— sustains {Math.round(p.fabric.a2aEff * 100)}% of peak
							{#if p.fabric.kind === 'multipath'}(multipath RDMA sprays packets across paths){:else}(flow-routed,
								collides on all-to-all){/if}
						</span>
					{/if}
				</span>
			</div>
		{/if}

		{#if p.comm.ppActive}
			<div class="flex items-start gap-2">
				<span
					class="mt-1 inline-block h-2 w-4 shrink-0 rounded-full"
					style:background-color={p.comm.ppOverFabric ? COLORS.network : COLORS.nvlink}
				></span>
				<span class="text-slate-400">
					PP hand-off over {p.comm.ppOverFabric
						? p.fabric.label + ' fabric'
						: hasNvlink
							? 'NVLink'
							: 'PCIe'}
					· <span class="font-mono">{p.comm.ppGBs.toFixed(2)} GB/s</span>
					<span class="text-slate-500"> — point-to-point, tiny</span>
				</span>
			</div>
		{/if}

		{#if c.numNodes > 1 && !c.crossesFabric}
			<p class="text-emerald-400">
				Fabric idle — every model-parallel group fits inside one node; extra nodes are independent
				replicas.
			</p>
		{/if}
	</div>
</div>
