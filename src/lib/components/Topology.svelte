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
			? { text: 'Single node', tone: 'dim' }
			: c.scaleOut === 'data-parallel'
				? { text: 'Data-parallel scale-out', tone: 'ok' }
				: { text: 'Model-parallel across nodes', tone: 'warn' }
	);
	const parallelLabel = $derived(
		`TP ${c.tp}` +
			(c.pp > 1 ? ` · PP ${c.pp}` : '') +
			(c.ep > 1 ? ` · EP ${c.ep}` : '') +
			` · DP ${c.dp}`
	);
</script>

<div class="topo">
	<div class="kv head">
		<span class="title">Cluster topology</span>
		<span class="num small dim">
			{c.numNodes} node × {c.gpusPerNode} GPU · {parallelLabel}
			<span class={badge.tone}> · {badge.text}</span>
		</span>
	</div>

	{#if c.numNodes > 1}
		<!-- shared inter-node fabric spine -->
		<div class="spine" style:opacity={c.crossesFabric ? 1 : 0.4}>
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
	<div class="nodes">
		{#each Array(shownNodes) as _, n (n)}
			<div class="node">
				<div class="node-head">
					<span>Node {n}</span>
					{#if c.numNodes > 1}
						<span
							class="dot"
							style:background-color={c.crossesFabric ? COLORS.network : '#5659a4'}
						></span>
					{/if}
				</div>
				<div class="gpus">
					{#each Array(Math.min(gpn, gpusInNode(n))) as _, col (col)}
						{@const g = n * c.gpusPerNode + col}
						<div
							class="gpu"
							style:border-color="{COLORS.nvlink}66"
							style:color={COLORS.nvlink}
							style:background-color="{COLORS.nvlink}18"
						>
							g{g}
						</div>
						{#if col < Math.min(gpn, gpusInNode(n)) - 1}
							{@const live = c.tp > 1 && sameTpGroup(g, g + 1)}
							<div class="link" style:opacity={live ? 1 : 0.2}>
								<Pipe frac={live ? p.nvlinkFrac : 0} color={COLORS.nvlink} />
							</div>
						{/if}
					{/each}
				</div>
			</div>
		{/each}
	</div>
	{#if c.numNodes > MAX_NODES}
		<div class="more small mute">+{c.numNodes - MAX_NODES} more nodes</div>
	{/if}

	<!-- legend -->
	<div class="key small">
		{#if !p.comm.tpActive && !p.comm.epActive && !p.comm.ppActive}
			<p class="mute">No model parallelism — GPUs run independent replicas.</p>
		{/if}

		{#if p.comm.tpActive}
			<div class="key-row">
				<span
					class="swatch line"
					style:background-color={p.comm.tpOverFabric ? COLORS.network : COLORS.nvlink}
				></span>
				<span class="dim">
					TP all-reduce over {p.comm.tpOverFabric
						? p.fabric.label + ' fabric'
						: hasNvlink
							? 'NVLink'
							: 'PCIe'}
					· <span class="num">{p.comm.tpGBs.toFixed(1)} GB/s</span>
					{#if p.comm.tpOverFabric}<span class="warn"> — crosses nodes</span>{/if}
				</span>
			</div>
		{/if}

		{#if p.comm.epActive}
			<div class="key-row">
				<span
					class="swatch line"
					style:background-color={p.comm.epOverFabric ? COLORS.network : COLORS.nvlink}
				></span>
				<span class="dim">
					EP all-to-all over {p.comm.epOverFabric
						? p.fabric.label + ' fabric'
						: hasNvlink
							? 'NVLink'
							: 'PCIe'}
					· <span class="num">{p.comm.epGBs.toFixed(1)} GB/s</span>
					{#if p.comm.epOverFabric}
						<span class={p.fabric.kind === 'multipath' ? 'accent' : 'warn'}>
							— sustains {Math.round(p.fabric.a2aEff * 100)}% of peak
							{#if p.fabric.kind === 'multipath'}(multipath RDMA sprays packets across paths){:else}(flow-routed,
								collides on all-to-all){/if}
						</span>
					{/if}
				</span>
			</div>
		{/if}

		{#if p.comm.ppActive}
			<div class="key-row">
				<span
					class="swatch line"
					style:background-color={p.comm.ppOverFabric ? COLORS.network : COLORS.nvlink}
				></span>
				<span class="dim">
					PP hand-off over {p.comm.ppOverFabric
						? p.fabric.label + ' fabric'
						: hasNvlink
							? 'NVLink'
							: 'PCIe'}
					· <span class="num">{p.comm.ppGBs.toFixed(2)} GB/s</span>
					<span class="mute"> — point-to-point, tiny</span>
				</span>
			</div>
		{/if}

		{#if c.numNodes > 1 && !c.crossesFabric}
			<p class="ok">
				Fabric idle — every model-parallel group fits inside one node; extra nodes are independent
				replicas.
			</p>
		{/if}
	</div>
</div>

<style>
	.topo { padding: var(--nd-space-4); border: 1px solid var(--nd-line-strong); background: var(--nd-surface-1); }
	.head { flex-wrap: wrap; margin-bottom: var(--nd-space-3); }
	.title { color: var(--nd-text); font-size: var(--nd-text-sm); font-weight: 500; }
	.accent { color: var(--nd-accent); }
	.spine { margin-bottom: var(--nd-space-3); }
	.nodes { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--nd-space-3); }
	@media (min-width: 640px) { .nodes { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
	.node {
		overflow: hidden;
		padding: var(--nd-space-2);
		border: 1px solid var(--nd-line-strong);
		background: color-mix(in srgb, var(--nd-surface-2) 40%, transparent);
	}
	.node-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: var(--nd-space-2);
		color: var(--nd-text-dim);
		font-size: 0.6875rem;
		letter-spacing: 0.05em;
		text-transform: uppercase;
	}
	.dot { width: 0.375rem; height: 0.375rem; }
	.gpus { display: flex; align-items: center; gap: 2px; }
	.gpu {
		display: flex;
		flex: 1;
		align-items: center;
		justify-content: center;
		min-width: 0;
		height: 2.25rem;
		overflow: hidden;
		padding: 0 2px;
		border: 1px solid;
		font-size: 0.625rem;
		font-weight: 500;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.link { flex: none; width: 0.625rem; }
	.more { margin-top: var(--nd-space-2); }
	.key { display: grid; gap: var(--nd-space-1); margin-top: var(--nd-space-3); }
	.key p { margin: 0; }
	.key-row { display: flex; align-items: flex-start; gap: var(--nd-space-2); }
	.key-row .swatch { margin-top: var(--nd-space-1); }
</style>
