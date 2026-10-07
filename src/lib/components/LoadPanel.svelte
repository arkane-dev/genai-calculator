<script lang="ts">
	import InfoTip from './InfoTip.svelte';
	import Toggle from './Toggle.svelte';
	import { estimateLoad, SOURCES } from '$lib/load/estimate';
	import { GPUS_BY_ID } from '$lib/profiler/data';

	interface Props {
		gpuId: string;
		numGpus: number;
		weightBytes: number; // total model weight bytes (before sharding)
	}
	let { gpuId, numGpus, weightBytes }: Props = $props();

	let sourceId = $state('parallel-fs');
	let sharded = $state(true);

	const gpu = $derived(GPUS_BY_ID.get(gpuId));
	const r = $derived(
		gpu
			? estimateLoad({
					weightBytes,
					numGpus,
					pcieGBs: gpu.pcieGBs,
					sourceId,
					sharded,
					shardWaste: 1
				})
			: null
	);

	const fmtBytes = (b: number) =>
		b >= 1e12
			? `${(b / 1e12).toFixed(1)} TB`
			: b >= 1e9
				? `${(b / 1e9).toFixed(1)} GB`
				: `${(b / 1e6).toFixed(0)} MB`;
	const fmtSec = (s: number) =>
		s < 60
			? `${s.toFixed(1)} s`
			: s < 3600
				? `${(s / 60).toFixed(1)} min`
				: `${(s / 3600).toFixed(2)} h`;
</script>

<section class="rounded-xl border border-slate-700 bg-slate-900/60 p-5">
	<div class="mb-3 flex items-center gap-1.5 text-sm font-medium text-slate-200">
		Cold-start / model load
		<InfoTip
			text="Time to stream weights from storage into GPU HBM at startup. Two things bound the pipe: the storage source (object storage, block storage, local NVMe, a parallel file system) and the PCIe host↔GPU link on each GPU. A properly parallelised loader saturates the slower of the two. Real deployments overlap network + PCIe so wall-clock ≈ weights ÷ min(source, PCIe). This is where Gen5 PCIe (~2× Gen4) pays off, and where the storage class you pick shows up in tokens-per-dollar."
		/>
	</div>

	{#if !r || !gpu}
		<p class="text-sm text-slate-400">Pick a GPU with a PCIe rate.</p>
	{:else}
		<div class="mb-4 flex flex-wrap items-center gap-3">
			<label class="flex items-center gap-2 text-xs text-slate-400">
				Weight source
				<select
					bind:value={sourceId}
					class="rounded-md border border-slate-600 bg-slate-800 px-2 py-1 text-xs text-slate-100"
				>
					{#each SOURCES as s (s.id)}<option value={s.id}>{s.label}</option>{/each}
				</select>
			</label>
			<Toggle
				label="Sharded load"
				bind:checked={sharded}
				hint="FSDP / tensor-parallel"
				info="Sharded (FSDP / tensor-parallel): each GPU pulls only its shard, so load time drops with GPU count. Data-parallel (off): every GPU pulls all the weights."
			/>
		</div>

		<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
			<div class="rounded-lg border border-slate-700 bg-slate-800/40 p-3">
				<div class="text-[11px] tracking-wide text-slate-500 uppercase">Cold start</div>
				<div class="mt-1 font-mono text-lg text-slate-100">{fmtSec(r.seconds)}</div>
				<div class="mt-0.5 text-xs text-slate-400">
					{r.bottleneck === 'pcie' ? 'PCIe-bound' : 'source-bound'}
				</div>
			</div>
			<div class="rounded-lg border border-slate-700 bg-slate-800/40 p-3">
				<div class="text-[11px] tracking-wide text-slate-500 uppercase">Per-GPU bytes</div>
				<div class="mt-1 font-mono text-lg text-slate-100">{fmtBytes(r.perGpuBytes)}</div>
				<div class="mt-0.5 text-xs text-slate-400">
					{sharded ? `1/${numGpus} of the model` : 'full model'}
				</div>
			</div>
			<div class="rounded-lg border border-slate-700 bg-slate-800/40 p-3">
				<div class="text-[11px] tracking-wide text-slate-500 uppercase">PCIe (node)</div>
				<div class="mt-1 font-mono text-lg text-slate-100">{r.nodePcieGBs} GB/s</div>
				<div class="mt-0.5 text-xs text-slate-400">
					{r.pcieGBs} per GPU × {Math.min(8, numGpus)}
				</div>
			</div>
			<div class="rounded-lg border border-slate-700 bg-slate-800/40 p-3">
				<div class="text-[11px] tracking-wide text-slate-500 uppercase">Source</div>
				<div class="mt-1 font-mono text-lg text-slate-100">{r.sourceGBs} GB/s</div>
				<div class="mt-0.5 truncate text-xs text-slate-400" title={r.sourceLabel}>
					{r.sourceLabel}
				</div>
			</div>
		</div>
		<p class="mt-2 text-[10px] text-slate-600">
			Wall-clock ≈ weights ÷ min(source, node PCIe). Sharded assumes a well-parallelised loader
			(e.g. FSDP with parallel range reads). This is where Gen5 PCIe hosts (H100 and newer) pay off
			vs Gen4 (A100, L4, L40S): ~2× faster once the source can keep up.
		</p>
	{/if}
</section>
