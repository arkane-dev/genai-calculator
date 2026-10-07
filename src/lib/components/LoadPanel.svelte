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

<section class="panel">
	<div class="panel-title spaced">
		Cold-start / model load
		<InfoTip
			text="Time to stream weights from storage into GPU HBM at startup. Two things bound the pipe: the storage source (object storage, block storage, local NVMe, a parallel file system) and the PCIe host↔GPU link on each GPU. A properly parallelised loader saturates the slower of the two. Real deployments overlap network + PCIe so wall-clock ≈ weights ÷ min(source, PCIe). This is where Gen5 PCIe (~2× Gen4) pays off, and where the storage class you pick shows up in tokens-per-dollar."
		/>
	</div>

	{#if !r || !gpu}
		<p class="size-sm dim">Pick a GPU with a PCIe rate.</p>
	{:else}
		<div class="row wrap spaced-l">
			<label class="inline-field">
				Weight source
				<select
					bind:value={sourceId}
					class="input compact"
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

		<div class="tiles">
			<div class="tile">
				<div class="tile-label">Cold start</div>
				<div class="tile-value">{fmtSec(r.seconds)}</div>
				<div class="tile-sub">
					{r.bottleneck === 'pcie' ? 'PCIe-bound' : 'source-bound'}
				</div>
			</div>
			<div class="tile">
				<div class="tile-label">Per-GPU bytes</div>
				<div class="tile-value">{fmtBytes(r.perGpuBytes)}</div>
				<div class="tile-sub">
					{sharded ? `1/${numGpus} of the model` : 'full model'}
				</div>
			</div>
			<div class="tile">
				<div class="tile-label">PCIe (node)</div>
				<div class="tile-value">{r.nodePcieGBs} GB/s</div>
				<div class="tile-sub">
					{r.pcieGBs} per GPU × {Math.min(8, numGpus)}
				</div>
			</div>
			<div class="tile">
				<div class="tile-label">Source</div>
				<div class="tile-value">{r.sourceGBs} GB/s</div>
				<div class="tile-sub truncate" title={r.sourceLabel}>
					{r.sourceLabel}
				</div>
			</div>
		</div>
		<p class="note">
			Wall-clock ≈ weights ÷ min(source, node PCIe). Sharded assumes a well-parallelised loader
			(e.g. FSDP with parallel range reads). This is where Gen5 PCIe hosts (H100 and newer) pay off
			vs Gen4 (A100, L4, L40S): ~2× faster once the source can keep up.
		</p>
	{/if}
</section>
