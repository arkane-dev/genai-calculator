// Cold-start / model-load time.
//
// Streaming weights from storage → host RAM → HBM. Two things bound the pipe:
//   1. the storage source (object storage, network block storage, local NVMe,
//      a parallel file system)
//   2. the PCIe host↔GPU link
// A saturated pipeline runs at min(source, pcie); real deployments overlap
// them so wall-clock ≈ weights ÷ min(source, pcie). Sharded loads (FSDP,
// tensor-parallel) fan out across all GPUs' PCIe lanes so each holds a slice.
//
// Numbers are steady-state throughputs after warmup: cold object storage in
// particular ramps for several seconds. The fastest options (parallel file
// system, parallel object reads) assume a properly parallelised loader that
// fetches many ranges at once.

export interface WeightSource {
	id: string;
	label: string;
	gbPerSec: number; // effective steady-state read throughput, GB/s (aggregate per-node)
	note?: string;
}

export const SOURCES: WeightSource[] = [
	{ id: 'object-single', label: 'Object storage (single stream)', gbPerSec: 0.1, note: 'one GET at a time' },
	{
		id: 'object-parallel',
		label: 'Object storage (parallel loader)',
		gbPerSec: 8,
		note: 'many parallel ranges'
	},
	{ id: 'block-standard', label: 'Network block storage (~1 GB/s)', gbPerSec: 1 },
	{ id: 'block-fast', label: 'Network block storage (high-performance)', gbPerSec: 4 },
	{
		id: 'local-nvme',
		label: 'Local NVMe (GPU server)',
		gbPerSec: 30,
		note: 'aggregate across RAIDed drives'
	},
	{
		id: 'parallel-fs',
		label: 'Parallel file system (Lustre-class)',
		gbPerSec: 100,
		note: 'scales with provisioned capacity'
	},
	{ id: 'host-ram', label: 'Cached in host RAM', gbPerSec: 400, note: 'DDR5 bandwidth; PCIe-bound' }
];

export const SOURCES_BY_ID = new Map(SOURCES.map((s) => [s.id, s]));

export interface LoadInputs {
	weightBytes: number; // total weight bytes for the model (before sharding)
	numGpus: number; // GPUs the weights are sharded across
	pcieGBs: number; // per-GPU PCIe host↔GPU bandwidth
	sourceId: string; // one of SOURCES
	sharded: boolean; // FSDP/tensor-parallel: each GPU loads weights/numGpus
	shardWaste: number; // ~1.0 for pure DP (every GPU loads all weights); overrides sharded
}

export interface LoadResult {
	perGpuBytes: number; // bytes each GPU pulls from host into HBM
	sourceGBs: number;
	pcieGBs: number;
	nodePcieGBs: number; // effective aggregate PCIe if `numGpus` fit in one node
	bottleneck: 'source' | 'pcie';
	seconds: number;
	sourceLabel: string;
}

export function estimateLoad(i: LoadInputs): LoadResult {
	const src = SOURCES_BY_ID.get(i.sourceId) ?? SOURCES[0];
	// Sharded loads: each GPU only pulls its share; DP replicas each pull all.
	const perGpuBytes = i.sharded ? i.weightBytes / Math.max(1, i.numGpus) : i.weightBytes;

	// Aggregate PCIe = min(GPUs, 8) × per-GPU PCIe (one node's PCIe lanes).
	const nodePcieGBs = Math.min(8, i.numGpus) * i.pcieGBs;
	// Effective source bandwidth per node — most sources are per-node not per-GPU.
	const sourceGBsEff = src.gbPerSec;

	// Wall-clock: total bytes pulled into the node ÷ min(source, node-PCIe).
	// Bytes per node = perGpuBytes × min(numGpus, 8) when sharded (each GPU distinct),
	// or perGpuBytes × min(numGpus, 8) when DP (each GPU loads all).
	const bytesPerNode = perGpuBytes * Math.min(8, i.numGpus);
	const pcieSec = bytesPerNode / (nodePcieGBs * 1e9);
	const sourceSec = bytesPerNode / (sourceGBsEff * 1e9);
	const seconds = Math.max(pcieSec, sourceSec);

	return {
		perGpuBytes,
		sourceGBs: sourceGBsEff,
		pcieGBs: i.pcieGBs,
		nodePcieGBs,
		bottleneck: sourceSec >= pcieSec ? 'source' : 'pcie',
		seconds,
		sourceLabel: src.label
	};
}
