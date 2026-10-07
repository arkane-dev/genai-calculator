import type { Config } from '$lib/profiler/types';

// Demand + SLOs for the inverse (sizing) problem: given a workload, find the
// smallest cluster that meets it.
export interface WorkloadSpec {
	modelId: string;
	gpuId: string;
	gpusPerNode: number;
	fabricId: string;
	weightFormatId: string;
	kvBits: 16 | 8;
	kvAllocation: 'paged' | 'contiguous';

	concurrency: number; // concurrent requests to serve
	basis: 'average' | 'p90' | 'peak'; // how that number was measured -> headroom
	inputTokens: number; // avg prompt length
	outputTokens: number; // avg generation length

	ttftTargetMs: number; // per-user SLO: time to first token
	throughputTargetTps: number; // per-user SLO: min generation tok/s

	// Speculative decoding (LLM/VLM only). A small draft model proposes `specTokens`
	// tokens per step, verified in one target pass; accepted at `draftAcceptRate`.
	// Lifts per-user decode throughput, so fewer replicas meet the same SLO.
	specDecode: boolean;
	draftAcceptRate: number;
	specTokens: number;
	// Prefix caching (LLM/VLM): fraction of the prompt served from cached KV, so
	// prefill only processes the uncached suffix — lowers TTFT.
	cachedPrefixFrac: number;

	// diffusion demand + SLOs (used when the model kind is 'diffusion')
	steps: number;
	resolution: number;
	guidance: boolean;
	targetImagesPerSec: number; // demand: images/sec the cluster must produce
	maxSecPerImage: number; // SLO: longest acceptable time for one image (batch)

	// JEPA demand + SLOs (used when the model kind is 'jepa')
	frames: number; // frames per clip
	targetClipsPerSec: number; // demand: clips/sec the cluster must encode
	maxSecPerClip: number; // SLO: longest acceptable time for one clip (batch)

	// Encoder demand + SLOs (used when the model kind is 'encoder')
	targetDocsPerSec: number; // demand: docs (or query-doc pairs) per second
	maxSecPerDoc: number; // SLO: longest acceptable per-batch latency

	// VLM inputs (used when the LLM has a `vlm` decoration)
	imagesPerRequest: number; // 0 = text-only; 1+ = each request carries N images

	// VLA demand + SLOs (used when the model kind is 'vla')
	targetRobots: number; // demand: robots driven at the model's control frequency
	maxSecPerControl: number; // SLO: longest acceptable per-chunk latency

	// ASR demand + SLOs (used when the model kind is 'asr')
	targetStreams: number; // demand: concurrent real-time audio streams
	minRtf: number; // SLO: minimum real-time factor (≥ 1 to keep up with live audio)
}

export interface SizeResult {
	gpuId: string;
	gpuName: string;
	feasible: boolean;
	reason?: string; // why not, when infeasible

	tp: number;
	pp: number;
	ep: number; // expert-parallel degree (MoE); 1 for dense
	dp: number;
	replicaGpus: number; // GPUs holding one model replica (pp × max(tp, ep))
	batchPerReplica: number; // concurrent seqs per replica meeting the throughput SLO
	provisionedConcurrency: number; // concurrency the cluster is sized to serve
	numGpus: number;
	numNodes: number;

	// achieved at the sized point (from computeProfile on `config`)
	perUserTps: number;
	ttftMs: number;
	fullReqMs: number;
	clusterTps: number;

	// diffusion outputs (present when the model kind is 'diffusion')
	imagesPerSec?: number; // aggregate across the cluster
	secPerImage?: number; // per-image latency at the sized batch

	// JEPA outputs (present when the model kind is 'jepa')
	clipsPerSec?: number; // aggregate across the cluster
	secPerClip?: number; // per-clip latency at the sized batch

	// Encoder outputs (present when the model kind is 'encoder')
	docsPerSec?: number; // aggregate across the cluster
	secPerDoc?: number; // per-doc latency at the sized batch

	// VLA outputs (present when the model kind is 'vla')
	robotsDriven?: number; // robots the cluster can drive at the target Hz
	secPerControl?: number; // per-chunk latency at the sized batch

	// ASR outputs (present when the model kind is 'asr')
	streamsServed?: number; // concurrent real-time streams the cluster can handle
	rtf?: number; // real-time factor at the sized batch (per replica)

	config: Config; // final config, ready for computeProfile → visuals
}

// One pool (prefill or decode) in a disaggregated deployment.
export interface PoolSizing {
	tp: number;
	pp: number;
	ep: number;
	replicas: number; // replicas of the model in this pool (dp for decode; count for prefill)
	replicaGpus: number; // GPUs per replica (pp × max(tp, ep))
	gpus: number;
	nodes: number;
	config: Config;
}

// Disaggregated prefill/decode sizing (transformer/LLM only): prefill and decode
// run on separate pools, each optimised for its own bottleneck, joined by a KV-
// cache transfer over the fabric.
export interface DisaggResult {
	feasible: boolean;
	reason?: string;
	provisionedConcurrency: number;
	promptsPerSec: number; // steady-state request rate the pools must sustain

	prefill: PoolSizing & { msPerPrompt: number; promptsPerSecPerReplica: number };
	decode: PoolSizing & { batchPerReplica: number; perUserTps: number; clusterTps: number };

	kvTransferMs: number; // per-request KV hand-off prefill → decode, over the fabric
	ttftMs: number; // prefill time + KV transfer
	totalGpus: number;
	totalNodes: number;
}
