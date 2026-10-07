// Domain types for the transformer inference profiler.
// All memory is tracked in bytes; bandwidth in bytes/s; compute in FLOP/s.

export interface MoeSpec {
	numExperts: number; // total experts per MoE layer
	expertsPerToken: number; // top-k routed experts per token
}

// Multi-head Latent Attention (DeepSeek, Kimi): KV is compressed to one small
// latent per token per layer instead of full per-head K and V, and it is
// replicated across tensor-parallel ranks rather than sharded.
export interface MlaSpec {
	kvLatentDim: number; // kv_lora_rank + qk_rope_head_dim, elements per token/layer
}

// Hybrid Mamba-2 / Transformer models (Nemotron-H, Nemotron Nano 2). The SSM
// layers keep a fixed-size recurrent state per sequence — it does NOT grow with
// context like a KV cache, but it is re-read and updated every decode step. The
// attention layers (few) still hold a normal KV cache (see kvLayers).
export interface MambaSpec {
	layers: number; // number of Mamba-2 (SSM) layers
	stateBytesPerLayer: number; // recurrent state bytes per sequence, per SSM layer
}

// Diffusion image/video models: inference is N denoising steps over a fixed
// latent (no KV cache, no autoregression). params = the denoiser (UNet/DiT) that
// runs every step; the encoders + VAE are extra memory. See computeProfile's
// diffusion branch.
export interface DiffusionSpec {
	arch: 'unet' | 'dit' | 'mmdit';
	vaeDownsample: number; // latent = image / this (typically 8)
	patch: number; // DiT patchify (2 for FLUX/SD3.5); for a conv UNet an EFFECTIVE
	// value tuned so 2·params·tokens matches the UNet's real per-forward FLOPs
	latentChannels: number;
	defaultSteps: number;
	defaultResolution: number; // px per side at which defaults are quoted
	cfg: boolean; // classifier-free guidance runs cond+uncond → per-step batch ×2
	frames?: number; // >1 for video; multiplies latent tokens
	aspect?: number; // latent width:height (default 1 = square); non-square video
	textEncoderParams?: number; // CLIP/T5 — extra weights to hold
	vaeParams?: number;
}

// V-JEPA / V-JEPA 2: self-supervised video models — a ViT encoder plus a lighter
// predictor. NOT generative. Inference is a SINGLE forward pass over the video
// patches to produce embeddings (understanding), or the predictor is rolled out
// for world-model planning. No KV cache, no autoregression. `params` is the
// encoder; the predictor is held resident. Video patch count is large, so the
// quadratic attention term matters — computeJepa counts it. See its branch.
export interface JepaSpec {
	patch: number; // spatial patch, px per side (16)
	tubelet: number; // temporal patch, frames grouped per token (2)
	defaultFrames: number; // frames per clip the card quotes
	defaultResolution: number; // crop, px per side
	predictorParams?: number; // predictor weights held resident (small)
	// Action-conditioned world model (V-JEPA 2-AC): the predictor is rolled out
	// over a planning horizon. When set, computeJepa adds rolloutSteps predictor
	// passes on top of the single encoder forward.
	predHidden?: number; // predictor width (for its attention term)
	predLayers?: number; // predictor depth
	rolloutSteps?: number; // planning-horizon predictor passes; unset = encoder only
}

// Encoder-only text models: embeddings (BGE, E5, Jina) and rerankers. Inference
// is a SINGLE forward pass over the input tokens producing a vector or score.
// No KV cache, no autoregression, no generated tokens. Attention is bidirectional
// full self-attention. Reuses the transformer roofline with kv=0, output=0.
export interface EncoderSpec {
	task: 'embedding' | 'reranker'; // how the head is used
	defaultSeqLen: number; // typical document length in tokens
	maxSeqLen: number; // model's positional-embedding cap
	embedDim?: number; // output vector dimension (embeddings only)
}

// Automatic Speech Recognition (ASR): Whisper-family encoder-decoder. Inference
// is one encoder pass over a fixed audio window (mel spectrogram → fixed number
// of audio tokens), then autoregressive text-token decode with cross-attention
// to the encoder output. The encoder is compute-bound (fixed cost per audio
// window); the decoder is memory-bound (auto-regressive).
export interface AsrSpec {
	encoderParams: number; // encoder weights
	decoderParams: number; // decoder weights
	audioTokens: number; // encoder sequence length after mel + conv stack (1500 for Whisper)
	audioWindowSec: number; // seconds of audio consumed per encoder pass (30 for Whisper)
	avgTextTokens: number; // typical decoder output length for a full audio window
	numLayersDecoder: number; // decoder depth (needed for KV cache size)
	numHeadsDecoder: number;
	headDimDecoder: number;
}

// Vision-Language Model (VLM) decoration on a transformer LLM. Images become
// extra input tokens (visual patches from a vision encoder) that the LLM reads
// during prefill. The vision encoder is small compared to the LLM but still
// held resident. Setting a VLM on a model doesn't change its kind — it's still
// a transformer that decodes text; images just add to inputTokens.
export interface VlmSpec {
	visionParams: number; // extra weights held resident (ViT / SigLIP encoder)
	tokensPerImage: number; // patches an image contributes at defaultImagePx
	defaultImagePx: number; // resolution the tokensPerImage figure is quoted at
}

// Vision-Language-Action (VLA) model: a VLM whose "output" is a chunk of
// continuous actions rather than tokens. Inference cycle = one observation
// (image(s) + state) → the model produces chunkSize actions via `flowSteps`
// flow-matching (or diffusion) passes through a small action expert. Reports
// controls/sec, latency per chunk, and effective closed-loop control rate.
export interface VlaSpec {
	chunkSize: number; // actions produced per inference call
	actionDim: number; // dimensionality of one action
	flowSteps: number; // flow-matching / diffusion sampling steps
	controlHz: number; // deployed control frequency (open-loop chunk replay)
	expertParams: number; // action-expert params, applied per flow step
	camerasPerObs: number; // number of camera views concatenated per observation
	tokensPerImage: number; // vision tokens per camera at defaultImagePx
	defaultImagePx: number;
	visionParams: number; // frozen vision-encoder weights (typically SigLIP)
	stateDim: number; // proprioceptive state (joint positions, etc.)
}

export interface ModelSpec {
	id: string;
	name: string;
	kind?: 'transformer' | 'diffusion' | 'jepa' | 'encoder' | 'vla' | 'asr'; // default 'transformer'
	diffusion?: DiffusionSpec; // present when kind === 'diffusion'
	jepa?: JepaSpec; // present when kind === 'jepa'
	encoder?: EncoderSpec; // present when kind === 'encoder'
	vlm?: VlmSpec; // decoration on a transformer: image inputs (VLM)
	vla?: VlaSpec; // present when kind === 'vla' (robotics action model)
	asr?: AsrSpec; // present when kind === 'asr' (Whisper-family)
	params: number; // total parameter count (denoiser params for diffusion)
	hiddenSize: number; // d_model
	numLayers: number;
	numHeads: number; // query heads
	numKvHeads: number; // key/value heads (== numHeads for MHA, fewer for GQA)
	headDim: number;
	intermediateSize: number; // FFN inner dim
	vocabSize: number;
	moe?: MoeSpec; // present only for mixture-of-experts models
	mla?: MlaSpec; // present for models using latent attention
	mamba?: MambaSpec; // present for hybrid Mamba-2 / Transformer models
	activeParams?: number; // card-stated active params/token; when set, drives the expert split
	kvLayers?: number; // layers that hold a growing KV cache (hybrid models); default numLayers
	visualAR?: boolean; // transformer that generates images by autoregression (VAR); groups separately
	reasoning?: boolean; // reasoning model (long chain-of-thought); groups separately, expect big output-token budgets
	tts?: boolean; // text-to-speech: autoregressive decode of audio-codec tokens (transformer path); groups separately
	projected?: boolean; // specs are an estimate of an unreleased model
}

export interface GpuSpec {
	id: string;
	name: string;
	memoryGiB: number; // HBM capacity
	memBandwidthTBs: number; // HBM bandwidth, TB/s (1e12 B/s)
	fp16Tflops: number; // dense FP16/BF16 tensor throughput
	fp8Tflops: number; // dense FP8 (or INT8 proxy) tensor throughput
	fp4Tflops: number; // dense FP4 tensor throughput; 0 => no native FP4 path
	nvlinkGBs: number; // per-GPU aggregate interconnect bandwidth, GB/s
	hasNvlink: boolean; // false => interconnect is PCIe, not NVLink
	tdpWatts: number; // published board TDP (max sustained power draw), watts
	pcieGBs: number; // PCIe host↔GPU bandwidth (Gen4 x16 ≈ 32; Gen5 x16 ≈ 63)
	// Per-GPU multiplier on the compute-bound MFU. The global MFU (0.74) was
	// calibrated on H100; other parts sustain a different fraction of tensor peak
	// (e.g. L40S compute-bound paths run ~0.75× the H100 figure). Default 1.
	// Memory-bound paths (decode) are unaffected. Only applies to compute time.
	mfuFactor?: number;
	die: DieUnit[]; // approximate die-area breakdown
}

// One block of the GPU die. areaFrac is that block's approximate share of the
// die area; the blocks for a GPU sum to ~1. kind decides how it is driven:
// tensor/cuda/rt fill to a utilization, cache is structural (never "used").
// kind drives how the block's usage is estimated:
//   tensor -> tensor-core utilization, cuda -> vector utilization, rt -> unused,
//   hbm    -> HBM-bandwidth fill (memory controllers; low in prefill),
//   l2     -> L2 activity (busy streaming in decode OR feeding compute in prefill),
//   link   -> interconnect fill (NVLink / PCIe on-die I/O),
//   sched  -> scheduler / uncore, genuinely structural (no estimate).
export interface DieUnit {
	kind: 'tensor' | 'cuda' | 'rt' | 'hbm' | 'l2' | 'link' | 'sched';
	label: string;
	areaFrac: number;
}

// Inter-node network fabric. allReduceEff / a2aEff are the fraction of peak an
// all-reduce vs an all-to-all sustains. Multipath RDMA (packet spraying) keeps
// all-to-all close to peak; InfiniBand's flow-based routing collides on all-to-all.
export interface FabricSpec {
	id: string;
	label: string;
	kind: 'eth' | 'ib' | 'multipath';
	gbps: number; // per-GPU bandwidth, Gb/s
	allReduceEff: number;
	a2aEff: number;
	// α (latency) in microseconds — the fixed per-message cost, independent of
	// message size. Real collective time ≈ α · log2(N) + bytes / (bw · efficiency).
	// NVLink and InfiniBand are low-latency; multipath RDMA sits close behind;
	// classic Ethernet is high. This term dominates for small-message decode
	// collectives across many nodes.
	alphaUs: number;
}

// A weight quantization format. bitsPerWeight includes micro-scaling overhead
// (e.g. NVFP4 is 4 + one FP8 scale per 16 = 4.5). tier picks the tensor-core
// datapath; 'fp4' falls back to the fp16 rate on GPUs with no native FP4.
export interface WeightFormat {
	id: string;
	label: string;
	bitsPerWeight: number;
	tier: 'fp16' | 'fp8' | 'fp4';
}

export interface Config {
	modelId: string;
	gpuId: string;
	numGpus: number;
	gpusPerNode: number; // NVLink domain size (<= 8 for standard servers)
	fabricId: string; // inter-node network fabric
	tp: number; // tensor-parallel degree
	pp: number; // pipeline-parallel degree
	ppEnabled: boolean; // pipeline parallelism switch
	ep: number; // expert-parallel degree (MoE only)
	epEnabled: boolean; // expert parallelism switch (MoE only)
	batchSize: number; // concurrent sequences
	inputTokens: number; // prompt tokens (drives prefill FLOPs and TTFT)
	outputTokens: number; // generated tokens (adds to KV, drives decode duration)
	weightFormatId: string; // weight quantization format
	kvBits: 16 | 8;
	phase: 'decode' | 'prefill'; // which forward pass to profile
	kvAllocation: 'paged' | 'contiguous'; // KV-cache allocation strategy
	// speculative decoding (decode only): a draft proposes tokens the target verifies
	specDecode: boolean;
	draftAcceptRate: number; // α, per-token acceptance probability (0..1)
	specTokens: number; // γ, draft tokens proposed per cycle
	// Prefix caching (prefill only): fraction of the prompt that is an already-cached
	// shared prefix (system prompt, RAG boilerplate, chat history). Its KV is reused,
	// so prefill only processes the uncached suffix — lower TTFT + prefill FLOPs. KV
	// memory is unchanged (the prefix KV is resident). 0 = no caching.
	cachedPrefixFrac: number;
	// VLM inputs (a transformer with model.vlm reads these as extra input tokens)
	imagesPerRequest: number; // 0 = text-only; 1+ = a VLM request with N images
	// diffusion runtime knobs (ignored for transformer models)
	steps: number; // denoising steps
	resolution: number; // px per side
	guidance: boolean; // classifier-free guidance on/off
	frames: number; // frames per clip (video diffusion / JEPA); 1 otherwise
}

export interface MemorySegment {
	key:
		| 'weights'
		| 'kv'
		| 'mamba'
		| 'activations'
		| 'gradients'
		| 'optimizer'
		| 'overhead'
		| 'free'
		| 'overflow';
	label: string;
	bytes: number;
	color: string;
}

export interface Profile {
	// per-GPU memory
	perGpu: {
		weights: number;
		kv: number;
		activations: number;
		overhead: number;
		used: number;
		capacity: number;
		fits: boolean;
		headroom: number; // capacity - used (negative when overflowing)
	};
	segments: MemorySegment[];

	// KV-cache allocation
	kvWasteFactor: number; // resident KV / ideal KV (paging rounding or over-reservation)
	kvPerSeqBytes: number; // per-GPU KV for one sequence at this context length
	maxConcurrentSeqs: number; // sequences of this context length the KV budget holds

	// performance
	phase: 'decode' | 'prefill';
	stepTimeMs: number;
	throughputTps: number; // aggregate tokens/s across the deployment
	perUserTps: number; // tokens/s for a single sequence
	ttftMs: number; // time to first token (prefill of full context)
	specSpeedup?: number; // decode throughput multiplier from speculative decoding (1 = off)
	bottleneck: 'memory' | 'compute' | 'network';

	// diffusion outputs (present only when the model kind is 'diffusion')
	diffusion?: {
		imagesPerSec: number; // aggregate across the cluster
		secPerImage: number; // wall latency to produce one image (a batch)
		stepTimeMs: number; // one denoising step
		steps: number;
		latentTokens: number; // per image (compute tokens for a UNet)
		cfg: boolean;
	};

	// JEPA outputs (present only when the model kind is 'jepa')
	jepa?: {
		clipsPerSec: number; // aggregate embeddings/sec across the cluster
		secPerClip: number; // wall latency for one clip (a batch)
		forwardTimeMs: number; // single encoder forward pass
		tokens: number; // video patches per clip
		frames: number;
	};

	// Encoder outputs (present when the model kind is 'encoder')
	encoder?: {
		docsPerSec: number; // aggregate embeddings/scores per second, across the cluster
		secPerDoc: number; // wall latency for one document (or a batch)
		forwardTimeMs: number; // single encoder forward pass
		tokens: number; // sequence length used
	};

	// VLA outputs (present when the model kind is 'vla')
	vla?: {
		controlsPerSec: number; // aggregate cluster-wide inference calls per second (chunks)
		secPerControl: number; // per-chunk latency (one observation → chunkSize actions)
		effectiveHz: number; // closed-loop control rate = chunksPerSec × chunkSize (per replica)
		actionsPerSec: number; // total actions produced per second across the cluster
	};

	// ASR outputs (present when the model kind is 'asr')
	asr?: {
		secPerWindow: number; // wall-clock time to transcribe one audio window
		audioSecPerSec: number; // seconds of audio processed per real second across the cluster (RTF × dp)
		rtf: number; // real-time factor per replica: audioWindowSec / secPerWindow
		windowsPerSec: number; // audio windows processed per second across the cluster
		encoderMs: number; // encoder pass alone (one 30s window)
		decoderMs: number; // full autoregressive decode of avgTextTokens
	};

	// utilization fractions (0..1) that drive the fills
	memBwFrac: number;
	computeFrac: number;
	l2Frac: number; // L2-cache activity (max of memory / compute pressure)
	nvlinkFrac: number; // intra-node NVLink
	netFrac: number; // inter-node fabric (InfiniBand / RDMA Ethernet)

	// absolute rates, for labels
	bandwidthUsedGBs: number;
	bandwidthPeakGBs: number;
	computeAchievedTflops: number;
	computePeakTflops: number;
	nvlinkUsedGBs: number;
	nvlinkPeakGBs: number;
	netUsedGBs: number;
	netPeakGBs: number;

	// per-collective detail, for the topology legend
	comm: {
		tpActive: boolean;
		tpOverFabric: boolean;
		tpGBs: number;
		epActive: boolean;
		epOverFabric: boolean;
		epGBs: number;
		ppActive: boolean;
		ppOverFabric: boolean;
		ppGBs: number;
	};

	fabric: {
		label: string;
		kind: FabricSpec['kind'];
		peakGBs: number;
		allReduceEff: number;
		a2aEff: number;
	};

	cluster: {
		numGpus: number;
		tp: number;
		pp: number;
		ep: number;
		dp: number;
		gpusPerNode: number;
		numNodes: number;
		crossesFabric: boolean; // any collective spans a node boundary
		scaleOut: 'single' | 'data-parallel' | 'model-across-nodes';
	};
	activeParams: number;
}
