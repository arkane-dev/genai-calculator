import type { DieUnit, FabricSpec, GpuSpec, ModelSpec, WeightFormat } from './types';

// Segment colors, from the NEONDECK chart palette. Legible on the dark canvas and
// distinct from each other for the common forms of color-vision deficiency.
export const COLORS = {
	weights: '#ff2bd6', // magenta
	kv: '#22f2f7', // cyan
	mamba: '#a66bff', // violet, SSM recurrent state (hybrid Mamba models)
	activations: '#f5ec58', // yellow
	gradients: '#ef6352', // coral, training gradients
	optimizer: '#3ff0b8', // jade, optimizer states (Adam m/v + fp32 master)
	overhead: '#8885b9', // muted, framework overhead
	free: '#12163a', // empty capacity (surface-2)
	overflow: '#ff3b52', // red, does-not-fit
	memPipe: '#22f2f7', // cyan, HBM->compute bandwidth (ties to KV)
	tensor: '#f6bd6a', // gold, tensor cores
	cuda: '#6077ff', // blue, vector/CUDA cores
	rt: '#ef6352', // coral, ray-tracing cores
	hbm: '#22f2f7', // cyan, memory controllers / HBM I/O (ties to the HBM pipe)
	l2: '#3ff0b8', // jade, L2 cache
	link: '#a66bff', // violet, on-die NVLink/PCIe I/O (ties to NVLink pipes)
	sched: '#8885b9', // muted, scheduler / uncore (structural)
	nvlink: '#a66bff', // violet, intra-node interconnect
	network: '#f5ec58' // yellow, inter-node fabric (InfiniBand / RDMA Ethernet)
} as const;

// Approximate die-area breakdowns. Real floorplans aren't published cleanly, so
// these are estimates: Hopper/Ampere data-center parts have no RT cores and
// spend large area on L2 + memory controllers + NVLink; Ada parts (L40S, RTX)
// carry RT cores for graphics that LLM inference never touches.
const DIE_HOPPER: DieUnit[] = [
	{ kind: 'tensor', label: 'Tensor cores', areaFrac: 0.3 },
	{ kind: 'cuda', label: 'CUDA / vector cores', areaFrac: 0.25 },
	{ kind: 'l2', label: 'L2 cache', areaFrac: 0.16 },
	{ kind: 'hbm', label: 'HBM I/O (mem controllers)', areaFrac: 0.08 },
	{ kind: 'link', label: 'NVLink I/O', areaFrac: 0.1 },
	{ kind: 'sched', label: 'Scheduler · uncore', areaFrac: 0.11 }
];
const DIE_AMPERE: DieUnit[] = [
	{ kind: 'tensor', label: 'Tensor cores', areaFrac: 0.22 },
	{ kind: 'cuda', label: 'CUDA / vector cores', areaFrac: 0.3 },
	{ kind: 'l2', label: 'L2 cache', areaFrac: 0.18 },
	{ kind: 'hbm', label: 'HBM I/O (mem controllers)', areaFrac: 0.1 },
	{ kind: 'link', label: 'NVLink I/O', areaFrac: 0.08 },
	{ kind: 'sched', label: 'Scheduler · uncore', areaFrac: 0.12 }
];
const DIE_ADA_DC: DieUnit[] = [
	{ kind: 'tensor', label: 'Tensor cores', areaFrac: 0.2 },
	{ kind: 'cuda', label: 'CUDA / vector cores', areaFrac: 0.3 },
	{ kind: 'rt', label: 'RT cores', areaFrac: 0.12 },
	{ kind: 'l2', label: 'L2 cache', areaFrac: 0.14 },
	{ kind: 'hbm', label: 'GDDR I/O (mem controllers)', areaFrac: 0.08 },
	{ kind: 'link', label: 'PCIe host I/O', areaFrac: 0.06 },
	{ kind: 'sched', label: 'Scheduler · uncore', areaFrac: 0.1 }
];
const DIE_ADA_CONSUMER: DieUnit[] = [
	{ kind: 'cuda', label: 'CUDA / vector cores', areaFrac: 0.38 },
	{ kind: 'tensor', label: 'Tensor cores', areaFrac: 0.18 },
	{ kind: 'rt', label: 'RT cores', areaFrac: 0.14 },
	{ kind: 'l2', label: 'L2 cache', areaFrac: 0.12 },
	{ kind: 'hbm', label: 'GDDR I/O (mem controllers)', areaFrac: 0.06 },
	{ kind: 'link', label: 'PCIe host I/O', areaFrac: 0.05 },
	{ kind: 'sched', label: 'Scheduler · uncore', areaFrac: 0.07 }
];
// Blackwell data-center (B200/B300): even more tensor area, big cache, NVLink 5.
const DIE_BLACKWELL_DC: DieUnit[] = [
	{ kind: 'tensor', label: 'Tensor cores', areaFrac: 0.34 },
	{ kind: 'cuda', label: 'CUDA / vector cores', areaFrac: 0.22 },
	{ kind: 'l2', label: 'L2 cache', areaFrac: 0.15 },
	{ kind: 'hbm', label: 'HBM I/O (mem controllers)', areaFrac: 0.08 },
	{ kind: 'link', label: 'NVLink 5 I/O', areaFrac: 0.11 },
	{ kind: 'sched', label: 'Scheduler · uncore', areaFrac: 0.1 }
];
// Blackwell workstation (RTX PRO): carries RT cores for graphics.
const DIE_BLACKWELL_WS: DieUnit[] = [
	{ kind: 'cuda', label: 'CUDA / vector cores', areaFrac: 0.34 },
	{ kind: 'tensor', label: 'Tensor cores', areaFrac: 0.22 },
	{ kind: 'rt', label: 'RT cores', areaFrac: 0.13 },
	{ kind: 'l2', label: 'L2 cache', areaFrac: 0.12 },
	{ kind: 'hbm', label: 'GDDR7 I/O (mem controllers)', areaFrac: 0.06 },
	{ kind: 'link', label: 'PCIe host I/O', areaFrac: 0.05 },
	{ kind: 'sched', label: 'Scheduler · uncore', areaFrac: 0.08 }
];

// Turing (T4): tensor + RT cores, GDDR6, PCIe only.
const DIE_TURING: DieUnit[] = [
	{ kind: 'cuda', label: 'CUDA / vector cores', areaFrac: 0.34 },
	{ kind: 'tensor', label: 'Tensor cores', areaFrac: 0.16 },
	{ kind: 'rt', label: 'RT cores', areaFrac: 0.1 },
	{ kind: 'l2', label: 'L2 cache', areaFrac: 0.1 },
	{ kind: 'hbm', label: 'GDDR6 I/O (mem controllers)', areaFrac: 0.08 },
	{ kind: 'link', label: 'PCIe host I/O', areaFrac: 0.06 },
	{ kind: 'sched', label: 'Scheduler · uncore', areaFrac: 0.16 }
];
// Ampere graphics/inference (A10G): tensor + RT cores, GDDR6, PCIe.
const DIE_AMPERE_G: DieUnit[] = [
	{ kind: 'cuda', label: 'CUDA / vector cores', areaFrac: 0.34 },
	{ kind: 'tensor', label: 'Tensor cores', areaFrac: 0.18 },
	{ kind: 'rt', label: 'RT cores', areaFrac: 0.12 },
	{ kind: 'l2', label: 'L2 cache', areaFrac: 0.1 },
	{ kind: 'hbm', label: 'GDDR6 I/O (mem controllers)', areaFrac: 0.08 },
	{ kind: 'link', label: 'PCIe host I/O', areaFrac: 0.06 },
	{ kind: 'sched', label: 'Scheduler · uncore', areaFrac: 0.12 }
];

// Inter-node fabrics, per-GPU Gb/s plus collective efficiencies.
//
// all-reduce is a structured ring/tree; even flow-routed InfiniBand runs it near
// peak. all-to-all (MoE dispatch/combine) is the hard case: InfiniBand and plain RoCE
// pin each flow to one ECMP path, so flows collide and effective bandwidth drops.
// Multipath RDMA fabrics spray packets across many paths and reassemble out of order
// (adaptive routing: e.g. NVIDIA Spectrum-X, the Ultra Ethernet transport, and some
// cloud providers' own transports), which spreads an all-to-all evenly and sustains
// most of peak. Bandwidth is per GPU: a common 8-GPU server with eight 400G NICs is
// 400 Gb/s per GPU.
export const FABRICS: FabricSpec[] = [
	// Latency α (μs) is per-message; ~10 for datacentre Ethernet, ~2 for IB, ~3-5 for
	// multipath RDMA. Real collective time = α·log2(N) + β·bytes.
	{
		// A cloud VM's standard network (no RDMA). Typically ~100 Gb/s for a whole 8-GPU
		// server, so ≈12.5 Gb/s per GPU. TCP transport → high latency, and collectives
		// sit well below line rate. Smaller VMs get less. Real multi-node GPU work uses
		// an RDMA fabric instead.
		id: 'cloud-eth',
		label: 'Cloud VPC Ethernet (no RDMA)',
		kind: 'eth',
		gbps: 12.5,
		allReduceEff: 0.5,
		a2aEff: 0.35,
		alphaUs: 30
	},
	{
		// 100G Ethernet with one NIC per GPU (HPC / on-prem style).
		id: 'eth100',
		label: '100G Ethernet (per GPU)',
		kind: 'eth',
		gbps: 100,
		allReduceEff: 0.6,
		a2aEff: 0.4,
		alphaUs: 10
	},
	{
		id: 'ib-hdr',
		label: 'IB HDR 200G',
		kind: 'ib',
		gbps: 200,
		allReduceEff: 0.82,
		a2aEff: 0.55,
		alphaUs: 2
	},
	{
		id: 'ib-ndr',
		label: 'IB NDR 400G',
		kind: 'ib',
		gbps: 400,
		allReduceEff: 0.85,
		a2aEff: 0.58,
		alphaUs: 2
	},
	{
		id: 'ib-xdr',
		label: 'IB XDR 800G',
		kind: 'ib',
		gbps: 800,
		allReduceEff: 0.85,
		a2aEff: 0.6,
		alphaUs: 1.5
	},
	{
		id: 'mp-50',
		label: 'Multipath RDMA 50G/GPU',
		kind: 'multipath',
		gbps: 50,
		allReduceEff: 0.8,
		a2aEff: 0.88,
		alphaUs: 15
	},
	{
		id: 'mp-400',
		label: 'Multipath RDMA 400G/GPU',
		kind: 'multipath',
		gbps: 400,
		allReduceEff: 0.82,
		a2aEff: 0.9,
		alphaUs: 5
	},
	{
		id: 'mp-800',
		label: 'Multipath RDMA 800G/GPU',
		kind: 'multipath',
		gbps: 800,
		allReduceEff: 0.82,
		a2aEff: 0.9,
		alphaUs: 5
	}
];

export const FABRICS_BY_ID = new Map(FABRICS.map((f) => [f.id, f]));

// The usual inter-node fabric for each GPU: HGX-class parts sit on 400G (B300: 800G)
// RDMA per GPU; A100 servers typically on InfiniBand HDR. PCIe cards usually run
// on standard datacentre Ethernet. Used to default the fabric when a GPU is picked
// (user-overridable).
const DEFAULT_FABRIC: Record<string, string> = {
	'h100-sxm': 'mp-400',
	h200: 'mp-400',
	b200: 'mp-400',
	b300: 'mp-800',
	'a100-80': 'ib-hdr',
	'a100-40': 'ib-hdr'
};

export function defaultFabricFor(gpuId: string): string {
	return DEFAULT_FABRIC[gpuId] ?? 'cloud-eth';
}

// NVLink stays inside a node and runs both collectives near peak.
export const NVLINK_ALLREDUCE_EFF = 0.85;
export const NVLINK_A2A_EFF = 0.85;
export const PP_P2P_EFF = 0.9; // pipeline hand-off is point-to-point, efficient

export const GPUS_PER_NODE = [1, 2, 4, 8] as const;

// Weight quantization formats. bitsPerWeight includes micro-scaling overhead.
// tier selects the tensor-core datapath: fp16 (also bf16, and any format
// dequantized to fp16), fp8 (fp8 e4m3/e5m2 and fp6 share the FP8 datapath on
// Blackwell), fp4 (fp4/nvfp4/mxfp4 — 2x fp8 where a native path exists, else
// falls back to fp16 dequant). NVFP4 = 4 + FP8 scale/16 = 4.5 b; MXFP4 = 4 +
// E8M0/32 = 4.25 b. FP6 formats are 6 bits and run at FP8 rate.
export const WEIGHT_FORMATS: WeightFormat[] = [
	{ id: 'bf16', label: 'BF16 / FP16', bitsPerWeight: 16, tier: 'fp16' },
	{ id: 'fp8', label: 'FP8 (e4m3/e5m2)', bitsPerWeight: 8, tier: 'fp8' },
	{ id: 'mxfp6', label: 'MXFP6 (e3m2/e2m3)', bitsPerWeight: 6, tier: 'fp8' },
	{ id: 'nvfp4', label: 'NVFP4', bitsPerWeight: 4.5, tier: 'fp4' },
	{ id: 'mxfp4', label: 'MXFP4', bitsPerWeight: 4.25, tier: 'fp4' },
	{ id: 'int4', label: 'INT4 (dequant)', bitsPerWeight: 4, tier: 'fp16' }
];

export const WEIGHT_FORMATS_BY_ID = new Map(WEIGHT_FORMATS.map((f) => [f.id, f]));

// GPUs with native FP8 (E4M3/E5M2) tensor cores: Hopper, Ada, Blackwell (incl. the
// Ada/Blackwell workstation parts). Ampere/Turing lack FP8. `fp8Tflops` on GpuSpec
// doubles as an INT8 proxy, so it can't distinguish these — hence the explicit set.
export const FP8_NATIVE_GPUS = new Set([
	'h100-sxm',
	'h200',
	'l40s',
	'l4',
	'b200',
	'b300',
	'rtx4090',
	'rtx-pro-6000',
	'rtx-pro-4500'
]);

// Whether a GPU can run a weight format at native speed. fp16-tier (BF16/FP16, and
// INT4 which dequantizes to fp16 compute) is universal. FP4 and the OCP microscaling
// formats (NVFP4/MXFP4/MXFP6) are Blackwell-only (fp4Tflops > 0). Plain FP8 needs the
// broader FP8-native set. Used to disable unsupported formats in the pickers.
export function gpuSupportsFormat(gpu: GpuSpec, fmt: WeightFormat): boolean {
	if (fmt.tier === 'fp16') return true;
	if (fmt.tier === 'fp4' || fmt.id === 'mxfp6') return gpu.fp4Tflops > 0;
	return FP8_NATIVE_GPUS.has(gpu.id);
}

// Segmented-control options for the compact weight-format pickers (Workload / Training /
// Economics), with `disabled` set for formats the GPU can't run natively. `ids` picks the
// subset a tab offers (training omits fp4). Short labels to fit the segmented buttons.
const WF_SHORT: Record<string, string> = { bf16: 'BF16', fp8: 'FP8', nvfp4: 'NVFP4' };
export function weightFormatOptions(
	gpu: GpuSpec,
	ids: string[] = ['bf16', 'fp8', 'nvfp4']
): { value: string; label: string; disabled: boolean }[] {
	return ids.map((id) => {
		const f = WEIGHT_FORMATS_BY_ID.get(id)!;
		return { value: id, label: WF_SHORT[id] ?? f.label, disabled: !gpuSupportsFormat(gpu, f) };
	});
}

// Model presets. Params/config approximate the public architectures.
export const MODELS: ModelSpec[] = [
	{
		id: 'llama32-3b',
		name: 'Llama 3.2 3B',
		params: 3.21e9,
		hiddenSize: 3072,
		numLayers: 28,
		numHeads: 24,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 8192,
		vocabSize: 128256
	},
	{
		id: 'llama31-8b',
		name: 'Llama 3.1 8B',
		params: 8.03e9,
		hiddenSize: 4096,
		numLayers: 32,
		numHeads: 32,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 14336,
		vocabSize: 128256
	},
	{
		id: 'llama31-70b',
		name: 'Llama 3.1 70B',
		params: 70.6e9,
		hiddenSize: 8192,
		numLayers: 80,
		numHeads: 64,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 28672,
		vocabSize: 128256
	},
	{
		id: 'llama31-405b',
		name: 'Llama 3.1 405B',
		params: 406e9,
		hiddenSize: 16384,
		numLayers: 126,
		numHeads: 128,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 53248,
		vocabSize: 128256
	},
	{
		id: 'qwen25-72b',
		name: 'Qwen2.5 72B',
		params: 72.7e9,
		hiddenSize: 8192,
		numLayers: 80,
		numHeads: 64,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 29568,
		vocabSize: 152064
	},
	{
		id: 'qwen25-14b',
		name: 'Qwen2.5 14B',
		params: 14.77e9,
		hiddenSize: 5120,
		numLayers: 48,
		numHeads: 40,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 13824,
		vocabSize: 152064
	},
	// --- Reasoning models (long chain-of-thought; plan for large output-token budgets) ---
	{
		id: 'r1-distill-qwen-1_5b',
		name: 'DeepSeek-R1-Distill-Qwen-1.5B',
		reasoning: true,
		params: 1.78e9,
		hiddenSize: 1536,
		numLayers: 28,
		numHeads: 12,
		numKvHeads: 2,
		headDim: 128,
		intermediateSize: 8960,
		vocabSize: 151936
	},
	{
		id: 'r1-distill-qwen-7b',
		name: 'DeepSeek-R1-Distill-Qwen-7B',
		reasoning: true,
		params: 7.62e9,
		hiddenSize: 3584,
		numLayers: 28,
		numHeads: 28,
		numKvHeads: 4,
		headDim: 128,
		intermediateSize: 18944,
		vocabSize: 152064
	},
	{
		id: 'r1-distill-qwen-32b',
		name: 'DeepSeek-R1-Distill-Qwen-32B',
		reasoning: true,
		params: 32.8e9,
		hiddenSize: 5120,
		numLayers: 64,
		numHeads: 40,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 27648,
		vocabSize: 152064
	},
	{
		id: 'qwq-32b',
		name: 'QwQ-32B',
		reasoning: true,
		params: 32.8e9,
		hiddenSize: 5120,
		numLayers: 64,
		numHeads: 40,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 27648,
		vocabSize: 152064
	},
	{
		// openai/gpt-oss-120b: MoE, 128 experts, top-4, ~5.1B active of 116.8B total.
		// Ships MXFP4 for the expert weights (pick NVFP4/MXFP4 to model that).
		id: 'gpt-oss-120b',
		name: 'GPT-OSS 120B',
		reasoning: true,
		params: 116.8e9,
		activeParams: 5.1e9,
		hiddenSize: 2880,
		numLayers: 36,
		numHeads: 64,
		numKvHeads: 8,
		headDim: 64,
		intermediateSize: 2880,
		vocabSize: 201088,
		moe: { numExperts: 128, expertsPerToken: 4 }
	},
	{
		id: 'mistral-7b',
		name: 'Mistral 7B',
		params: 7.24e9,
		hiddenSize: 4096,
		numLayers: 32,
		numHeads: 32,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 14336,
		vocabSize: 32000
	},
	{
		id: 'mixtral-8x7b',
		name: 'Mixtral 8x7B (MoE)',
		params: 46.7e9,
		hiddenSize: 4096,
		numLayers: 32,
		numHeads: 32,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 14336,
		vocabSize: 32000,
		moe: { numExperts: 8, expertsPerToken: 2 }
	},
	// --- Verified from HuggingFace config.json (Sept 2026). See notes per model. ---
	{
		// deepseek-ai/DeepSeek-V4-Pro. V4 replaced V3's MLA with a new compressed
		// attention (lightning indexer + per-layer compression); config has 1 KV
		// head at head_dim 512, which we model as the effective KV footprint.
		id: 'deepseek-v4',
		name: 'DeepSeek V4 (Pro)',
		params: 1.6e12,
		hiddenSize: 7168,
		numLayers: 61,
		numHeads: 128,
		numKvHeads: 1,
		headDim: 512,
		intermediateSize: 3072, // moe_intermediate (per expert)
		vocabSize: 129280,
		moe: { numExperts: 384, expertsPerToken: 6 },
		activeParams: 49e9 // card-stated
	},
	{
		// deepseek-ai/DeepSeek-V4-Flash
		id: 'deepseek-v4-flash',
		name: 'DeepSeek V4 Flash',
		params: 284e9,
		hiddenSize: 4096,
		numLayers: 43,
		numHeads: 64,
		numKvHeads: 1,
		headDim: 512,
		intermediateSize: 2048,
		vocabSize: 129280,
		moe: { numExperts: 256, expertsPerToken: 6 },
		activeParams: 13e9
	},
	{
		// deepseek-ai/DeepSeek-V4.1-Flash — config.json. MoE, MLA (1 KV head @ head_dim
		// 512). Multimodal: has a vision tower (vision_config), unlike V4-Flash which is
		// text-only. Total/active params + visionParams estimated from the architecture.
		id: 'deepseek-v4-1-flash',
		name: 'DeepSeek V4.1 Flash',
		params: 550e9,
		hiddenSize: 5120,
		numLayers: 40,
		numHeads: 64,
		numKvHeads: 1,
		headDim: 512,
		intermediateSize: 2304, // moe_intermediate (per expert)
		vocabSize: 129280,
		moe: { numExperts: 384, expertsPerToken: 6 },
		activeParams: 15e9,
		// vision_config: 32-layer ViT, hidden 1024, patch 14, downsample 3,
		// max_image_tokens 1024. ~576 tokens for a ~1024px image at the 42px effective stride.
		vlm: { visionParams: 400e6, tokensPerImage: 576, defaultImagePx: 1024 }
	},
	{
		// moonshotai/Kimi-K2.5 — same 1T-MoE DeepseekV3-style architecture as Kimi K2
		// (hidden 7168, 61 layers, MLA 512+64 latent, 384 experts top-8 + 1 shared,
		// moe_intermediate 2048). 32B active/token.
		id: 'kimi-k2-5',
		name: 'Kimi K2.5',
		params: 1.04e12,
		hiddenSize: 7168,
		numLayers: 61,
		numHeads: 64,
		numKvHeads: 64,
		headDim: 192, // qk_nope 128 + qk_rope 64
		intermediateSize: 2048, // moe_intermediate
		vocabSize: 163840,
		moe: { numExperts: 384, expertsPerToken: 8 },
		mla: { kvLatentDim: 576 }, // kv_lora_rank 512 + qk_rope 64
		activeParams: 32e9
	},
	{
		// moonshotai/Kimi-K3. Hybrid: ~24 of 93 layers are MLA full-attention (the
		// only ones that cache KV, as a 512+64 latent); the rest are linear (KDA).
		id: 'kimi-k3',
		name: 'Kimi K3',
		params: 2.8e12,
		hiddenSize: 7168,
		numLayers: 93,
		numHeads: 96,
		numKvHeads: 96,
		headDim: 128,
		intermediateSize: 3072, // moe_intermediate
		vocabSize: 163840,
		moe: { numExperts: 896, expertsPerToken: 16 },
		mla: { kvLatentDim: 576 }, // kv_lora_rank 512 + qk_rope 64
		kvLayers: 24, // only the full-attention layers hold a KV cache
		activeParams: 104e9
	},
	{
		// zai-org/GLM-5.3 (GLM-5.2 shares an identical config). MLA + sparse
		// indexer. Active params not officially published — derived from the MoE
		// geometry (~50B); treat as an estimate.
		id: 'glm-5-3',
		name: 'GLM 5.3',
		params: 753e9,
		hiddenSize: 6144,
		numLayers: 78,
		numHeads: 64,
		numKvHeads: 64,
		headDim: 192,
		intermediateSize: 2048, // moe_intermediate
		vocabSize: 154880,
		moe: { numExperts: 256, expertsPerToken: 8 },
		mla: { kvLatentDim: 576 }, // kv_lora_rank 512 + qk_rope 64
		activeParams: 50e9 // derived, not card-stated
	},
	{
		id: 'glm-5-2',
		name: 'GLM 5.2',
		params: 753e9,
		hiddenSize: 6144,
		numLayers: 78,
		numHeads: 64,
		numKvHeads: 64,
		headDim: 192,
		intermediateSize: 2048,
		vocabSize: 154880,
		moe: { numExperts: 256, expertsPerToken: 8 },
		mla: { kvLatentDim: 576 },
		activeParams: 50e9 // derived, not card-stated
	},
	{
		// zai-org/GLM-5.3-Flash. MoE + MLA. Hybrid: 11 of 45 layers are sparse
		// full-attention (MLA, hold KV); the other 34 are gated-linear (no cache).
		id: 'glm-5-3-flash',
		name: 'GLM 5.3 Flash',
		params: 320e9,
		hiddenSize: 4096,
		numLayers: 45,
		numHeads: 64,
		numKvHeads: 64,
		headDim: 256, // qk_head_dim (NoPE MLA)
		intermediateSize: 2048, // moe_intermediate
		vocabSize: 154880,
		moe: { numExperts: 288, expertsPerToken: 8 },
		mla: { kvLatentDim: 512 }, // kv_lora_rank 512 + qk_rope 0
		kvLayers: 11,
		activeParams: 18e9
	},
	{
		// Qwen/Qwen3.8-Flash-Next. Fully-MoE. Hybrid: 12 of 48 layers are Qwen
		// sparse full-attention (hold KV); 36 are Gated DeltaNet (linear, no cache).
		// Total 180B includes a ~51B n-gram embedding table (stored, not active).
		id: 'qwen38-flash',
		name: 'Qwen3.8 Flash',
		params: 180e9,
		hiddenSize: 2560,
		numLayers: 48,
		numHeads: 24,
		numKvHeads: 2,
		headDim: 256,
		intermediateSize: 640, // moe_intermediate
		vocabSize: 248320,
		moe: { numExperts: 512, expertsPerToken: 10 },
		kvLayers: 12,
		activeParams: 6e9
	},
	{
		// XiaomiMiMo/MiMo-V2.5. MoE, hybrid SWA + global attention: 9 of 48 layers
		// are global full-attention (grow a KV cache); 39 are 128-token sliding
		// window. headDim 160 captures the asymmetric KV (qk 192 + v 128).
		id: 'mimo-v2-5',
		name: 'MiMo v2.5',
		params: 310e9,
		hiddenSize: 4096,
		numLayers: 48,
		numHeads: 64,
		numKvHeads: 8,
		headDim: 160,
		intermediateSize: 2048, // moe_intermediate
		vocabSize: 152576,
		moe: { numExperts: 256, expertsPerToken: 8 },
		kvLayers: 9,
		activeParams: 15e9
	},
	{
		// MiniMaxAI/MiniMax-M3. MoE with block-sparse attention — all 60 layers
		// keep a full KV cache (sparse compute, not a sparse cache). GQA 64/4.
		id: 'minimax-m3',
		name: 'MiniMax M3',
		params: 428e9,
		hiddenSize: 6144,
		numLayers: 60,
		numHeads: 64,
		numKvHeads: 4,
		headDim: 128,
		intermediateSize: 3072, // moe_intermediate
		vocabSize: 200064,
		moe: { numExperts: 128, expertsPerToken: 4 },
		activeParams: 23e9
	},
	{
		// tencent/Hy3 (Hunyuan V3). MoE, standard GQA 64/8 full attention on all
		// 80 layers. 1 shared expert (folded into the active-param split).
		id: 'hy3',
		name: 'Hunyuan V3 (Hy3)',
		params: 295e9,
		hiddenSize: 4096,
		numLayers: 80,
		numHeads: 64,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 1536, // moe_intermediate
		vocabSize: 120832,
		moe: { numExperts: 192, expertsPerToken: 8 },
		activeParams: 21e9
	},
	{
		// nvidia/Llama-3.1-Nemotron-Nano-8B-v1 — standard dense Llama 3.1 8B (GQA).
		id: 'nemotron-nano-8b',
		name: 'Llama-Nemotron Nano 8B',
		params: 8e9,
		hiddenSize: 4096,
		numLayers: 32,
		numHeads: 32,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 14336,
		vocabSize: 128256
	},
	{
		// nvidia/NVIDIA-Nemotron-Nano-9B-v2 — hybrid Mamba-2 + attention. Of 56
		// layers, 4 are attention (hold KV), 27 are Mamba-2 (SSM state), 25 are MLP.
		id: 'nemotron-nano-9b-v2',
		name: 'Nemotron Nano 9B v2',
		params: 8.9e9,
		hiddenSize: 4480,
		numLayers: 56,
		numHeads: 40,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 15680,
		vocabSize: 131072,
		kvLayers: 4,
		mamba: { layers: 27, stateBytesPerLayer: 5242880 } // 128 heads × 80 × 128 state, fp32
	},
	{
		// nvidia/NVIDIA-Nemotron-Nano-12B-v2 — hybrid. 62 layers: 6 attn, 28 Mamba, 28 MLP.
		id: 'nemotron-nano-12b-v2',
		name: 'Nemotron Nano 12B v2',
		params: 12.3e9,
		hiddenSize: 5120,
		numLayers: 62,
		numHeads: 40,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 20480,
		vocabSize: 131072,
		kvLayers: 6,
		mamba: { layers: 28, stateBytesPerLayer: 5242880 }
	},
	{
		// nvidia/Nemotron-H-8B-Base-8K — hybrid. 52 layers: 4 attn, 24 Mamba, 24 MLP.
		id: 'nemotron-h-8b',
		name: 'Nemotron-H 8B',
		params: 8e9,
		hiddenSize: 4096,
		numLayers: 52,
		numHeads: 32,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 21504,
		vocabSize: 131072,
		kvLayers: 4,
		mamba: { layers: 24, stateBytesPerLayer: 4194304 } // 128 × 64 × 128, fp32
	},
	{
		// nvidia/Nemotron-H-47B-Base-8K — hybrid. 98 layers: 5 attn, 45 Mamba, 48 MLP.
		id: 'nemotron-h-47b',
		name: 'Nemotron-H 47B',
		params: 47e9,
		hiddenSize: 8192,
		numLayers: 98,
		numHeads: 64,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 30720,
		vocabSize: 131072,
		kvLayers: 5,
		mamba: { layers: 45, stateBytesPerLayer: 16777216 } // 256 × 64 × 256, fp32
	},
	{
		// nvidia/Nemotron-H-56B-Base-8K — hybrid. 118 layers: 10 attn, 54 Mamba, 54 MLP.
		id: 'nemotron-h-56b',
		name: 'Nemotron-H 56B',
		params: 56e9,
		hiddenSize: 8192,
		numLayers: 118,
		numHeads: 64,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 32768,
		vocabSize: 131072,
		kvLayers: 10,
		mamba: { layers: 54, stateBytesPerLayer: 16777216 }
	},
	{
		// nvidia/Llama-3_3-Nemotron-Super-49B-v1 — NAS (DeciLM). 80 blocks, but only
		// 49 have live attention (the rest are attention-free), so KV lives on 49
		// layers. FFN width is variable per block; intermediateSize is nominal
		// (display only — dense weight/compute use the total param count).
		id: 'nemotron-super-49b',
		name: 'Llama-Nemotron Super 49B',
		params: 49e9,
		hiddenSize: 8192,
		numLayers: 80,
		numHeads: 64,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 28672, // nominal (NAS-variable per block)
		vocabSize: 128256,
		kvLayers: 49
	},
	{
		// nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 — NAS (DeciLM). 162 blocks, 64 with
		// live attention -> KV on 64 layers. FFN width variable per block;
		// intermediateSize is nominal (display only).
		id: 'nemotron-ultra-253b',
		name: 'Llama-Nemotron Ultra 253B',
		params: 253e9,
		hiddenSize: 16384,
		numLayers: 162,
		numHeads: 128,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 53248, // nominal (NAS-variable per block)
		vocabSize: 128256,
		kvLayers: 64
	},

	// --- Visual autoregressive (VAR) image models (transformer, visualAR) ---
	{
		// FoundationVision/var (VAR-d30) — Visual AutoRegressive next-scale prediction.
		// A decoder-only transformer (width = 64×depth, heads = depth). Class-conditioned
		// ImageNet 256², generated as 10 coarse→fine scales = 680 image tokens (set
		// Output tokens ≈ 680, Input tokens = 1 class token).
		id: 'var-d30',
		name: 'VAR-d30 (visual AR)',
		kind: 'transformer',
		visualAR: true,
		params: 2.0e9,
		hiddenSize: 1920,
		numLayers: 30,
		numHeads: 30,
		numKvHeads: 30,
		headDim: 64,
		intermediateSize: 7680, // 4 × 1920
		vocabSize: 4096 // VQ codebook
	},

	// --- Diffusion image models (kind: 'diffusion') ---
	{
		// stabilityai/stable-diffusion-xl-base-1.0 — UNet denoiser. Not token-based;
		// patch is an EFFECTIVE value so 2·params·tokens matches the conv UNet's real
		// per-forward FLOPs (~5 TFLOP @1024², vs the raw 128² latent grid).
		id: 'sdxl',
		name: 'Stable Diffusion XL',
		kind: 'diffusion',
		params: 2.567e9,
		hiddenSize: 1280,
		numLayers: 13,
		numHeads: 20,
		numKvHeads: 20,
		headDim: 64,
		intermediateSize: 5120,
		vocabSize: 0,
		diffusion: {
			arch: 'unet',
			vaeDownsample: 8,
			patch: 4,
			latentChannels: 4,
			defaultSteps: 50,
			defaultResolution: 1024,
			cfg: true,
			textEncoderParams: 0.82e9,
			vaeParams: 0.084e9
		}
	},
	{
		// black-forest-labs/FLUX.1-dev — 12B MMDiT. Guidance-distilled → NO CFG (no
		// batch doubling). 4096 latent tokens @1024² (patch 2 over an 8× VAE).
		id: 'flux1-dev',
		name: 'FLUX.1 [dev]',
		kind: 'diffusion',
		params: 11.9e9,
		hiddenSize: 3072,
		numLayers: 57, // 19 dual-stream + 38 single-stream MMDiT blocks
		numHeads: 24,
		numKvHeads: 24,
		headDim: 128,
		intermediateSize: 12288,
		vocabSize: 0,
		diffusion: {
			arch: 'mmdit',
			vaeDownsample: 8,
			patch: 2,
			latentChannels: 16,
			defaultSteps: 50,
			defaultResolution: 1024,
			cfg: false,
			textEncoderParams: 4.88e9, // CLIP-L 123M + T5-XXL 4.76B
			vaeParams: 0.084e9
		}
	},
	{
		// stabilityai/stable-diffusion-3.5-large — 8B MMDiT, standard CFG (batch ×2).
		id: 'sd35-large',
		name: 'Stable Diffusion 3.5 Large',
		kind: 'diffusion',
		params: 8.15e9,
		hiddenSize: 2432,
		numLayers: 38,
		numHeads: 38,
		numKvHeads: 38,
		headDim: 64,
		intermediateSize: 9728,
		vocabSize: 0,
		diffusion: {
			arch: 'mmdit',
			vaeDownsample: 8,
			patch: 2,
			latentChannels: 16,
			defaultSteps: 28,
			defaultResolution: 1024,
			cfg: true,
			textEncoderParams: 5.58e9, // CLIP-L + CLIP-G + T5-XXL
			vaeParams: 0.084e9
		}
	},
	{
		// PixArt-alpha/PixArt-Sigma-XL-2-1024-MS — 0.6B text-to-image DiT (Transformer2D),
		// T5-XXL cross-attention. 64² latent tokens @1024² (patch 2 over an 8× VAE).
		id: 'pixart-sigma',
		name: 'PixArt-Σ (XL/2)',
		kind: 'diffusion',
		params: 0.6e9,
		hiddenSize: 1152,
		numLayers: 28,
		numHeads: 16,
		numKvHeads: 16,
		headDim: 72,
		intermediateSize: 4608, // 4 × 1152
		vocabSize: 0,
		diffusion: {
			arch: 'dit',
			vaeDownsample: 8,
			patch: 2,
			latentChannels: 4,
			defaultSteps: 20,
			defaultResolution: 1024,
			cfg: true,
			textEncoderParams: 4.76e9, // T5-XXL
			vaeParams: 0.084e9
		}
	},
	{
		// facebook/DiT-XL-2 — the original DiT (Peebles & Xie). Class-conditioned on
		// ImageNet, so NO text encoder. 675M, 256² default (16² latent tokens).
		id: 'dit-xl2',
		name: 'DiT-XL/2 (256²)',
		kind: 'diffusion',
		params: 0.675e9,
		hiddenSize: 1152,
		numLayers: 28,
		numHeads: 16,
		numKvHeads: 16,
		headDim: 72,
		intermediateSize: 4608,
		vocabSize: 0,
		diffusion: {
			arch: 'dit',
			vaeDownsample: 8,
			patch: 2,
			latentChannels: 4,
			defaultSteps: 50,
			defaultResolution: 256,
			cfg: true,
			vaeParams: 0.084e9
			// class-conditioned: no textEncoderParams
		}
	},
	{
		// THUDM/CogVideoX-5b — 5B video DiT (Transformer3D). 3D VAE compresses 49
		// frames → 13 latent frames (temporal ratio 4). Native clip is 720×480, a 1.5:1
		// aspect: `resolution` is read as the short side (height), width = height×aspect.
		// Video → the quadratic attention term dominates.
		id: 'cogvideox-5b',
		name: 'CogVideoX-5B (video)',
		kind: 'diffusion',
		params: 5.0e9,
		hiddenSize: 3072,
		numLayers: 42,
		numHeads: 48,
		numKvHeads: 48,
		headDim: 64,
		intermediateSize: 12288, // 4 × 3072
		vocabSize: 0,
		diffusion: {
			arch: 'dit',
			vaeDownsample: 8,
			patch: 2,
			latentChannels: 16,
			defaultSteps: 50,
			defaultResolution: 512, // short side ≈ 480p; width = 1.5× via aspect
			aspect: 1.5, // 720:480 landscape
			cfg: true,
			frames: 13, // latent frames after 4× temporal compression of 49 input frames
			textEncoderParams: 4.76e9, // T5-XXL
			vaeParams: 0.2e9 // 3D causal VAE
		}
	},

	// --- ASR: Whisper family (encoder-decoder, kind: 'asr') ---
	// Fixed 30-second audio window → 1500 audio tokens through the encoder,
	// then autoregressive text decode via cross-attention. avgTextTokens is a
	// typical per-window transcription length (spoken English ≈ 2-4 wps ×
	// ~1.3 tokens/word × 30s ≈ 100-160 tokens).
	{
		id: 'whisper-large-v3',
		name: 'Whisper large-v3 (ASR)',
		kind: 'asr',
		params: 1.55e9,
		hiddenSize: 1280,
		numLayers: 64, // 32 enc + 32 dec (used for averaging only)
		numHeads: 20,
		numKvHeads: 20,
		headDim: 64,
		intermediateSize: 5120,
		vocabSize: 51866,
		asr: {
			encoderParams: 635e6,
			decoderParams: 906e6,
			audioTokens: 1500,
			audioWindowSec: 30,
			avgTextTokens: 128,
			numLayersDecoder: 32,
			numHeadsDecoder: 20,
			headDimDecoder: 64
		}
	},
	{
		// openai/whisper-large-v3-turbo — same encoder, 4-layer decoder (8× less
		// decoder). Real-world ~5-8× faster than large-v3 at similar accuracy.
		id: 'whisper-large-v3-turbo',
		name: 'Whisper large-v3 turbo (ASR)',
		kind: 'asr',
		params: 809e6,
		hiddenSize: 1280,
		numLayers: 36, // 32 enc + 4 dec
		numHeads: 20,
		numKvHeads: 20,
		headDim: 64,
		intermediateSize: 5120,
		vocabSize: 51866,
		asr: {
			encoderParams: 635e6,
			decoderParams: 174e6,
			audioTokens: 1500,
			audioWindowSec: 30,
			avgTextTokens: 128,
			numLayersDecoder: 4,
			numHeadsDecoder: 20,
			headDimDecoder: 64
		}
	},
	{
		// openai/whisper-medium — smaller model for edge / high-throughput serving.
		id: 'whisper-medium',
		name: 'Whisper medium (ASR)',
		kind: 'asr',
		params: 769e6,
		hiddenSize: 1024,
		numLayers: 48, // 24 enc + 24 dec
		numHeads: 16,
		numKvHeads: 16,
		headDim: 64,
		intermediateSize: 4096,
		vocabSize: 51865,
		asr: {
			encoderParams: 306e6,
			decoderParams: 463e6,
			audioTokens: 1500,
			audioWindowSec: 30,
			avgTextTokens: 128,
			numLayersDecoder: 24,
			numHeadsDecoder: 16,
			headDimDecoder: 64
		}
	},

	// --- Text-to-speech (autoregressive audio-codec decoders) ---
	// TTS here is an autoregressive transformer that emits neural-audio-codec tokens,
	// so it rides the standard LLM decode path: text prompt in (input tokens), audio
	// codec frames out (output tokens; a codec runs ~50–75 frames/s of audio, so ~10 s
	// of speech ≈ a few hundred tokens). The `tts` flag only groups them separately.
	// Params are approximate (public repos don't all publish a config).
	{
		// coqui/XTTS-v2 — GPT-2-style AR backbone over VQ audio codes + a vocoder.
		id: 'xtts-v2',
		name: 'XTTS-v2 (TTS)',
		tts: true,
		params: 467e6,
		hiddenSize: 1024,
		numLayers: 30,
		numHeads: 16,
		numKvHeads: 16,
		headDim: 64,
		intermediateSize: 4096,
		vocabSize: 8194 // text tokens + 1024 audio codes
	},
	{
		// fishaudio/fish-speech-1.5 — dual-AR (Llama-style) over a grouped audio codec.
		id: 'fish-speech-1_5',
		name: 'Fish Speech 1.5 (TTS)',
		tts: true,
		params: 500e6,
		hiddenSize: 1024,
		numLayers: 24,
		numHeads: 16,
		numKvHeads: 16,
		headDim: 64,
		intermediateSize: 4096,
		vocabSize: 32000
	},

	// --- Vision-Language Models (transformer LLMs with a `vlm` decoration) ---
	// Each is a normal LLM that also reads images: N images × tokensPerImage extra
	// input tokens per request; a small vision encoder is held resident.
	{
		// Qwen/Qwen2.5-VL-7B-Instruct. LLM: 3584/28L/28H, GQA 4KV, MLP 18944, 152k vocab.
		// Vision: SigLIP-flavour, 1280/32L/16H, patch 14, spatial-merge 2 (effective 28).
		// At 448² default → (448/28)² = 256 tokens per image; scales with resolution.
		id: 'qwen25-vl-7b',
		name: 'Qwen2.5-VL 7B',
		params: 7.62e9,
		hiddenSize: 3584,
		numLayers: 28,
		numHeads: 28,
		numKvHeads: 4,
		headDim: 128,
		intermediateSize: 18944,
		vocabSize: 152064,
		vlm: { visionParams: 675e6, tokensPerImage: 256, defaultImagePx: 448 }
	},
	{
		// mistralai/Pixtral-12B-2409. LLM: 5120/40L/32H, GQA 8KV, 131k vocab.
		// Vision: 1024/24L/16H, patch 16, image up to 1024². 512² default → 1024 tokens.
		id: 'pixtral-12b',
		name: 'Pixtral 12B',
		params: 12.7e9,
		hiddenSize: 5120,
		numLayers: 40,
		numHeads: 32,
		numKvHeads: 8,
		headDim: 128,
		intermediateSize: 14336,
		vocabSize: 131072,
		vlm: { visionParams: 400e6, tokensPerImage: 1024, defaultImagePx: 512 }
	},
	{
		// llava-hf/llava-onevision-qwen2-7b-ov. Qwen2-7B backbone + SigLIP-400M vision.
		// 384² image → 729 tokens (27×27 patches, no bilinear pool at base scale).
		id: 'llava-ov-7b',
		name: 'LLaVA-OneVision 7B',
		params: 7.62e9,
		hiddenSize: 3584,
		numLayers: 28,
		numHeads: 28,
		numKvHeads: 4,
		headDim: 128,
		intermediateSize: 18944,
		vocabSize: 152064,
		vlm: { visionParams: 400e6, tokensPerImage: 729, defaultImagePx: 384 }
	},

	// --- Vision-Language-Action (VLA) models for robotics (kind: 'vla') ---
	{
		// lerobot/pi0 (Physical Intelligence pi-0). PaliGemma-3B backbone + 300M action
		// expert; SigLIP-400M vision, 3 cameras at 224², 256 tokens each. Flow-matching
		// chunks of 50 actions (~1s at 50Hz control). See lerobot config.
		id: 'pi-0',
		name: 'π₀ (pi-zero)',
		kind: 'vla',
		params: 2.9e9, // PaliGemma-3B backbone
		hiddenSize: 2048, // Gemma-2B width
		numLayers: 18,
		numHeads: 8,
		numKvHeads: 1,
		headDim: 256,
		intermediateSize: 16384,
		vocabSize: 256000,
		vla: {
			chunkSize: 50,
			actionDim: 32,
			flowSteps: 10,
			controlHz: 50,
			expertParams: 300e6,
			camerasPerObs: 3,
			tokensPerImage: 256,
			defaultImagePx: 224,
			visionParams: 400e6,
			stateDim: 32
		}
	},
	{
		// lerobot/smolvla_base. SmolVLM2-500M backbone (frozen vision), 16 VLM layers,
		// action expert at 0.75× width. Padded images 512² across 3 cameras. Small
		// enough to run at kHz-class control on a single GPU.
		id: 'smolvla',
		name: 'SmolVLA (base)',
		kind: 'vla',
		params: 450e6, // SmolVLM2-500M-scale backbone
		hiddenSize: 960,
		numLayers: 16,
		numHeads: 15,
		numKvHeads: 5,
		headDim: 64,
		intermediateSize: 2560,
		vocabSize: 49152,
		vla: {
			chunkSize: 50,
			actionDim: 32,
			flowSteps: 10,
			controlHz: 50,
			expertParams: 100e6,
			camerasPerObs: 3,
			tokensPerImage: 64,
			defaultImagePx: 512,
			visionParams: 90e6,
			stateDim: 32
		}
	},

	// --- Encoder-only text models: embeddings & rerankers (kind: 'encoder') ---
	{
		// BAAI/bge-m3 — XLM-RoBERTa base, multilingual embedding + retrieval.
		// hidden 1024, 24 layers, 16 heads, vocab 250002, max 8194 tokens.
		id: 'bge-m3',
		name: 'BGE-M3 (embedding)',
		kind: 'encoder',
		params: 568e6,
		hiddenSize: 1024,
		numLayers: 24,
		numHeads: 16,
		numKvHeads: 16,
		headDim: 64,
		intermediateSize: 4096,
		vocabSize: 250002,
		encoder: { task: 'embedding', defaultSeqLen: 512, maxSeqLen: 8194, embedDim: 1024 }
	},
	{
		// jinaai/jina-embeddings-v3 — same XLM-RoBERTa shape as BGE-M3, LoRA task heads.
		id: 'jina-embed-v3',
		name: 'Jina Embeddings v3',
		kind: 'encoder',
		params: 570e6,
		hiddenSize: 1024,
		numLayers: 24,
		numHeads: 16,
		numKvHeads: 16,
		headDim: 64,
		intermediateSize: 4096,
		vocabSize: 250002,
		encoder: { task: 'embedding', defaultSeqLen: 512, maxSeqLen: 8194, embedDim: 1024 }
	},
	{
		// intfloat/e5-mistral-7b-instruct — Mistral-7B backbone repurposed as an
		// instruction-tuned embedding model. Encoder-only usage: last-token pooled.
		id: 'e5-mistral-7b',
		name: 'E5-Mistral-7B (embedding)',
		kind: 'encoder',
		params: 7.11e9,
		hiddenSize: 4096,
		numLayers: 32,
		numHeads: 32,
		numKvHeads: 8, // GQA — irrelevant here (no KV cache) but faithful to the config
		headDim: 128,
		intermediateSize: 14336,
		vocabSize: 32000,
		encoder: { task: 'embedding', defaultSeqLen: 512, maxSeqLen: 32768, embedDim: 4096 }
	},
	{
		// BAAI/bge-reranker-v2-m3 — same XLM-RoBERTa backbone as BGE-M3 but with a
		// classification head that scores (query, doc) pairs. Sequence length is the
		// concatenation of both, so default is 2× a normal doc.
		id: 'bge-reranker-v2-m3',
		name: 'BGE reranker v2-m3',
		kind: 'encoder',
		params: 568e6,
		hiddenSize: 1024,
		numLayers: 24,
		numHeads: 16,
		numKvHeads: 16,
		headDim: 64,
		intermediateSize: 4096,
		vocabSize: 250002,
		encoder: { task: 'reranker', defaultSeqLen: 1024, maxSeqLen: 8194 }
	},

	// --- JEPA video world models (kind: 'jepa') ---
	// ViT encoder + a light predictor (384-wide, 12-layer, ~21M). Inference is a
	// single forward over spatial×temporal patches; no KV, no autoregression.
	{
		// facebook/vjepa2-vitg-fpc64-256 — the V-JEPA 2 flagship (ViT-g/16, ~1B),
		// 64 frames at 256px. hidden 1408, 40 layers, 22 heads, mlp_ratio 4.36.
		id: 'vjepa2-vitg',
		name: 'V-JEPA 2 (ViT-g)',
		kind: 'jepa',
		params: 1.01e9,
		hiddenSize: 1408,
		numLayers: 40,
		numHeads: 22,
		numKvHeads: 22,
		headDim: 64,
		intermediateSize: 6144, // mlp_ratio 4.36 × 1408
		vocabSize: 0,
		jepa: {
			patch: 16,
			tubelet: 2,
			defaultFrames: 64,
			defaultResolution: 256,
			predictorParams: 21e6
		}
	},
	{
		// facebook/vjepa2-ac-vitg — V-JEPA 2-AC: the ViT-g encoder frozen, plus a ~300M
		// action-conditioned predictor rolled out for planning (MPC). Inference = one
		// encode + a rollout over the planning horizon (default 16). Real MPC also
		// samples many action trajectories (CEM), so treat the rate as per-rollout.
		id: 'vjepa2-ac',
		name: 'V-JEPA 2-AC (world model)',
		kind: 'jepa',
		params: 1.01e9, // frozen ViT-g encoder
		hiddenSize: 1408,
		numLayers: 40,
		numHeads: 22,
		numKvHeads: 22,
		headDim: 64,
		intermediateSize: 6144,
		vocabSize: 0,
		jepa: {
			patch: 16,
			tubelet: 2,
			defaultFrames: 64,
			defaultResolution: 256,
			predictorParams: 300e6, // AC predictor
			predHidden: 1024,
			predLayers: 24,
			rolloutSteps: 16 // planning horizon
		}
	},
	{
		// facebook/vjepa2-vith-fpc64-256 — V-JEPA 2 ViT-H/16 (~635M), 64 frames @256.
		id: 'vjepa2-vith',
		name: 'V-JEPA 2 (ViT-H)',
		kind: 'jepa',
		params: 6.35e8,
		hiddenSize: 1280,
		numLayers: 32,
		numHeads: 16,
		numKvHeads: 16,
		headDim: 80,
		intermediateSize: 5120,
		vocabSize: 0,
		jepa: {
			patch: 16,
			tubelet: 2,
			defaultFrames: 64,
			defaultResolution: 256,
			predictorParams: 21e6
		}
	},
	{
		// facebook/vjepa2-vitl-fpc16-256 — V-JEPA 2 ViT-L/16 (~300M), 16 frames @256.
		id: 'vjepa2-vitl',
		name: 'V-JEPA 2 (ViT-L)',
		kind: 'jepa',
		params: 3.02e8,
		hiddenSize: 1024,
		numLayers: 24,
		numHeads: 16,
		numKvHeads: 16,
		headDim: 64,
		intermediateSize: 4096,
		vocabSize: 0,
		jepa: {
			patch: 16,
			tubelet: 2,
			defaultFrames: 16,
			defaultResolution: 256,
			predictorParams: 21e6
		}
	},
	{
		// facebookresearch/jepa — the original V-JEPA (2024) flagship, ViT-H/16 (~633M)
		// trained on video at 16 frames, 224px.
		id: 'vjepa-vith',
		name: 'V-JEPA (ViT-H, 2024)',
		kind: 'jepa',
		params: 6.33e8,
		hiddenSize: 1280,
		numLayers: 32,
		numHeads: 16,
		numKvHeads: 16,
		headDim: 80,
		intermediateSize: 5120,
		vocabSize: 0,
		jepa: {
			patch: 16,
			tubelet: 2,
			defaultFrames: 16,
			defaultResolution: 224,
			predictorParams: 21e6
		}
	}
];

// GPU presets. Bandwidth in TB/s, tensor throughput in dense TFLOPS,
// interconnect in GB/s aggregate per GPU.
export const GPUS: GpuSpec[] = [
	// Blackwell figures are per-GPU dense (non-sparse), derived from NVIDIA's
	// 8-GPU/rack aggregates. B300 raises HBM + FP4 over B200 but FP8/BF16 and
	// bandwidth are unchanged. (NVIDIA HGX/DGX B200/B300, Blackwell Ultra blog.)
	{
		id: 'b300',
		name: 'B300 (Blackwell Ultra) 288GB',
		memoryGiB: 288,
		memBandwidthTBs: 8.0,
		fp16Tflops: 2250,
		fp8Tflops: 4500,
		fp4Tflops: 13500, // 1.5x B200 dense FP4
		nvlinkGBs: 1800, // NVLink 5, 1.8 TB/s per GPU
		hasNvlink: true,
		tdpWatts: 1400, // NVIDIA HGX B300 Ultra board TDP
		pcieGBs: 63, // PCIe Gen5 x16
		die: DIE_BLACKWELL_DC
	},
	{
		id: 'b200',
		name: 'B200 (Blackwell) 192GB',
		memoryGiB: 192,
		memBandwidthTBs: 8.0,
		fp16Tflops: 2250,
		fp8Tflops: 4500,
		fp4Tflops: 9000, // 2x FP8 dense
		nvlinkGBs: 1800, // NVLink 5, 1.8 TB/s per GPU
		hasNvlink: true,
		tdpWatts: 1000, // HGX B200 SXM board TDP
		pcieGBs: 63, // PCIe Gen5 x16
		mfuFactor: 0.81, // LLM prefill sustains ~0.81× the H100-anchored MFU
		die: DIE_BLACKWELL_DC
	},
	{
		id: 'h200',
		name: 'H200 SXM 141GB',
		memoryGiB: 141,
		memBandwidthTBs: 4.8,
		fp16Tflops: 989.5,
		fp8Tflops: 1979,
		fp4Tflops: 0, // no native FP4 (Hopper)
		nvlinkGBs: 900,
		hasNvlink: true,
		tdpWatts: 700, // H200 SXM board TDP
		pcieGBs: 63, // PCIe Gen5 x16 host (Gen4 hosts → 32)
		die: DIE_HOPPER
	},
	{
		id: 'h100-sxm',
		name: 'H100 SXM 80GB',
		memoryGiB: 80,
		memBandwidthTBs: 3.35,
		fp16Tflops: 989.5,
		fp8Tflops: 1979,
		fp4Tflops: 0, // no native FP4 (Hopper)
		nvlinkGBs: 900,
		hasNvlink: true,
		tdpWatts: 700, // H100 SXM board TDP
		pcieGBs: 63, // PCIe Gen5 x16
		die: DIE_HOPPER
	},
	{
		id: 'a100-80',
		name: 'A100 SXM 80GB',
		memoryGiB: 80,
		memBandwidthTBs: 2.039,
		fp16Tflops: 312,
		fp8Tflops: 624, // INT8 proxy; A100 has no native FP8
		fp4Tflops: 0, // no native FP4 (Ampere)
		nvlinkGBs: 600,
		hasNvlink: true,
		tdpWatts: 400, // A100 SXM 80GB TDP
		pcieGBs: 32, // PCIe Gen4 x16
		die: DIE_AMPERE
	},
	{
		id: 'a100-40',
		name: 'A100 SXM 40GB',
		memoryGiB: 40,
		memBandwidthTBs: 1.555,
		fp16Tflops: 312,
		fp8Tflops: 624,
		fp4Tflops: 0, // no native FP4 (Ampere)
		nvlinkGBs: 600,
		hasNvlink: true,
		tdpWatts: 400, // A100 SXM 40GB TDP
		pcieGBs: 32, // PCIe Gen4 x16
		die: DIE_AMPERE
	},
	{
		id: 'l40s',
		name: 'L40S 48GB',
		memoryGiB: 48,
		memBandwidthTBs: 0.864,
		fp16Tflops: 362,
		fp8Tflops: 733,
		fp4Tflops: 0, // no native FP4 (Ada)
		nvlinkGBs: 64, // PCIe gen4 x16
		hasNvlink: false,
		tdpWatts: 350, // L40S max board TDP
		pcieGBs: 32, // PCIe Gen4 x16
		mfuFactor: 0.75, // compute-bound paths ~0.75× the H100-calibrated MFU
		die: DIE_ADA_DC
	},
	{
		id: 'rtx4090',
		name: 'RTX 4090 24GB',
		memoryGiB: 24,
		memBandwidthTBs: 1.008,
		fp16Tflops: 165.2,
		fp8Tflops: 660.6,
		fp4Tflops: 0, // no native FP4 (Ada)
		nvlinkGBs: 64,
		hasNvlink: false,
		tdpWatts: 450,
		pcieGBs: 32, // PCIe Gen4 x16
		die: DIE_ADA_CONSUMER
	},
	// RTX PRO (Blackwell workstation): NVIDIA publishes only FP4-sparse "AI TOPS",
	// no dense FP16/FP8. Dense figures below are derived from that anchor (6000)
	// and scaled by CUDA-core count (4500); treat as estimates. Both are PCIe-only
	// (NVIDIA dropped NVLink from the RTX PRO line). GDDR7. (NVIDIA datasheets.)
	{
		id: 'rtx-pro-6000',
		name: 'RTX PRO 6000 Blackwell 96GB',
		memoryGiB: 96,
		memBandwidthTBs: 1.792,
		fp16Tflops: 500,
		fp8Tflops: 1000,
		fp4Tflops: 2000, // 2x FP8 (Blackwell)
		nvlinkGBs: 128, // PCIe gen5 x16, no NVLink
		hasNvlink: false,
		tdpWatts: 600, // RTX PRO 6000 Blackwell (Server Edition) TDP
		pcieGBs: 63, // PCIe Gen5 x16
		die: DIE_BLACKWELL_WS
	},
	{
		id: 'rtx-pro-4500',
		name: 'RTX PRO 4500 Blackwell 32GB',
		memoryGiB: 32,
		memBandwidthTBs: 0.896,
		fp16Tflops: 220,
		fp8Tflops: 440,
		fp4Tflops: 880, // 2x FP8 (Blackwell)
		nvlinkGBs: 128, // PCIe gen5 x16, no NVLink
		hasNvlink: false,
		tdpWatts: 200, // RTX PRO 4500 Blackwell workstation TDP
		pcieGBs: 63, // PCIe Gen5 x16
		die: DIE_BLACKWELL_WS
	},
	// --- Inference / previous-gen GPUs ---
	{
		// Ada, has FP8. GDDR6.
		id: 'l4',
		name: 'L4 24GB',
		memoryGiB: 24,
		memBandwidthTBs: 0.3,
		fp16Tflops: 121,
		fp8Tflops: 242,
		fp4Tflops: 0,
		nvlinkGBs: 64, // PCIe gen4 x16, no NVLink
		hasNvlink: false,
		tdpWatts: 72, // L4 board TDP (extremely efficient inference part)
		pcieGBs: 32, // PCIe Gen4 x16
		die: DIE_ADA_DC
	},
	{
		// Ampere, no FP8. A10G is a cloud variant of the A10; compute uses A10 specs.
		id: 'a10g',
		name: 'A10G 24GB',
		memoryGiB: 24,
		memBandwidthTBs: 0.6,
		fp16Tflops: 125,
		fp8Tflops: 0, // no FP8 on Ampere
		fp4Tflops: 0,
		nvlinkGBs: 64,
		hasNvlink: false,
		tdpWatts: 300, // A10G board TDP
		pcieGBs: 32, // PCIe Gen4 x16
		mfuFactor: 0.58, // LLM prefill ~0.58× the H100-anchored MFU
		die: DIE_AMPERE_G
	},
	{
		// Turing, no FP8. GDDR6.
		id: 't4',
		name: 'T4 16GB',
		memoryGiB: 16,
		memBandwidthTBs: 0.32,
		fp16Tflops: 65,
		fp8Tflops: 0,
		fp4Tflops: 0,
		nvlinkGBs: 64,
		hasNvlink: false,
		tdpWatts: 70, // T4 board TDP
		pcieGBs: 16, // PCIe Gen3 x16 (Turing)
		mfuFactor: 0.46, // LLM prefill ~0.46× the H100-anchored MFU
		die: DIE_TURING
	}
];

export const MODELS_BY_ID = new Map(MODELS.map((m) => [m.id, m]));
export const GPUS_BY_ID = new Map(GPUS.map((g) => [g.id, g]));
