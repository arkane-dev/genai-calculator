import {
	COLORS,
	FABRICS_BY_ID,
	GPUS_BY_ID,
	MODELS_BY_ID,
	NVLINK_A2A_EFF,
	NVLINK_ALLREDUCE_EFF,
	PP_P2P_EFF,
	WEIGHT_FORMATS_BY_ID
} from './data';
import type { Config, FabricSpec, GpuSpec, ModelSpec, Profile, WeightFormat } from './types';

const GiB = 1024 ** 3;
const TB = 1e12;
const GB = 1e9;

// Realized fractions of the theoretical peaks. Decode is bandwidth bound, so
// memory-bandwidth utilization runs high; model-flop utilization is lower.
// Calibrated against real H100 runs:
//  - prefill (compute bound) sustained ~0.75 of peak -> MFU set to 0.74.
//  - decode hardware bandwidth was ~0.72 of peak. We keep MBU at 0.80 because
//    the step below counts KV at full context (worst case) while a real run
//    ramps up to it; 0.80 lands decode throughput within ~10% and stays
//    conservative. Lowering MBU would double-count that headroom.
export const MBU = 0.8; // memory bandwidth utilization
export const MFU = 0.74; // model flop utilization (H100 anchor)
// Per-family MFU efficiency, applied on top of the per-GPU mfuFactor (which is
// calibrated on the big LLM prefill). From hardware runs:
//   - encoder (BGE-M3): small bidirectional GEMMs sustain a HIGHER fraction of peak
//     than the LLM prefill, so the scalar mfuFactor over-penalises them ~1.5×.
//   - jepa (V-JEPA2 ViT): 64-frame ViT forwards ran ~2× SLOWER than the roofline
//     (large-token attention), so effective MFU is ~0.5× the LLM figure.
// LLM/ASR/diffusion/VLA use 1.0 (diffusion carries its own FLOP correction).
export const FAMILY_MFU = { encoder: 1.5, jepa: 0.5 } as const;
export const OVERHEAD_BYTES = 1.5 * GiB; // CUDA context + framework + fragmentation
export const ACT_BYTES = 2; // activations stay fp16 regardless of weight quant
export const KV_BLOCK = 16; // paged-attention block size, tokens
const DRAFT_COST_RATIO = 0.2; // a spec-decode draft step ≈ this fraction of a target step

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

/** Parameters that live inside the MoE experts (SwiGLU: gate, up, down). */
export function expertParams(m: ModelSpec): number {
	if (!m.moe) return 0;
	return m.numLayers * m.moe.numExperts * 3 * m.hiddenSize * m.intermediateSize;
}

/** Split total params into expert vs non-expert. When the card gives active
 * params, solve exactly from (total, active, top-k, numExperts) — robust at any
 * scale. Otherwise fall back to the geometric estimate (clamped below total). */
export function moeSplit(m: ModelSpec): { nonExpert: number; expert: number } {
	if (!m.moe) return { nonExpert: m.params, expert: 0 };
	const kOverE = m.moe.expertsPerToken / m.moe.numExperts;
	let expert =
		m.activeParams && kOverE < 1 ? (m.params - m.activeParams) / (1 - kOverE) : expertParams(m);
	expert = Math.max(0, Math.min(expert, m.params * 0.98));
	return { nonExpert: Math.max(0, m.params - expert), expert };
}

/** Non-expert params: attention, embeddings, router. Sharded by TP (and PP). */
export function nonExpertParams(m: ModelSpec): number {
	return moeSplit(m).nonExpert;
}

/** Expert params that fire for a token (top-k of the experts). Sharded by EP. */
export function activeExpertParams(m: ModelSpec): number {
	if (!m.moe) return 0;
	return (m.moe.expertsPerToken / m.moe.numExperts) * moeSplit(m).expert;
}

/** Params actually multiplied for a single token. Prefer the card-stated value. */
export function activeParams(m: ModelSpec): number {
	return m.activeParams ?? nonExpertParams(m) + activeExpertParams(m);
}

export function computePeak(g: GpuSpec, tier: WeightFormat['tier']): number {
	// FP4 uses the native FP4 path where it exists (2x FP8), else the format is
	// dequantized to fp16 and runs at the fp16 rate.
	// A format falls back to the fp16 rate on GPUs with no native path for it
	// (no FP8 pre-Hopper; no FP4 pre-Blackwell) — the weights are dequantized.
	if (tier === 'fp8') return (g.fp8Tflops > 0 ? g.fp8Tflops : g.fp16Tflops) * TB;
	if (tier === 'fp4') return (g.fp4Tflops > 0 ? g.fp4Tflops : g.fp16Tflops) * TB;
	return g.fp16Tflops * TB;
}

interface Collective {
	bytes: number; // per-GPU bytes moved per decode step
	overFabric: boolean; // crosses a node boundary
	time: number; // seconds
}

// Diffusion: inference is `steps` denoising passes over a fixed latent (no KV,
// no autoregression). Each step is a full forward over the latent tokens, so it
// is compute-bound like a transformer prefill. Reuses the roofline + TP/PP
// parallelism + fills; reports images/sec instead of tokens/sec.
function computeDiffusion(
	cfg: Config,
	m: ModelSpec,
	g: GpuSpec,
	fmt: WeightFormat,
	fab: FabricSpec
): Profile {
	const ds = m.diffusion!;
	const steps = Math.max(1, cfg.steps ?? ds.defaultSteps);
	const res = Math.max(64, cfg.resolution ?? ds.defaultResolution);
	const frames = Math.max(1, cfg.frames ?? ds.frames ?? 1);
	const cfgOn = cfg.guidance ?? ds.cfg;
	const cfgFactor = cfgOn ? 2 : 1; // CFG runs a cond + uncond forward
	const batch = Math.max(1, cfg.batchSize);

	// dense parallelism: TP (intra-node) + PP; no MoE/EP for these models
	const pp = cfg.ppEnabled ? Math.max(1, Math.min(cfg.pp, cfg.numGpus)) : 1;
	const tp = Math.max(1, Math.min(cfg.tp, cfg.numGpus));
	const layerShard = tp * pp;
	const layersPerStage = m.numLayers / pp;

	// latent tokens per image (compute tokens for a conv UNet via effective patch).
	// `aspect` (w:h) scales the square grid area for non-square video (e.g. 720×480).
	const side = res / ds.vaeDownsample / ds.patch;
	const latentTokens = Math.max(1, Math.round(side * side * (ds.aspect ?? 1) * frames));
	const effTokens = batch * latentTokens * cfgFactor;

	// ---- memory ----
	const wb = fmt.bitsPerWeight / 8;
	const denoiserW = (m.params * wb) / layerShard;
	const extraW = (((ds.textEncoderParams ?? 0) + (ds.vaeParams ?? 0)) * wb) / tp;
	const weights = denoiserW + extraW;
	const perImageAct = (2 * latentTokens * cfgFactor * m.hiddenSize * ACT_BYTES) / layerShard;
	const activations = perImageAct * batch;
	const overhead = OVERHEAD_BYTES;
	const used = weights + activations + overhead;
	const capacity = g.memoryGiB * GiB;
	const fits = used <= capacity;
	const headroom = capacity - used;
	const actBudget = capacity - weights - overhead;
	const maxImages = perImageAct > 0 ? Math.max(0, Math.floor(actBudget / perImageAct)) : 0;

	const segments: Profile['segments'] = [
		{
			key: 'weights',
			label: 'Weights (denoiser + encoders)',
			bytes: weights,
			color: COLORS.weights
		},
		{ key: 'activations', label: 'Activations', bytes: activations, color: COLORS.activations },
		{ key: 'overhead', label: 'Overhead', bytes: overhead, color: COLORS.overhead }
	];
	if (fits) segments.push({ key: 'free', label: 'Free', bytes: headroom, color: COLORS.free });
	else
		segments.push({ key: 'overflow', label: 'Overflow', bytes: -headroom, color: COLORS.overflow });

	// ---- roofline per denoising step ----
	// Linear (2·params·tokens). For a real token grid (DiT/MMDiT) the quadratic
	// self-attention term matters at high token counts (esp. video), so add it.
	// A conv UNet's `patch` is an EFFECTIVE FLOP-matching value that already folds
	// in its attention, so it gets no separate term.
	const attnSeqs = batch * cfgFactor;
	const attnFlops =
		ds.arch === 'unet'
			? 0
			: 4 * m.numLayers * latentTokens * latentTokens * m.hiddenSize * attnSeqs;
	// Empirical correction from hardware runs: the linear per-step FLOP-match
	// undercounts real work ~1.6× for UNet (SDXL) and DiT (PixArt-Σ, CogVideoX). Large
	// MMDiT (FLUX.1 12B, SD3.5 8B) is already accurate at raw FLOPs — FLUX landed 1.01×
	// with no boost — so the correction is NOT applied there. (SD3.5 remains ~1.6× slow
	// and tiny DiT like DiT-XL/2 at 256² is launch-bound — residual, model-specific.)
	const DIFFUSION_FLOP_CORRECTION = 1.6;
	const linCorr = ds.arch === 'mmdit' ? 1 : DIFFUSION_FLOP_CORRECTION;
	const flopsPerStep = (2 * m.params * effTokens * linCorr + attnFlops) / layerShard;
	const peakFlops = computePeak(g, fmt.tier);
	const computeTime = flopsPerStep / (peakFlops * MFU * (g.mfuFactor ?? 1));
	const bytesPerStep = weights + activations; // weights read once/step + activations
	const peakBW = g.memBandwidthTBs * TB;
	const memTime = bytesPerStep / (peakBW * MBU);
	const stageWork = Math.max(memTime, computeTime);

	// ---- interconnect (TP all-reduce, PP hand-off) ----
	const peakNvlink = g.nvlinkGBs * GB;
	const peakNet = (fab.gbps * GB) / 8;
	const tpSpans = tp > cfg.gpusPerNode;
	const ppSpans = layerShard > cfg.gpusPerNode;
	let tpBytes = 0;
	let tpTime = 0;
	if (tp > 1) {
		tpBytes = layersPerStage * 2 * ((2 * (tp - 1)) / tp) * effTokens * m.hiddenSize * ACT_BYTES;
		tpTime =
			tpBytes /
			((tpSpans ? peakNet : peakNvlink) * (tpSpans ? fab.allReduceEff : NVLINK_ALLREDUCE_EFF));
	}
	let ppBytes = 0;
	let ppTime = 0;
	if (pp > 1) {
		ppBytes = effTokens * m.hiddenSize * ACT_BYTES;
		ppTime = ppBytes / ((ppSpans ? peakNet : peakNvlink) * PP_P2P_EFF);
	}
	const stepTime = stageWork + tpTime; // per denoising step (TP overlaps in-stage)
	const commTime = tpTime;

	const dp = Math.max(1, Math.floor(cfg.numGpus / layerShard));
	const secPerImage = steps * (pp * stepTime + Math.max(0, pp - 1) * ppTime);
	const imagesPerSec = (batch * dp) / secPerImage;

	// ---- fills ----
	const bandwidthUsed = bytesPerStep / stageWork;
	const computeAchieved = flopsPerStep / stageWork;
	const memBwFrac = clamp01(bandwidthUsed / peakBW);
	const computeFrac = clamp01(computeAchieved / peakFlops);
	const l2Frac = clamp01(Math.max(memBwFrac, computeFrac));
	const nvlinkBytes = (tpSpans ? 0 : tpBytes) + (ppSpans ? 0 : ppBytes);
	const fabricBytes = (tpSpans ? tpBytes : 0) + (ppSpans ? ppBytes : 0);
	const anyIntra = tp > 1 || pp > 1;
	const nvlinkFrac = anyIntra ? clamp01(nvlinkBytes / stageWork / peakNvlink) : 0;
	const netFrac = clamp01(fabricBytes / stageWork / peakNet);

	const heaviest = Math.max(memTime, computeTime, commTime);
	const bottleneck: Profile['bottleneck'] =
		commTime === heaviest && commTime > 0 && fabricBytes > 0
			? 'network'
			: memTime >= computeTime
				? 'memory'
				: 'compute';
	const numNodes = Math.ceil(cfg.numGpus / cfg.gpusPerNode);

	return {
		perGpu: { weights, kv: 0, activations, overhead, used, capacity, fits, headroom },
		segments,
		kvWasteFactor: 1,
		kvPerSeqBytes: 0,
		maxConcurrentSeqs: maxImages,
		phase: 'prefill',
		stepTimeMs: stepTime * 1000,
		throughputTps: 0,
		perUserTps: 0,
		ttftMs: 0,
		bottleneck,
		memBwFrac,
		computeFrac,
		l2Frac,
		nvlinkFrac,
		netFrac,
		bandwidthUsedGBs: bandwidthUsed / GB,
		bandwidthPeakGBs: peakBW / GB,
		computeAchievedTflops: computeAchieved / TB,
		computePeakTflops: peakFlops / TB,
		nvlinkUsedGBs: nvlinkBytes / stageWork / GB,
		nvlinkPeakGBs: peakNvlink / GB,
		netUsedGBs: fabricBytes / stageWork / GB,
		netPeakGBs: peakNet / GB,
		comm: {
			tpActive: tp > 1,
			tpOverFabric: tpSpans && tp > 1,
			tpGBs: tpBytes / stageWork / GB,
			epActive: false,
			epOverFabric: false,
			epGBs: 0,
			ppActive: pp > 1,
			ppOverFabric: ppSpans && pp > 1,
			ppGBs: ppBytes / stageWork / GB
		},
		fabric: {
			label: fab.label,
			kind: fab.kind,
			peakGBs: peakNet / GB,
			allReduceEff: fab.allReduceEff,
			a2aEff: fab.a2aEff
		},
		cluster: {
			numGpus: cfg.numGpus,
			tp,
			pp,
			ep: 1,
			dp,
			gpusPerNode: cfg.gpusPerNode,
			numNodes,
			crossesFabric: fabricBytes > 0,
			scaleOut: numNodes <= 1 ? 'single' : fabricBytes > 0 ? 'model-across-nodes' : 'data-parallel'
		},
		activeParams: m.params,
		diffusion: {
			imagesPerSec,
			secPerImage,
			stepTimeMs: stepTime * 1000,
			steps,
			latentTokens,
			cfg: cfgOn
		}
	};
}

// JEPA (V-JEPA / V-JEPA 2): a single ViT encoder forward over the video patches
// — no steps, no CFG, no KV. The patch count is large (spatial × temporal), so
// the quadratic attention term is counted alongside the linear 2·params·tokens.
// Reports clips/sec (embeddings) instead of tokens/sec. Reuses the roofline +
// TP/PP parallelism + fills, same as the diffusion branch.
function computeJepa(
	cfg: Config,
	m: ModelSpec,
	g: GpuSpec,
	fmt: WeightFormat,
	fab: FabricSpec
): Profile {
	const js = m.jepa!;
	const res = Math.max(64, cfg.resolution ?? js.defaultResolution);
	const frames = Math.max(js.tubelet, cfg.frames ?? js.defaultFrames);
	const batch = Math.max(1, cfg.batchSize);

	// dense parallelism: TP (intra-node) + PP; no MoE/EP
	const pp = cfg.ppEnabled ? Math.max(1, Math.min(cfg.pp, cfg.numGpus)) : 1;
	const tp = Math.max(1, Math.min(cfg.tp, cfg.numGpus));
	const layerShard = tp * pp;
	const layersPerStage = m.numLayers / pp;

	// video patches per clip: spatial (res/patch)² × temporal (frames/tubelet)
	const spatial = Math.round(res / js.patch) ** 2;
	const temporal = Math.max(1, Math.round(frames / js.tubelet));
	const tokens = Math.max(1, spatial * temporal);
	const effTokens = batch * tokens;

	// ---- memory ----
	const wb = fmt.bitsPerWeight / 8;
	const weights = ((m.params + (js.predictorParams ?? 0)) * wb) / layerShard;
	const perClipAct = (2 * tokens * m.hiddenSize * ACT_BYTES) / layerShard;
	const activations = perClipAct * batch;
	const overhead = OVERHEAD_BYTES;
	const used = weights + activations + overhead;
	const capacity = g.memoryGiB * GiB;
	const fits = used <= capacity;
	const headroom = capacity - used;
	const actBudget = capacity - weights - overhead;
	const maxClips = perClipAct > 0 ? Math.max(0, Math.floor(actBudget / perClipAct)) : 0;

	const segments: Profile['segments'] = [
		{
			key: 'weights',
			label: 'Weights (encoder + predictor)',
			bytes: weights,
			color: COLORS.weights
		},
		{ key: 'activations', label: 'Activations', bytes: activations, color: COLORS.activations },
		{ key: 'overhead', label: 'Overhead', bytes: overhead, color: COLORS.overhead }
	];
	if (fits) segments.push({ key: 'free', label: 'Free', bytes: headroom, color: COLORS.free });
	else
		segments.push({ key: 'overflow', label: 'Overflow', bytes: -headroom, color: COLORS.overflow });

	// ---- roofline for the forward pass(es) ----
	// Encoder: linear (2·params·tokens) + attention (4·layers·tokens²·hidden).
	const linearFlops = 2 * m.params * effTokens;
	const attnFlops = 4 * m.numLayers * tokens * tokens * m.hiddenSize * batch;
	// Predictor: a plain encoder model runs it 0× at inference (embeddings only — the
	// predictor is just resident weights). An action-conditioned world model
	// (V-JEPA 2-AC) rolls it out over the planning horizon.
	const rollout = js.rolloutSteps ?? 0;
	const predH = js.predHidden ?? 384;
	const predL = js.predLayers ?? 12;
	const predLinear = 2 * (js.predictorParams ?? 0) * effTokens;
	const predAttn = 4 * predL * tokens * tokens * predH * batch;
	const predFlops = rollout * (predLinear + predAttn);
	const flopsFwd = (linearFlops + attnFlops + predFlops) / layerShard;
	const peakFlops = computePeak(g, fmt.tier);
	const computeTime = flopsFwd / (peakFlops * MFU * (g.mfuFactor ?? 1) * FAMILY_MFU.jepa);
	const bytesFwd = weights + activations; // weights read once + activations
	const peakBW = g.memBandwidthTBs * TB;
	const memTime = bytesFwd / (peakBW * MBU);
	const stageWork = Math.max(memTime, computeTime);

	// ---- interconnect (TP all-reduce, PP hand-off) ----
	const peakNvlink = g.nvlinkGBs * GB;
	const peakNet = (fab.gbps * GB) / 8;
	const tpSpans = tp > cfg.gpusPerNode;
	const ppSpans = layerShard > cfg.gpusPerNode;
	let tpBytes = 0;
	let tpTime = 0;
	if (tp > 1) {
		tpBytes = layersPerStage * 2 * ((2 * (tp - 1)) / tp) * effTokens * m.hiddenSize * ACT_BYTES;
		tpTime =
			tpBytes /
			((tpSpans ? peakNet : peakNvlink) * (tpSpans ? fab.allReduceEff : NVLINK_ALLREDUCE_EFF));
	}
	let ppBytes = 0;
	let ppTime = 0;
	if (pp > 1) {
		ppBytes = effTokens * m.hiddenSize * ACT_BYTES;
		ppTime = ppBytes / ((ppSpans ? peakNet : peakNvlink) * PP_P2P_EFF);
	}
	const commTime = tpTime;

	const dp = Math.max(1, Math.floor(cfg.numGpus / layerShard));
	const secPerClip = pp * (stageWork + tpTime) + Math.max(0, pp - 1) * ppTime;
	const clipsPerSec = (batch * dp) / secPerClip;

	// ---- fills ----
	const bandwidthUsed = bytesFwd / stageWork;
	const computeAchieved = flopsFwd / stageWork;
	const memBwFrac = clamp01(bandwidthUsed / peakBW);
	const computeFrac = clamp01(computeAchieved / peakFlops);
	const l2Frac = clamp01(Math.max(memBwFrac, computeFrac));
	const nvlinkBytes = (tpSpans ? 0 : tpBytes) + (ppSpans ? 0 : ppBytes);
	const fabricBytes = (tpSpans ? tpBytes : 0) + (ppSpans ? ppBytes : 0);
	const anyIntra = tp > 1 || pp > 1;
	const nvlinkFrac = anyIntra ? clamp01(nvlinkBytes / stageWork / peakNvlink) : 0;
	const netFrac = clamp01(fabricBytes / stageWork / peakNet);

	const heaviest = Math.max(memTime, computeTime, commTime);
	const bottleneck: Profile['bottleneck'] =
		commTime === heaviest && commTime > 0 && fabricBytes > 0
			? 'network'
			: memTime >= computeTime
				? 'memory'
				: 'compute';
	const numNodes = Math.ceil(cfg.numGpus / cfg.gpusPerNode);

	return {
		perGpu: { weights, kv: 0, activations, overhead, used, capacity, fits, headroom },
		segments,
		kvWasteFactor: 1,
		kvPerSeqBytes: 0,
		maxConcurrentSeqs: maxClips,
		phase: 'prefill',
		stepTimeMs: secPerClip * 1000,
		throughputTps: 0,
		perUserTps: 0,
		ttftMs: 0,
		bottleneck,
		memBwFrac,
		computeFrac,
		l2Frac,
		nvlinkFrac,
		netFrac,
		bandwidthUsedGBs: bandwidthUsed / GB,
		bandwidthPeakGBs: peakBW / GB,
		computeAchievedTflops: computeAchieved / TB,
		computePeakTflops: peakFlops / TB,
		nvlinkUsedGBs: nvlinkBytes / stageWork / GB,
		nvlinkPeakGBs: peakNvlink / GB,
		netUsedGBs: fabricBytes / stageWork / GB,
		netPeakGBs: peakNet / GB,
		comm: {
			tpActive: tp > 1,
			tpOverFabric: tpSpans && tp > 1,
			tpGBs: tpBytes / stageWork / GB,
			epActive: false,
			epOverFabric: false,
			epGBs: 0,
			ppActive: pp > 1,
			ppOverFabric: ppSpans && pp > 1,
			ppGBs: ppBytes / stageWork / GB
		},
		fabric: {
			label: fab.label,
			kind: fab.kind,
			peakGBs: peakNet / GB,
			allReduceEff: fab.allReduceEff,
			a2aEff: fab.a2aEff
		},
		cluster: {
			numGpus: cfg.numGpus,
			tp,
			pp,
			ep: 1,
			dp,
			gpusPerNode: cfg.gpusPerNode,
			numNodes,
			crossesFabric: fabricBytes > 0,
			scaleOut: numNodes <= 1 ? 'single' : fabricBytes > 0 ? 'model-across-nodes' : 'data-parallel'
		},
		activeParams: m.params,
		jepa: {
			clipsPerSec,
			secPerClip,
			forwardTimeMs: secPerClip * 1000,
			tokens,
			frames
		}
	};
}

// Encoder-only text models (embeddings, rerankers). A single bidirectional
// forward pass over the input tokens; no KV cache, no autoregression. Reports
// docs/sec (or scores/sec for rerankers). Same shape as computeJepa but with
// text tokens rather than video patches, and no predictor.
function computeEncoder(
	cfg: Config,
	m: ModelSpec,
	g: GpuSpec,
	fmt: WeightFormat,
	fab: FabricSpec
): Profile {
	const es = m.encoder!;
	const tokens = Math.max(1, Math.min(cfg.inputTokens || es.defaultSeqLen, es.maxSeqLen));
	const batch = Math.max(1, cfg.batchSize);

	const pp = cfg.ppEnabled ? Math.max(1, Math.min(cfg.pp, cfg.numGpus)) : 1;
	const tp = Math.max(1, Math.min(cfg.tp, cfg.numGpus));
	const layerShard = tp * pp;
	const layersPerStage = m.numLayers / pp;
	const effTokens = batch * tokens;

	// ---- memory ----
	const wb = fmt.bitsPerWeight / 8;
	const weights = (m.params * wb) / layerShard;
	const perDocAct = (2 * tokens * m.hiddenSize * ACT_BYTES) / layerShard;
	const activations = perDocAct * batch;
	const overhead = OVERHEAD_BYTES;
	const used = weights + activations + overhead;
	const capacity = g.memoryGiB * GiB;
	const fits = used <= capacity;
	const headroom = capacity - used;
	const actBudget = capacity - weights - overhead;
	const maxDocs = perDocAct > 0 ? Math.max(0, Math.floor(actBudget / perDocAct)) : 0;

	const segments: Profile['segments'] = [
		{ key: 'weights', label: 'Weights', bytes: weights, color: COLORS.weights },
		{ key: 'activations', label: 'Activations', bytes: activations, color: COLORS.activations },
		{ key: 'overhead', label: 'Overhead', bytes: overhead, color: COLORS.overhead }
	];
	if (fits) segments.push({ key: 'free', label: 'Free', bytes: headroom, color: COLORS.free });
	else
		segments.push({ key: 'overflow', label: 'Overflow', bytes: -headroom, color: COLORS.overflow });

	// ---- roofline for one forward pass: linear (2·params·tokens) + attention ----
	const linearFlops = 2 * m.params * effTokens;
	const attnFlops = 4 * m.numLayers * tokens * tokens * m.hiddenSize * batch;
	const flopsFwd = (linearFlops + attnFlops) / layerShard;
	const peakFlops = computePeak(g, fmt.tier);
	const computeTime = flopsFwd / (peakFlops * MFU * (g.mfuFactor ?? 1) * FAMILY_MFU.encoder);
	const bytesFwd = weights + activations;
	const peakBW = g.memBandwidthTBs * TB;
	const memTime = bytesFwd / (peakBW * MBU);
	const stageWork = Math.max(memTime, computeTime);

	// ---- interconnect (TP all-reduce, PP hand-off) ----
	const peakNvlink = g.nvlinkGBs * GB;
	const peakNet = (fab.gbps * GB) / 8;
	const tpSpans = tp > cfg.gpusPerNode;
	const ppSpans = layerShard > cfg.gpusPerNode;
	let tpBytes = 0;
	let tpTime = 0;
	if (tp > 1) {
		tpBytes = layersPerStage * 2 * ((2 * (tp - 1)) / tp) * effTokens * m.hiddenSize * ACT_BYTES;
		tpTime =
			tpBytes /
			((tpSpans ? peakNet : peakNvlink) * (tpSpans ? fab.allReduceEff : NVLINK_ALLREDUCE_EFF));
	}
	let ppBytes = 0;
	let ppTime = 0;
	if (pp > 1) {
		ppBytes = effTokens * m.hiddenSize * ACT_BYTES;
		ppTime = ppBytes / ((ppSpans ? peakNet : peakNvlink) * PP_P2P_EFF);
	}
	const commTime = tpTime;

	const dp = Math.max(1, Math.floor(cfg.numGpus / layerShard));
	const secPerDoc = pp * (stageWork + tpTime) + Math.max(0, pp - 1) * ppTime;
	const docsPerSec = (batch * dp) / secPerDoc;

	// ---- fills ----
	const bandwidthUsed = bytesFwd / stageWork;
	const computeAchieved = flopsFwd / stageWork;
	const memBwFrac = clamp01(bandwidthUsed / peakBW);
	const computeFrac = clamp01(computeAchieved / peakFlops);
	const l2Frac = clamp01(Math.max(memBwFrac, computeFrac));
	const nvlinkBytes = (tpSpans ? 0 : tpBytes) + (ppSpans ? 0 : ppBytes);
	const fabricBytes = (tpSpans ? tpBytes : 0) + (ppSpans ? ppBytes : 0);
	const anyIntra = tp > 1 || pp > 1;
	const nvlinkFrac = anyIntra ? clamp01(nvlinkBytes / stageWork / peakNvlink) : 0;
	const netFrac = clamp01(fabricBytes / stageWork / peakNet);

	const heaviest = Math.max(memTime, computeTime, commTime);
	const bottleneck: Profile['bottleneck'] =
		commTime === heaviest && commTime > 0 && fabricBytes > 0
			? 'network'
			: memTime >= computeTime
				? 'memory'
				: 'compute';
	const numNodes = Math.ceil(cfg.numGpus / cfg.gpusPerNode);

	return {
		perGpu: { weights, kv: 0, activations, overhead, used, capacity, fits, headroom },
		segments,
		kvWasteFactor: 1,
		kvPerSeqBytes: 0,
		maxConcurrentSeqs: maxDocs,
		phase: 'prefill',
		stepTimeMs: secPerDoc * 1000,
		throughputTps: (effTokens * dp) / secPerDoc, // tokens/sec processed across the cluster
		perUserTps: 0,
		ttftMs: 0,
		bottleneck,
		memBwFrac,
		computeFrac,
		l2Frac,
		nvlinkFrac,
		netFrac,
		bandwidthUsedGBs: bandwidthUsed / GB,
		bandwidthPeakGBs: peakBW / GB,
		computeAchievedTflops: computeAchieved / TB,
		computePeakTflops: peakFlops / TB,
		nvlinkUsedGBs: nvlinkBytes / stageWork / GB,
		nvlinkPeakGBs: peakNvlink / GB,
		netUsedGBs: fabricBytes / stageWork / GB,
		netPeakGBs: peakNet / GB,
		comm: {
			tpActive: tp > 1,
			tpOverFabric: tpSpans && tp > 1,
			tpGBs: tpBytes / stageWork / GB,
			epActive: false,
			epOverFabric: false,
			epGBs: 0,
			ppActive: pp > 1,
			ppOverFabric: ppSpans && pp > 1,
			ppGBs: ppBytes / stageWork / GB
		},
		fabric: {
			label: fab.label,
			kind: fab.kind,
			peakGBs: peakNet / GB,
			allReduceEff: fab.allReduceEff,
			a2aEff: fab.a2aEff
		},
		cluster: {
			numGpus: cfg.numGpus,
			tp,
			pp,
			ep: 1,
			dp,
			gpusPerNode: cfg.gpusPerNode,
			numNodes,
			crossesFabric: fabricBytes > 0,
			scaleOut: numNodes <= 1 ? 'single' : fabricBytes > 0 ? 'model-across-nodes' : 'data-parallel'
		},
		activeParams: m.params,
		encoder: { docsPerSec, secPerDoc, forwardTimeMs: secPerDoc * 1000, tokens }
	};
}

// Vision-Language-Action (VLA): a small VLM whose "output" is a chunk of
// continuous actions produced via flow matching. One inference cycle:
//   1. vision encode: cameras × tokensPerImage patches (frozen encoder)
//   2. prefill: language + state + image tokens through the backbone
//   3. action rollout: `flowSteps` passes through an action expert over
//      `chunkSize` action tokens
// Reports controls/sec (one chunk = one inference call) plus the closed-loop
// control rate you can serve (chunk × chunksPerSec) and total actions/sec.
function computeVla(
	cfg: Config,
	m: ModelSpec,
	g: GpuSpec,
	fmt: WeightFormat,
	fab: FabricSpec
): Profile {
	const vs = m.vla!;
	const batch = Math.max(1, cfg.batchSize);
	const pp = cfg.ppEnabled ? Math.max(1, Math.min(cfg.pp, cfg.numGpus)) : 1;
	const tp = Math.max(1, Math.min(cfg.tp, cfg.numGpus));
	const layerShard = tp * pp;

	const imgTokens = vs.camerasPerObs * vs.tokensPerImage;
	const stateTokens = 1; // proprioceptive state is one embedded token
	const textTokens = 48; // small language prompt, per configs
	const inputTokens = imgTokens + stateTokens + textTokens;
	const effTokens = batch * inputTokens;

	// ---- memory ----
	const wb = fmt.bitsPerWeight / 8;
	const backboneW = (m.params * wb) / layerShard;
	const expertW = (vs.expertParams * wb) / layerShard;
	const visionW = (vs.visionParams * wb) / tp; // frozen SigLIP; not sharded by pp
	const weights = backboneW + expertW + visionW;
	const perObsAct = (2 * inputTokens * m.hiddenSize * ACT_BYTES) / layerShard;
	const activations = perObsAct * batch;
	const overhead = OVERHEAD_BYTES;
	const used = weights + activations + overhead;
	const capacity = g.memoryGiB * GiB;
	const fits = used <= capacity;
	const headroom = capacity - used;
	const actBudget = capacity - weights - overhead;
	const maxObs = perObsAct > 0 ? Math.max(0, Math.floor(actBudget / perObsAct)) : 0;

	const segments: Profile['segments'] = [
		{
			key: 'weights',
			label: 'Weights (backbone + expert + vision)',
			bytes: weights,
			color: COLORS.weights
		},
		{ key: 'activations', label: 'Activations', bytes: activations, color: COLORS.activations },
		{ key: 'overhead', label: 'Overhead', bytes: overhead, color: COLORS.overhead }
	];
	if (fits) segments.push({ key: 'free', label: 'Free', bytes: headroom, color: COLORS.free });
	else
		segments.push({ key: 'overflow', label: 'Overflow', bytes: -headroom, color: COLORS.overflow });

	// ---- FLOPs per observation ----
	// Backbone prefill (linear + attention over the vision+state+text tokens).
	const backboneLinear = 2 * m.params * effTokens;
	const backboneAttn = 4 * m.numLayers * inputTokens * inputTokens * m.hiddenSize * batch;
	// Vision encoder forward: 2 · visionParams · (image patch tokens across cameras).
	const visionForward = 2 * vs.visionParams * imgTokens * batch;
	// Action expert: `flowSteps` sampling passes over chunkSize action tokens.
	const expertForward =
		vs.flowSteps *
		(2 * vs.expertParams * vs.chunkSize * batch +
			4 * m.numLayers * vs.chunkSize * vs.chunkSize * m.hiddenSize * batch);
	const flops = (backboneLinear + backboneAttn + visionForward + expertForward) / layerShard;

	const peakFlops = computePeak(g, fmt.tier);
	const computeTime = flops / (peakFlops * MFU * (g.mfuFactor ?? 1));
	const bytesFwd = weights + activations;
	const peakBW = g.memBandwidthTBs * TB;
	const memTime = bytesFwd / (peakBW * MBU);
	const stageWork = Math.max(memTime, computeTime);

	// ---- interconnect (TP all-reduce, PP hand-off) ----
	const peakNvlink = g.nvlinkGBs * GB;
	const peakNet = (fab.gbps * GB) / 8;
	const tpSpans = tp > cfg.gpusPerNode;
	const ppSpans = layerShard > cfg.gpusPerNode;
	let tpBytes = 0,
		tpTime = 0;
	if (tp > 1) {
		tpBytes = (m.numLayers / pp) * 2 * ((2 * (tp - 1)) / tp) * effTokens * m.hiddenSize * ACT_BYTES;
		tpTime =
			tpBytes /
			((tpSpans ? peakNet : peakNvlink) * (tpSpans ? fab.allReduceEff : NVLINK_ALLREDUCE_EFF));
	}
	let ppBytes = 0,
		ppTime = 0;
	if (pp > 1) {
		ppBytes = effTokens * m.hiddenSize * ACT_BYTES;
		ppTime = ppBytes / ((ppSpans ? peakNet : peakNvlink) * PP_P2P_EFF);
	}

	const dp = Math.max(1, Math.floor(cfg.numGpus / layerShard));
	const secPerControl = pp * (stageWork + tpTime) + Math.max(0, pp - 1) * ppTime;
	const controlsPerSec = (batch * dp) / secPerControl; // chunks per second cluster-wide
	const actionsPerSec = controlsPerSec * vs.chunkSize;
	const effectiveHz = vs.chunkSize / secPerControl; // per replica: closed-loop control frequency it can sustain

	const memBwFrac = clamp01(bytesFwd / stageWork / peakBW);
	const computeFrac = clamp01(flops / stageWork / peakFlops);
	const l2Frac = clamp01(Math.max(memBwFrac, computeFrac));
	const nvlinkBytes = (tpSpans ? 0 : tpBytes) + (ppSpans ? 0 : ppBytes);
	const fabricBytes = (tpSpans ? tpBytes : 0) + (ppSpans ? ppBytes : 0);
	const nvlinkFrac = tp > 1 || pp > 1 ? clamp01(nvlinkBytes / stageWork / peakNvlink) : 0;
	const netFrac = clamp01(fabricBytes / stageWork / peakNet);
	const commTime = tpTime;
	const heaviest = Math.max(memTime, computeTime, commTime);
	const bottleneck: Profile['bottleneck'] =
		commTime === heaviest && commTime > 0 && fabricBytes > 0
			? 'network'
			: memTime >= computeTime
				? 'memory'
				: 'compute';
	const numNodes = Math.ceil(cfg.numGpus / cfg.gpusPerNode);

	return {
		perGpu: { weights, kv: 0, activations, overhead, used, capacity, fits, headroom },
		segments,
		kvWasteFactor: 1,
		kvPerSeqBytes: 0,
		maxConcurrentSeqs: maxObs,
		phase: 'prefill',
		stepTimeMs: secPerControl * 1000,
		throughputTps: actionsPerSec,
		perUserTps: 0,
		ttftMs: 0,
		bottleneck,
		memBwFrac,
		computeFrac,
		l2Frac,
		nvlinkFrac,
		netFrac,
		bandwidthUsedGBs: bytesFwd / stageWork / GB,
		bandwidthPeakGBs: peakBW / GB,
		computeAchievedTflops: flops / stageWork / TB,
		computePeakTflops: peakFlops / TB,
		nvlinkUsedGBs: nvlinkBytes / stageWork / GB,
		nvlinkPeakGBs: peakNvlink / GB,
		netUsedGBs: fabricBytes / stageWork / GB,
		netPeakGBs: peakNet / GB,
		comm: {
			tpActive: tp > 1,
			tpOverFabric: tpSpans && tp > 1,
			tpGBs: tpBytes / stageWork / GB,
			epActive: false,
			epOverFabric: false,
			epGBs: 0,
			ppActive: pp > 1,
			ppOverFabric: ppSpans && pp > 1,
			ppGBs: ppBytes / stageWork / GB
		},
		fabric: {
			label: fab.label,
			kind: fab.kind,
			peakGBs: peakNet / GB,
			allReduceEff: fab.allReduceEff,
			a2aEff: fab.a2aEff
		},
		cluster: {
			numGpus: cfg.numGpus,
			tp,
			pp,
			ep: 1,
			dp,
			gpusPerNode: cfg.gpusPerNode,
			numNodes,
			crossesFabric: fabricBytes > 0,
			scaleOut: numNodes <= 1 ? 'single' : fabricBytes > 0 ? 'model-across-nodes' : 'data-parallel'
		},
		activeParams: m.params,
		vla: { controlsPerSec, secPerControl, effectiveHz, actionsPerSec }
	};
}

// ASR (Whisper-family): encoder-decoder. One inference cycle = one encoder pass
// over a fixed audio window (audioTokens patches) + autoregressive decode of
// avgTextTokens output tokens through the decoder with cross-attention to the
// encoded audio. Reports RTF (real-time factor), audio-sec/sec throughput, and
// per-window latency broken into encoder vs decoder time.
function computeAsr(
	cfg: Config,
	m: ModelSpec,
	g: GpuSpec,
	fmt: WeightFormat,
	fab: FabricSpec
): Profile {
	const as = m.asr!;
	const batch = Math.max(1, cfg.batchSize);
	const pp = cfg.ppEnabled ? Math.max(1, Math.min(cfg.pp, cfg.numGpus)) : 1;
	const tp = Math.max(1, Math.min(cfg.tp, cfg.numGpus));
	const layerShard = tp * pp;

	// ---- memory ----
	const wb = fmt.bitsPerWeight / 8;
	const weights = ((as.encoderParams + as.decoderParams) * wb) / layerShard;
	// KV cache for the decoder (self-attn on generated tokens + cross-attn on audio tokens).
	// Self KV: numLayersDecoder × 2 × avgTextTokens × numHeads × headDim per sequence.
	// Cross KV: numLayersDecoder × 2 × audioTokens × numHeads × headDim per sequence.
	const kvBytesPerSeq =
		as.numLayersDecoder *
		2 *
		(as.avgTextTokens + as.audioTokens) *
		as.numHeadsDecoder *
		as.headDimDecoder *
		(cfg.kvBits / 8);
	const kv = (kvBytesPerSeq * batch) / layerShard;
	const perWindowAct = (2 * as.audioTokens * m.hiddenSize * ACT_BYTES) / layerShard;
	const activations = perWindowAct * batch;
	const overhead = OVERHEAD_BYTES;
	const used = weights + kv + activations + overhead;
	const capacity = g.memoryGiB * GiB;
	const fits = used <= capacity;
	const headroom = capacity - used;
	const kvBudget = capacity - weights - activations - overhead;
	const maxSeqs = kvBytesPerSeq > 0 ? Math.max(0, Math.floor(kvBudget / kvBytesPerSeq)) : 0;

	const segments: Profile['segments'] = [
		{ key: 'weights', label: 'Weights (encoder + decoder)', bytes: weights, color: COLORS.weights },
		{ key: 'kv', label: 'KV cache (self + cross)', bytes: kv, color: COLORS.kv },
		{ key: 'activations', label: 'Activations', bytes: activations, color: COLORS.activations },
		{ key: 'overhead', label: 'Overhead', bytes: overhead, color: COLORS.overhead }
	];
	if (fits) segments.push({ key: 'free', label: 'Free', bytes: headroom, color: COLORS.free });
	else
		segments.push({ key: 'overflow', label: 'Overflow', bytes: -headroom, color: COLORS.overflow });

	const peakFlops = computePeak(g, fmt.tier);
	const peakBW = g.memBandwidthTBs * TB;

	// Encoder: single pass over audio tokens. Compute-bound on the audio window.
	const encFlops =
		(2 * as.encoderParams * as.audioTokens * batch +
			4 * (m.numLayers / 2) * as.audioTokens * as.audioTokens * m.hiddenSize * batch) /
		layerShard;
	const encCompute = encFlops / (peakFlops * MFU * (g.mfuFactor ?? 1));
	const encMem = (weights + activations) / (peakBW * MBU);
	const encTime = Math.max(encCompute, encMem);

	// Decoder: autoregressive over avgTextTokens with cross-attention to audio.
	// Per-token work: read decoder weights + read audio KV context. Memory-bound.
	const decoderWeightBytes = (as.decoderParams * wb) / layerShard;
	const perTokenMem = decoderWeightBytes + kv / batch; // per-sequence per-token bytes
	const perTokenCompute =
		(2 * as.decoderParams * batch) / (peakFlops * MFU * (g.mfuFactor ?? 1) * layerShard);
	const perTokenTime = Math.max(perTokenMem / (peakBW * MBU), perTokenCompute);
	const decTime = as.avgTextTokens * perTokenTime;

	// ---- interconnect (TP all-reduce during encoder + decoder) ----
	const peakNvlink = g.nvlinkGBs * GB;
	const peakNet = (fab.gbps * GB) / 8;
	const tpSpans = tp > cfg.gpusPerNode;
	let tpBytesEnc = 0,
		tpTimeEnc = 0,
		tpBytesDec = 0,
		tpTimeDec = 0;
	if (tp > 1) {
		tpBytesEnc = as.audioTokens * batch * m.hiddenSize * ACT_BYTES * 2 * ((2 * (tp - 1)) / tp);
		tpTimeEnc =
			tpBytesEnc /
			((tpSpans ? peakNet : peakNvlink) * (tpSpans ? fab.allReduceEff : NVLINK_ALLREDUCE_EFF));
		// decode: per-token collective is tiny but happens for every generated token
		tpBytesDec = as.avgTextTokens * batch * m.hiddenSize * ACT_BYTES * 2 * ((2 * (tp - 1)) / tp);
		tpTimeDec =
			tpBytesDec /
			((tpSpans ? peakNet : peakNvlink) * (tpSpans ? fab.allReduceEff : NVLINK_ALLREDUCE_EFF));
	}

	const dp = Math.max(1, Math.floor(cfg.numGpus / layerShard));
	const secPerWindow = encTime + tpTimeEnc + decTime + tpTimeDec;
	const rtf = as.audioWindowSec / secPerWindow; // per replica (real-time factor)
	const windowsPerSec = (batch * dp) / secPerWindow;
	const audioSecPerSec = windowsPerSec * as.audioWindowSec;

	const bandwidthUsed = (weights + activations + kv) / secPerWindow;
	const memBwFrac = clamp01(bandwidthUsed / peakBW);
	const totalFlops = encFlops + (2 * as.decoderParams * as.avgTextTokens * batch) / layerShard;
	const computeAchieved = totalFlops / secPerWindow;
	const computeFrac = clamp01(computeAchieved / peakFlops);
	const l2Frac = clamp01(Math.max(memBwFrac, computeFrac));
	const bottleneck: Profile['bottleneck'] =
		encCompute + decTime >= encMem + decTime ? 'compute' : 'memory';
	const numNodes = Math.ceil(cfg.numGpus / cfg.gpusPerNode);

	return {
		perGpu: { weights, kv, activations, overhead, used, capacity, fits, headroom },
		segments,
		kvWasteFactor: 1,
		kvPerSeqBytes: kvBytesPerSeq,
		maxConcurrentSeqs: maxSeqs,
		phase: 'decode',
		stepTimeMs: secPerWindow * 1000,
		throughputTps: (as.avgTextTokens * batch * dp) / secPerWindow,
		perUserTps: as.avgTextTokens / decTime,
		ttftMs: (encTime + tpTimeEnc) * 1000, // audio decoded to first text token
		bottleneck,
		memBwFrac,
		computeFrac,
		l2Frac,
		nvlinkFrac:
			tp > 1 && !tpSpans ? clamp01((tpBytesEnc + tpBytesDec) / secPerWindow / peakNvlink) : 0,
		netFrac: tp > 1 && tpSpans ? clamp01((tpBytesEnc + tpBytesDec) / secPerWindow / peakNet) : 0,
		bandwidthUsedGBs: bandwidthUsed / GB,
		bandwidthPeakGBs: peakBW / GB,
		computeAchievedTflops: computeAchieved / TB,
		computePeakTflops: peakFlops / TB,
		nvlinkUsedGBs: tp > 1 && !tpSpans ? (tpBytesEnc + tpBytesDec) / secPerWindow / GB : 0,
		nvlinkPeakGBs: peakNvlink / GB,
		netUsedGBs: tp > 1 && tpSpans ? (tpBytesEnc + tpBytesDec) / secPerWindow / GB : 0,
		netPeakGBs: peakNet / GB,
		comm: {
			tpActive: tp > 1,
			tpOverFabric: tpSpans && tp > 1,
			tpGBs: tp > 1 ? (tpBytesEnc + tpBytesDec) / secPerWindow / GB : 0,
			epActive: false,
			epOverFabric: false,
			epGBs: 0,
			ppActive: false,
			ppOverFabric: false,
			ppGBs: 0
		},
		fabric: {
			label: fab.label,
			kind: fab.kind,
			peakGBs: peakNet / GB,
			allReduceEff: fab.allReduceEff,
			a2aEff: fab.a2aEff
		},
		cluster: {
			numGpus: cfg.numGpus,
			tp,
			pp: 1,
			ep: 1,
			dp,
			gpusPerNode: cfg.gpusPerNode,
			numNodes,
			crossesFabric: tpSpans && tp > 1,
			scaleOut:
				numNodes <= 1 ? 'single' : tpSpans && tp > 1 ? 'model-across-nodes' : 'data-parallel'
		},
		activeParams: m.params,
		asr: {
			secPerWindow,
			audioSecPerSec,
			rtf,
			windowsPerSec,
			encoderMs: (encTime + tpTimeEnc) * 1000,
			decoderMs: (decTime + tpTimeDec) * 1000
		}
	};
}

export function computeProfile(cfg: Config): Profile {
	const m = MODELS_BY_ID.get(cfg.modelId)!;
	const g = GPUS_BY_ID.get(cfg.gpuId)!;
	const fab = FABRICS_BY_ID.get(cfg.fabricId) ?? FABRICS_BY_ID.get('ib-ndr')!;
	const fmt = WEIGHT_FORMATS_BY_ID.get(cfg.weightFormatId) ?? WEIGHT_FORMATS_BY_ID.get('bf16')!;

	if (m.kind === 'diffusion' && m.diffusion) return computeDiffusion(cfg, m, g, fmt, fab);
	if (m.kind === 'jepa' && m.jepa) return computeJepa(cfg, m, g, fmt, fab);
	if (m.kind === 'encoder' && m.encoder) return computeEncoder(cfg, m, g, fmt, fab);
	if (m.kind === 'vla' && m.vla) return computeVla(cfg, m, g, fmt, fab);
	if (m.kind === 'asr' && m.asr) return computeAsr(cfg, m, g, fmt, fab);

	// Clamp the config to something physical.
	const pp = cfg.ppEnabled ? Math.max(1, Math.min(cfg.pp, cfg.numGpus)) : 1;
	// EP overlays the tensor + data dimensions (not pipeline): experts spread
	// across everything except the pipeline stages.
	const epBudget = Math.max(1, Math.floor(cfg.numGpus / pp));
	// For a VLM, N images each contribute tokensPerImage visual patches that the
	// LLM reads during prefill. Fold them into inputTokens.
	const imageTokens =
		m.vlm && (cfg.imagesPerRequest ?? 0) > 0
			? (cfg.imagesPerRequest ?? 0) * m.vlm.tokensPerImage
			: 0;
	const c: Config = {
		...cfg,
		tp: Math.max(1, Math.min(cfg.tp, cfg.numGpus)),
		pp,
		ep: m.moe && cfg.epEnabled ? Math.max(1, Math.min(cfg.ep, epBudget)) : 1,
		gpusPerNode: Math.max(1, Math.min(cfg.gpusPerNode, cfg.numGpus)),
		batchSize: Math.max(1, cfg.batchSize),
		inputTokens: Math.max(1, cfg.inputTokens + imageTokens),
		outputTokens: Math.max(1, cfg.outputTokens)
	};
	const epActive = !!m.moe && c.ep > 1;
	const ppActive = c.pp > 1;

	// KV cache grows to input + output; prefill only processes the input tokens.
	// Prefix caching reuses the KV of a shared prefix, so prefill computes only the
	// uncached suffix (KV memory is unchanged — the prefix KV is still resident).
	const cachedFrac = Math.max(0, Math.min(1, cfg.cachedPrefixFrac ?? 0));
	const prefillTokens = Math.max(1, Math.round(c.inputTokens * (1 - cachedFrac)));
	const seqLen = c.inputTokens + c.outputTokens;
	const tokensPerSeq = cfg.phase === 'prefill' ? prefillTokens : 1;
	const tokensInStep = c.batchSize * tokensPerSeq;

	// tp and pp both consume GPUs within one model replica.
	const modelParallel = c.tp * c.pp;
	const layerShard = modelParallel; // non-expert weights/kv split across tp*pp
	// One replica occupies pp pipeline stages, each `max(tp, ep)` wide: attention
	// shards tp ways and experts ep ways over the same non-pipeline GPUs, so the
	// larger dimension sets the width. Data-parallel replicas tile numGpus by this
	// — using tp*pp alone would over-count replicas (inflate throughput) when
	// ep > tp.
	const replicaGpus = c.pp * Math.max(c.tp, c.ep);
	const layersPerStage = m.numLayers / c.pp;

	// ---- memory (per GPU) ----
	const wb = fmt.bitsPerWeight / 8;
	let weights: number;
	if (!m.moe) {
		weights = (m.params * wb) / layerShard;
	} else {
		// non-expert weights split by tp*pp; experts split by ep*pp.
		const { nonExpert, expert } = moeSplit(m);
		weights = (nonExpert * wb) / layerShard + (expert * wb) / (c.ep * c.pp);
	}
	// Vision encoder for a VLM is held resident; typically small compared to the LLM.
	if (m.vlm) weights += (m.vlm.visionParams * wb) / c.tp;
	const kvb = c.kvBits / 8;
	// Bytes of KV per token per layer. MLA keeps one small compressed latent
	// (no ×2, no per-head); standard attention keeps K and V per KV head (GQA).
	const kvPerTokenLayer = m.mla ? m.mla.kvLatentDim * kvb : 2 * m.numKvHeads * m.headDim * kvb;
	// Hybrid models (e.g. linear + periodic full attention) only cache KV on the
	// full-attention layers.
	const kvLayers = m.kvLayers ?? m.numLayers;
	// MLA's latent is replicated across TP; standard KV shards across TP but no
	// finer than the KV-head count (can't split 1 KV head across 8 ranks).
	const kvShard = (m.mla ? 1 : Math.min(c.tp, m.numKvHeads)) * c.pp;
	// Allocation waste vs the ideal tight packing:
	//  - paged (vLLM/TGI): round each sequence up to a KV block + small pool overhead
	//  - contiguous (naive): pre-reserve and fragment
	const kvWasteFactor =
		c.kvAllocation === 'paged' ? ((Math.ceil(seqLen / KV_BLOCK) * KV_BLOCK) / seqLen) * 1.02 : 1.6;
	const kvPerSeqBytes = ((kvLayers * seqLen * kvPerTokenLayer) / kvShard) * kvWasteFactor;
	const kv = kvPerSeqBytes * c.batchSize;
	// Hybrid Mamba: a fixed-size SSM recurrent state per sequence, independent of
	// context length (unlike KV). Its heads shard across TP and its layers split
	// across PP, so it divides by the model-parallel width.
	const mambaPerSeq = m.mamba ? m.mamba.layers * m.mamba.stateBytesPerLayer : 0;
	const mambaState = (mambaPerSeq * c.batchSize) / modelParallel;
	const activations = (2 * c.batchSize * seqLen * m.hiddenSize * ACT_BYTES) / layerShard;
	const overhead = OVERHEAD_BYTES;
	const used = weights + kv + mambaState + activations + overhead;
	const capacity = g.memoryGiB * GiB;
	const fits = used <= capacity;
	const headroom = capacity - used;

	// How many sequences the leftover HBM holds. Each sequence costs its KV
	// (grows with context) plus its Mamba state (fixed), both sharded.
	const kvBudget = capacity - weights - activations - overhead;
	const perSeqBytes = kvPerSeqBytes + mambaPerSeq / modelParallel;
	const maxConcurrentSeqs = perSeqBytes > 0 ? Math.max(0, Math.floor(kvBudget / perSeqBytes)) : 0;

	const segments: Profile['segments'] = [
		{ key: 'weights', label: 'Weights', bytes: weights, color: COLORS.weights },
		{ key: 'kv', label: 'KV cache', bytes: kv, color: COLORS.kv },
		...(mambaState > 0
			? [{ key: 'mamba' as const, label: 'Mamba state', bytes: mambaState, color: COLORS.mamba }]
			: []),
		{ key: 'activations', label: 'Activations', bytes: activations, color: COLORS.activations },
		{ key: 'overhead', label: 'Overhead', bytes: overhead, color: COLORS.overhead }
	];
	if (fits) segments.push({ key: 'free', label: 'Free', bytes: headroom, color: COLORS.free });
	else
		segments.push({ key: 'overflow', label: 'Overflow', bytes: -headroom, color: COLORS.overflow });

	// ---- per-stage compute / memory (roofline) ----
	const act = activeParams(m);
	// Weights are read once per step and reused across every token in the step,
	// so the read is independent of how many tokens the step processes. The Mamba
	// state is both read and rewritten every decode step (×2 traffic) even though
	// its footprint does not grow with context. Checked against a real hybrid-model
	// decode: the ×2 term brings the estimate to within ~12% (the roofline is an
	// upper bound).
	const bytesPerStep = weights + kv + 2 * mambaState;
	const peakBW = g.memBandwidthTBs * TB;
	const memTime = bytesPerStep / (peakBW * MBU);

	// FLOPs scale with tokens. Attention is TP-sharded; experts are EP-sharded.
	const flopsPerStep =
		2 * tokensInStep * (nonExpertParams(m) / (c.tp * c.pp) + activeExpertParams(m) / (c.ep * c.pp));
	const peakFlops = computePeak(g, fmt.tier);
	const computeTime = flopsPerStep / (peakFlops * MFU * (g.mfuFactor ?? 1));
	const stageWork = Math.max(memTime, computeTime);

	// ---- interconnect ----
	const peakNvlink = g.nvlinkGBs * GB;
	const peakNet = (fab.gbps * GB) / 8; // Gb/s -> GB/s

	const link = (spans: boolean) => (spans ? peakNet : peakNvlink);
	const arEff = (spans: boolean) => (spans ? fab.allReduceEff : NVLINK_ALLREDUCE_EFF);
	const a2aEff = (spans: boolean) => (spans ? fab.a2aEff : NVLINK_A2A_EFF);
	// α-β latency model: collective time ≈ α·log2(N) + β·bytes, where β = 1/(bw·eff).
	// α_intra (NVLink/PCIe): a few μs; α_fabric: fab.alphaUs. Per-layer overhead
	// scales with the layers-that-collective count (all-reduce per attention layer,
	// all-to-all per MoE layer).
	const NVLINK_ALPHA_US = 2; // NVLink hop latency, μs
	const alphaFor = (spans: boolean) => (spans ? fab.alphaUs : NVLINK_ALPHA_US) * 1e-6;

	// TP all-reduce: 2 per layer, ~2(tp-1)/tp volume of a hidden vector per token.
	const tpSpans = c.tp > c.gpusPerNode;
	const tp: Collective = { bytes: 0, overFabric: tpSpans, time: 0 };
	if (c.tp > 1) {
		tp.bytes =
			layersPerStage * 2 * ((2 * (c.tp - 1)) / c.tp) * tokensInStep * m.hiddenSize * ACT_BYTES;
		// α per collective call; 2 all-reduces per layer × layersPerStage
		const tpAlpha = 2 * layersPerStage * alphaFor(tpSpans) * Math.log2(Math.max(2, c.tp));
		tp.time = tp.bytes / (link(tpSpans) * arEff(tpSpans)) + tpAlpha;
	}

	// EP all-to-all: dispatch + combine per MoE layer. Each token's activation is
	// routed to its top-k experts spread over the ep ranks; the (ep-1)/ep fraction
	// leaves this GPU. Two passes (dispatch, then combine).
	const epSpans = c.ep > c.gpusPerNode;
	const ep: Collective = { bytes: 0, overFabric: epSpans, time: 0 };
	if (epActive) {
		const topk = m.moe!.expertsPerToken;
		ep.bytes =
			layersPerStage * 2 * ((c.ep - 1) / c.ep) * tokensInStep * topk * m.hiddenSize * ACT_BYTES;
		const epAlpha = 2 * layersPerStage * alphaFor(epSpans) * Math.log2(Math.max(2, c.ep));
		ep.time = ep.bytes / (link(epSpans) * a2aEff(epSpans)) + epAlpha;
	}

	// PP point-to-point: one activation hand-off to the next stage per step.
	const ppSpans = replicaGpus > c.gpusPerNode;
	const ppLink: Collective = { bytes: 0, overFabric: ppSpans, time: 0 };
	if (ppActive) {
		ppLink.bytes = tokensInStep * m.hiddenSize * ACT_BYTES;
		ppLink.time = ppLink.bytes / (link(ppSpans) * PP_P2P_EFF) + alphaFor(ppSpans);
	}

	// A stage overlaps its TP + EP collectives with the next microbatch, so its
	// throughput is bounded by work + in-stage collectives. PP hand-off adds to
	// end-to-end latency, not stage throughput.
	const stageTime = stageWork + tp.time + ep.time;

	const dp = Math.max(1, Math.floor(c.numGpus / replicaGpus));
	let perUserTps = tokensPerSeq / (c.pp * stageTime + Math.max(0, c.pp - 1) * ppLink.time);
	let throughputTps = (tokensInStep / stageTime) * dp;

	// ---- speculative decoding (decode phase only) ----
	// A small draft proposes γ tokens; the target verifies them in ONE forward.
	// Expected accepted per cycle (geometric acceptance α, +1 target token):
	//   n = (1 - α^(γ+1)) / (1 - α).
	// Cycle cost ≈ one target forward + γ draft forwards (draft ≈ DRAFT_COST_RATIO
	// of a target step). Decode is memory-bound, so verifying γ+1 tokens is ~free.
	let specSpeedup = 1;
	if (cfg.phase === 'decode' && cfg.specDecode) {
		const alpha = Math.min(0.99, Math.max(0.01, cfg.draftAcceptRate ?? 0.7));
		const gamma = Math.max(1, Math.round(cfg.specTokens ?? 4));
		const tokensPerCycle = (1 - alpha ** (gamma + 1)) / (1 - alpha);
		specSpeedup = tokensPerCycle / (1 + gamma * DRAFT_COST_RATIO);
		perUserTps *= specSpeedup;
		throughputTps *= specSpeedup;
	}

	// ---- fills: achieved vs peak ----
	const bandwidthUsed = bytesPerStep / stageTime;
	const computeAchieved = flopsPerStep / stageTime;
	const memBwFrac = clamp01(bandwidthUsed / peakBW);
	const computeFrac = clamp01(computeAchieved / peakFlops);
	// HBM controllers track actual HBM traffic; L2 is busy whenever it feeds
	// either HBM streaming (decode) or the tensor cores (prefill).
	const l2Frac = clamp01(Math.max(memBwFrac, computeFrac));

	const collectives = [tp, ep, ppLink];
	const nvlinkBytes = collectives.filter((x) => !x.overFabric).reduce((s, x) => s + x.bytes, 0);
	const fabricBytes = collectives.filter((x) => x.overFabric).reduce((s, x) => s + x.bytes, 0);
	const anyIntra = c.tp > 1 || epActive || ppActive;
	const nvlinkFrac = anyIntra ? clamp01(nvlinkBytes / stageTime / peakNvlink) : 0;
	const netFrac = clamp01(fabricBytes / stageTime / peakNet);

	// ---- bottleneck ----
	const commTime = tp.time + ep.time;
	const heaviest = Math.max(memTime, computeTime, commTime);
	let bottleneck: Profile['bottleneck'];
	if (commTime === heaviest && commTime > 0 && fabricBytes > 0) bottleneck = 'network';
	else bottleneck = memTime >= computeTime ? 'memory' : 'compute';

	const numNodes = Math.ceil(c.numGpus / c.gpusPerNode);
	const crossesFabric = fabricBytes > 0;
	const scaleOut: Profile['cluster']['scaleOut'] =
		numNodes <= 1 ? 'single' : crossesFabric ? 'model-across-nodes' : 'data-parallel';

	// ---- TTFT: prefill of the uncached INPUT tokens (compute bound, phase-independent) ----
	const prefillFlops =
		2 *
		c.batchSize *
		prefillTokens *
		(nonExpertParams(m) / (c.tp * c.pp) + activeExpertParams(m) / (c.ep * c.pp));
	const ttftMs = (prefillFlops / (peakFlops * MFU * (g.mfuFactor ?? 1))) * 1000;

	return {
		perGpu: { weights, kv, activations, overhead, used, capacity, fits, headroom },
		segments,
		kvWasteFactor,
		kvPerSeqBytes,
		maxConcurrentSeqs,
		phase: cfg.phase,
		stepTimeMs: stageTime * 1000,
		throughputTps,
		perUserTps,
		ttftMs,
		specSpeedup,
		bottleneck,
		memBwFrac,
		computeFrac,
		l2Frac,
		nvlinkFrac,
		netFrac,
		bandwidthUsedGBs: bandwidthUsed / GB,
		bandwidthPeakGBs: peakBW / GB,
		computeAchievedTflops: computeAchieved / TB,
		computePeakTflops: peakFlops / TB,
		nvlinkUsedGBs: nvlinkBytes / stageTime / GB,
		nvlinkPeakGBs: peakNvlink / GB,
		netUsedGBs: fabricBytes / stageTime / GB,
		netPeakGBs: peakNet / GB,
		comm: {
			tpActive: c.tp > 1,
			tpOverFabric: tpSpans && c.tp > 1,
			tpGBs: tp.bytes / stageTime / GB,
			epActive,
			epOverFabric: epSpans && epActive,
			epGBs: ep.bytes / stageTime / GB,
			ppActive,
			ppOverFabric: ppSpans && ppActive,
			ppGBs: ppLink.bytes / stageTime / GB
		},
		fabric: {
			label: fab.label,
			kind: fab.kind,
			peakGBs: peakNet / GB,
			allReduceEff: fab.allReduceEff,
			a2aEff: fab.a2aEff
		},
		cluster: {
			numGpus: c.numGpus,
			tp: c.tp,
			pp: c.pp,
			ep: c.ep,
			dp,
			gpusPerNode: c.gpusPerNode,
			numNodes,
			crossesFabric,
			scaleOut
		},
		activeParams: act
	};
}

// ---- formatting helpers shared by the UI ----
export function fmtBytes(b: number): string {
	if (b <= 0) return '0 GB';
	if (b >= GiB) return `${(b / GiB).toFixed(1)} GB`;
	return `${(b / (1024 * 1024)).toFixed(0)} MB`;
}

export function fmtTps(t: number): string {
	if (!isFinite(t)) return '—';
	if (t >= 1000) return `${(t / 1000).toFixed(1)}k`;
	if (t >= 100) return t.toFixed(0);
	return t.toFixed(1);
}

export function fmtMs(ms: number): string {
	if (!isFinite(ms)) return '—';
	if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
	return `${ms.toFixed(0)} ms`;
}

export { GiB };
