import {
	COLORS,
	FABRICS_BY_ID,
	GPUS_BY_ID,
	MODELS_BY_ID,
	NVLINK_ALLREDUCE_EFF,
	WEIGHT_FORMATS_BY_ID
} from '$lib/profiler/data';
import type { ModelSpec } from '$lib/profiler/types';
import type { Optimizer, TrainConfig, TrainProfile } from './types';

const GiB = 1024 ** 3;
const TB = 1e12;
const GB = 1e9;

// Training realizes a lower model-flop utilization than inference prefill:
// pipeline bubbles, optimizer/comm overhead, and recompute all eat into it.
export const MFU_TRAIN = 0.45;
const ACT_ELEM = 2; // activations are bf16
export const GRAD_BYTES = 2; // gradients kept in bf16 (the classic 16 B/param figure)
const MASTER_BYTES = 4; // fp32 master copy of trainable weights
export const TRAIN_WEIGHT_BYTES = 2; // trainable weights in bf16 (adapters, or full weights)
// CUDA context + comm buffers + allocator fragmentation. On a real full fine-tune
// the overhead was ~0.5 GiB (model states alone ≈ the peak);
// 1 GiB is a conservative floor that also covers NCCL buffers at scale.
export const TRAIN_OVERHEAD = 1 * GiB;
export const QLORA_COMPUTE_TAX = 1.15; // dequant overhead on the 4-bit base

// Optimizer-state bytes per trainable param (fp32 master + moments).
export const OPT_BYTES: Record<Optimizer, number> = {
	adamw: MASTER_BYTES + 4 + 4, // master + m + v (fp32) = 12
	adamw8bit: MASTER_BYTES + 1 + 1, // master + 8-bit m + 8-bit v = 6
	sgd: MASTER_BYTES + 4 // master + fp32 momentum = 8
};

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

/** Compute-active params/token. MoE cards state this; dense = total. */
export function activeParamsOf(m: ModelSpec): number {
	return m.activeParams ?? m.params;
}

/** Trainable params for LoRA/QLoRA: adapters on attention (q,k,v,o) + MLP
 * (gate,up,down). Each adapted linear of shape in×out adds r·(in+out). Summed
 * per layer ≈ r·(11·hidden + 3·intermediate). First-order (ignores which exact
 * modules are targeted and MoE expert counts). */
export function loraParams(m: ModelSpec, rank: number): number {
	const perLayerInOut = 11 * m.hiddenSize + 3 * m.intermediateSize;
	return rank * m.numLayers * perLayerInOut;
}

/** Activation memory for one transformer layer, bytes (Korthikanti et al.
 * "Reducing Activation Recomputation"): s·b·h·(34 + 5·a·s/h) without
 * checkpointing. With checkpointing you keep the layer input (2·s·b·h) per
 * layer; the extra recompute working set is added once at the whole-model level
 * (see computeTraining) rather than per layer. */
export function activationBytesPerLayer(m: ModelSpec, cfg: TrainConfig, ckpt: boolean): number {
	const s = cfg.seqLen;
	const b = cfg.microBatchSize;
	const h = m.hiddenSize;
	const a = m.numHeads;
	if (ckpt) return 2 * s * b * h;
	return s * b * h * (34 + (5 * a * s) / h);
}

export function computeTraining(cfg: TrainConfig): TrainProfile {
	const m = MODELS_BY_ID.get(cfg.modelId)!;
	const g = GPUS_BY_ID.get(cfg.gpuId)!;
	const fab = FABRICS_BY_ID.get(cfg.fabricId) ?? FABRICS_BY_ID.get('ib-ndr')!;
	const notes: string[] = [];

	// ---- parallelism ----
	const pp = cfg.ppEnabled ? Math.max(1, Math.min(cfg.pp, cfg.numGpus)) : 1;
	const tp = Math.max(1, Math.min(cfg.tp, cfg.numGpus));
	const replicaGpus = tp * pp;
	const dp = Math.max(1, Math.floor(cfg.numGpus / replicaGpus));
	const layersPerStage = m.numLayers / pp;

	// ---- what is trainable ----
	const baseParams = m.params;
	const isLora = cfg.method === 'lora' || cfg.method === 'qlora';
	const trainableParams = isLora ? loraParams(m, cfg.loraRank) : baseParams;

	// base-weight bytes/param: QLoRA freezes the base in 4-bit; else the format.
	const fmt = WEIGHT_FORMATS_BY_ID.get(cfg.weightFormatId) ?? WEIGHT_FORMATS_BY_ID.get('bf16')!;
	const nf4 = WEIGHT_FORMATS_BY_ID.get('nvfp4')!;
	const baseWeightBytes = (cfg.method === 'qlora' ? nf4.bitsPerWeight : fmt.bitsPerWeight) / 8;

	// ---- memory (per GPU) ----
	// Base (frozen for LoRA) weights: sharded by tp·pp; ZeRO-3 also shards by dp.
	const zeroWeightDivisor = cfg.zeroStage >= 3 ? dp : 1;
	const frozenBase = isLora ? baseParams : 0;
	const trainableWeightBytesTotal = isLora
		? trainableParams * TRAIN_WEIGHT_BYTES
		: baseParams * TRAIN_WEIGHT_BYTES;
	const weights =
		(frozenBase * baseWeightBytes) /
			replicaGpus /
			(cfg.method === 'qlora' ? 1 : zeroWeightDivisor) +
		trainableWeightBytesTotal / replicaGpus / zeroWeightDivisor;

	// Gradients (trainable only): ZeRO-2+ shards across dp.
	const gradDivisor = cfg.zeroStage >= 2 ? dp : 1;
	const gradients = (trainableParams * GRAD_BYTES) / replicaGpus / gradDivisor;

	// Optimizer states (trainable only): ZeRO-1+ shards across dp. CPU offload puts
	// them in host RAM so on-device is 0 — at the cost of a PCIe shuttle per step.
	const optDivisor = cfg.zeroStage >= 1 ? dp : 1;
	const optOnDevice = (trainableParams * OPT_BYTES[cfg.optimizer]) / replicaGpus / optDivisor;
	const optimizer = cfg.cpuOffload ? 0 : optOnDevice;

	// Activations: per layer × layers on this stage, sharded by tp.
	const actPerLayer = activationBytesPerLayer(m, cfg, cfg.activationCheckpointing);
	let activations = (actPerLayer * layersPerStage) / tp;
	// With checkpointing, backward recomputes one layer at a time — its full
	// (un-checkpointed) activation is live transiently. Real frameworks also keep
	// dequant/attention workspace, so the recompute peak dominates the stored
	// inputs. Calibrated against real LoRA/QLoRA runs (activation was several ×
	// the stored-inputs floor): add a small multiple of the full per-layer term.
	if (cfg.activationCheckpointing) {
		const fullLayer = activationBytesPerLayer(m, cfg, false) / tp;
		activations += 3 * fullLayer; // recompute working set + framework/attn buffers
	}

	const overhead = TRAIN_OVERHEAD;
	const used = weights + gradients + optimizer + activations + overhead;
	const capacity = g.memoryGiB * GiB;
	const fits = used <= capacity;
	const headroom = capacity - used;

	const segments: TrainProfile['segments'] = [
		{
			key: 'weights',
			label: isLora ? 'Base weights (frozen)' : 'Weights',
			bytes: weights,
			color: COLORS.weights
		},
		{ key: 'gradients', label: 'Gradients', bytes: gradients, color: COLORS.gradients },
		{ key: 'optimizer', label: 'Optimizer states', bytes: optimizer, color: COLORS.optimizer },
		{ key: 'activations', label: 'Activations', bytes: activations, color: COLORS.activations },
		{ key: 'overhead', label: 'Overhead', bytes: overhead, color: COLORS.overhead }
	];
	if (fits) segments.push({ key: 'free', label: 'Free', bytes: headroom, color: COLORS.free });
	else
		segments.push({ key: 'overflow', label: 'Overflow', bytes: -headroom, color: COLORS.overflow });

	// ---- compute / throughput ----
	// fwd+bwd ≈ 6·N/token; LoRA skips frozen weight-grads (~5); checkpointing adds
	// one recompute forward (+2); QLoRA adds dequant tax.
	const active = activeParamsOf(m);
	let flopsFactor = isLora ? 5 : 6;
	if (cfg.activationCheckpointing) flopsFactor += 2;
	const flopsPerToken = flopsFactor * active * (cfg.method === 'qlora' ? QLORA_COMPUTE_TAX : 1);
	const peakFlops = g.fp16Tflops * TB; // training math is bf16

	const globalBatchTokens = cfg.microBatchSize * cfg.seqLen * cfg.gradAccum * dp;
	// per-GPU compute for one micro-batch (work of one replica ÷ its GPUs)
	const microTokens = cfg.microBatchSize * cfg.seqLen;
	const microComputeTime =
		(flopsPerToken * microTokens) / (replicaGpus * peakFlops * MFU_TRAIN * (g.mfuFactor ?? 1));
	const computeTime = cfg.gradAccum * microComputeTime;

	// DP gradient sync once per optimizer step (reduce-scatter + all-gather ≈ 2×).
	const peakNvlink = g.nvlinkGBs * GB;
	const peakNet = (fab.gbps * GB) / 8;
	const numNodes = Math.ceil(cfg.numGpus / cfg.gpusPerNode);
	const dpCrossesFabric = dp > 1 && numNodes > 1;
	const gradSyncBytesPerGpu = (trainableParams * GRAD_BYTES) / replicaGpus; // pre-shard grad volume
	let dpSyncTime = 0;
	if (dp > 1) {
		const bw = dpCrossesFabric ? peakNet : peakNvlink;
		const eff = dpCrossesFabric ? fab.allReduceEff : NVLINK_ALLREDUCE_EFF;
		dpSyncTime = (2 * ((dp - 1) / dp) * gradSyncBytesPerGpu) / (bw * eff);
	}

	// CPU offload: every optimizer step, per-GPU optimizer bytes travel host↔GPU
	// twice (grads up + fresh params down) over PCIe. This bounds the step time.
	let offloadTime = 0;
	if (cfg.cpuOffload) {
		const bytesPerStep =
			2 * ((trainableParams * OPT_BYTES[cfg.optimizer]) / replicaGpus / optDivisor);
		offloadTime = bytesPerStep / (g.pcieGBs * 1e9);
	}
	const stepTime = computeTime + dpSyncTime + offloadTime;
	const tokensPerSec = globalBatchTokens / stepTime;
	const bottleneck: TrainProfile['bottleneck'] =
		offloadTime > Math.max(computeTime, dpSyncTime)
			? 'network'
			: dpSyncTime > computeTime
				? 'network'
				: 'compute';

	// achieved MFU: ideal compute ÷ actual step time, scaled by the modeled MFU
	const mfu = clamp01((computeTime / stepTime) * MFU_TRAIN * (g.mfuFactor ?? 1));
	const computeAchievedTflops = (flopsPerToken * globalBatchTokens) / stepTime / cfg.numGpus / TB;

	const timeToTrainHours = (cfg.datasetTokens * cfg.epochs) / tokensPerSec / 3600;

	// ---- notes ----
	if (isLora && m.moe)
		notes.push('LoRA param estimate uses dense dims; MoE experts are not separately adapted.');
	if (cfg.method === 'qlora') notes.push('QLoRA base held in 4-bit; compute dequantizes to bf16.');
	if (cfg.cpuOffload)
		notes.push(
			`ZeRO-Offload: optimizer states in host RAM. Per-step PCIe shuttle ~${(offloadTime * 1000).toFixed(0)} ms — Gen5 (${g.pcieGBs} GB/s) roughly halves this vs Gen4.`
		);
	if (bottleneck === 'network')
		notes.push(
			'Gradient sync dominates the step — raise grad-accum, use ZeRO, or a faster fabric.'
		);
	if (!fits)
		notes.push(
			'Does not fit: raise TP/PP, a higher ZeRO stage, enable checkpointing, or LoRA/QLoRA.'
		);

	return {
		perGpu: {
			weights,
			gradients,
			optimizer,
			activations,
			overhead,
			used,
			capacity,
			fits,
			headroom
		},
		segments,
		trainableParams,
		baseParams,
		dp,
		replicaGpus,
		globalBatchTokens,
		stepTimeMs: stepTime * 1000,
		tokensPerSec,
		mfu,
		timeToTrainHours,
		bottleneck,
		computeAchievedTflops,
		computePeakTflops: (peakFlops * MFU_TRAIN * (g.mfuFactor ?? 1)) / TB,
		notes
	};
}
