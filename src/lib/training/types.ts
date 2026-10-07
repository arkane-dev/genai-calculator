import type { MemorySegment } from '$lib/profiler/types';

// Training / fine-tuning sizing. Unlike inference (weights + KV), training holds
// weights + gradients + optimizer states + activations, so the memory story is
// the point. Mixed-precision AdamW ≈ 16 bytes/param of model states.

export type TrainMethod = 'full' | 'lora' | 'qlora';
export type Optimizer = 'adamw' | 'adamw8bit' | 'sgd';
export type ZeroStage = 0 | 1 | 2 | 3;

export interface TrainConfig {
	modelId: string;
	gpuId: string;
	numGpus: number;
	gpusPerNode: number;
	fabricId: string;

	method: TrainMethod; // full fine-tune / LoRA / QLoRA (4-bit frozen base)
	optimizer: Optimizer;
	weightFormatId: string; // base-weight format for full/LoRA (QLoRA forces 4-bit)

	tp: number; // tensor-parallel degree (intra-node)
	pp: number; // pipeline-parallel degree
	ppEnabled: boolean;
	zeroStage: ZeroStage; // shards optimizer(1) / +grads(2) / +weights(3) across DP
	activationCheckpointing: boolean; // recompute activations in backward
	cpuOffload: boolean; // ZeRO-Offload: optimizer state (and grads) live in host RAM,
	// shuttled over PCIe every step. Slashes GPU memory, adds a PCIe-bound step term.

	microBatchSize: number; // sequences per micro-batch per DP replica
	gradAccum: number; // micro-batches per optimizer step
	seqLen: number; // training sequence length (tokens)

	loraRank: number; // rank for LoRA / QLoRA adapters

	datasetTokens: number; // dataset size for time-to-train
	epochs: number;
}

export interface TrainProfile {
	perGpu: {
		weights: number;
		gradients: number;
		optimizer: number;
		activations: number;
		overhead: number;
		used: number;
		capacity: number;
		fits: boolean;
		headroom: number;
	};
	segments: MemorySegment[];

	trainableParams: number; // params that get gradients + optimizer state
	baseParams: number; // total model params
	dp: number; // data-parallel replicas
	replicaGpus: number; // GPUs per model replica (tp × pp)
	globalBatchTokens: number; // tokens per optimizer step across the cluster

	stepTimeMs: number; // one optimizer step (all grad-accum micro-steps + sync)
	tokensPerSec: number; // training throughput across the cluster
	mfu: number; // achieved model-flop utilization (0..1)
	timeToTrainHours: number; // datasetTokens × epochs ÷ tokensPerSec
	bottleneck: 'compute' | 'network';

	computeAchievedTflops: number;
	computePeakTflops: number;

	// notes for the UI (approximations, warnings)
	notes: string[];
}
