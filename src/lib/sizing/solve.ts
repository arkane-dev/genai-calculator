import { computeProfile } from '$lib/profiler/calc';
import { FABRICS_BY_ID, GPUS, GPUS_BY_ID, MODELS_BY_ID } from '$lib/profiler/data';
import type { Config } from '$lib/profiler/types';
import type { DisaggResult, PoolSizing, SizeResult, WorkloadSpec } from './types';

// A given concurrency number means different provisioning depending on how it
// was measured. Peak is what you must serve; average needs burst headroom.
const BASIS_FACTOR = { average: 2.0, p90: 1.2, peak: 1.0 } as const;

const POW2 = [1, 2, 4, 8, 16, 32];
const PP_MAX = 16; // pipeline stages can span many nodes
const EP_MAX = 64; // expert shards (MoE) can span many nodes
// Batch-per-replica probe points. perUserTps decreases monotonically with batch,
// so we walk up and stop at the first point that breaks the SLO.
const BATCHES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512];

function makeConfig(
	spec: WorkloadSpec,
	gpuId: string,
	numGpus: number,
	tp: number,
	pp: number,
	ep: number,
	batchSize: number
): Config {
	return {
		modelId: spec.modelId,
		gpuId,
		numGpus,
		gpusPerNode: Math.min(spec.gpusPerNode, numGpus),
		fabricId: spec.fabricId,
		tp,
		pp,
		ppEnabled: pp > 1,
		ep,
		epEnabled: ep > 1,
		batchSize,
		inputTokens: spec.inputTokens,
		outputTokens: spec.outputTokens,
		weightFormatId: spec.weightFormatId,
		kvBits: spec.kvBits,
		phase: 'decode',
		kvAllocation: spec.kvAllocation,
		steps: spec.steps,
		resolution: spec.resolution,
		guidance: spec.guidance,
		frames: spec.frames,
		specDecode: spec.specDecode,
		draftAcceptRate: spec.draftAcceptRate,
		specTokens: spec.specTokens,
		cachedPrefixFrac: spec.cachedPrefixFrac,
		imagesPerRequest: 0
	};
}

/** Diffusion sizing: smallest cluster that produces the target images/sec while
 * keeping each image under the latency SLO. Dense (TP/PP + DP replicas). */
function sizeForGpuDiffusion(spec: WorkloadSpec, gpuId: string): SizeResult {
	const gpu = GPUS_BY_ID.get(gpuId)!;
	const targetIps = Math.max(0.0001, spec.targetImagesPerSec);
	const maxSec = spec.maxSecPerImage;
	const tpMax = Math.min(spec.gpusPerNode, 8);

	let best: {
		tp: number;
		pp: number;
		dp: number;
		B: number;
		replicaGpus: number;
		numGpus: number;
		ips: number;
		secPerImage: number;
	} | null = null;
	let sawFit = false;
	let sawLatency = false;

	for (const tp of POW2.filter((n) => n <= tpMax)) {
		for (const pp of POW2.filter((n) => n <= PP_MAX)) {
			const replicaGpus = tp * pp;
			const base = makeConfig(spec, gpuId, replicaGpus, tp, pp, 1, 1);
			const p1 = computeProfile(base);
			if (!p1.perGpu.fits || !p1.diffusion) continue;
			sawFit = true;
			if (p1.diffusion.secPerImage > maxSec) continue; // batch 1 already too slow
			sawLatency = true;

			// pick the batch that maximizes replica images/sec while staying under
			// the latency SLO and fitting in memory
			let B = 1;
			let ips = p1.diffusion.imagesPerSec;
			let secAt = p1.diffusion.secPerImage;
			for (const b of BATCHES.filter((n) => n > 1)) {
				const pb = computeProfile({ ...base, batchSize: b });
				if (!pb.perGpu.fits || !pb.diffusion || pb.diffusion.secPerImage > maxSec) break;
				if (pb.diffusion.imagesPerSec > ips) {
					B = b;
					ips = pb.diffusion.imagesPerSec;
					secAt = pb.diffusion.secPerImage;
				}
			}
			const dp = Math.max(1, Math.ceil(targetIps / ips));
			const numGpus = replicaGpus * dp;
			if (!best || numGpus < best.numGpus)
				best = { tp, pp, dp, B, replicaGpus, numGpus, ips, secPerImage: secAt };
		}
	}

	if (!best) {
		const reason = !sawFit
			? 'model does not fit even at max tensor+pipeline parallelism'
			: !sawLatency
				? 'cannot meet the per-image latency target on this GPU'
				: 'no feasible configuration';
		return {
			gpuId,
			gpuName: gpu.name,
			feasible: false,
			reason,
			tp: 0,
			pp: 0,
			ep: 0,
			dp: 0,
			replicaGpus: 0,
			batchPerReplica: 0,
			provisionedConcurrency: 0,
			numGpus: 0,
			numNodes: 0,
			perUserTps: 0,
			ttftMs: 0,
			fullReqMs: 0,
			clusterTps: 0,
			config: makeConfig(spec, gpuId, tpMax * PP_MAX, tpMax, PP_MAX, 1, 1) // best-attempt (max-parallel) layout so the UI can render die/memory with overflow
		};
	}

	const config = makeConfig(spec, gpuId, best.numGpus, best.tp, best.pp, 1, best.B);
	const p = computeProfile(config);
	return {
		gpuId,
		gpuName: gpu.name,
		feasible: true,
		tp: best.tp,
		pp: best.pp,
		ep: 1,
		dp: best.dp,
		replicaGpus: best.replicaGpus,
		batchPerReplica: best.B,
		provisionedConcurrency: 0,
		numGpus: best.numGpus,
		numNodes: Math.ceil(best.numGpus / spec.gpusPerNode),
		perUserTps: 0,
		ttftMs: 0,
		fullReqMs: 0,
		clusterTps: 0,
		imagesPerSec: p.diffusion?.imagesPerSec ?? best.ips * best.dp,
		secPerImage: best.secPerImage,
		config
	};
}

/** JEPA sizing: smallest cluster that encodes the target clips/sec while keeping
 * each clip under the latency SLO. Dense (TP/PP + DP replicas), like diffusion. */
function sizeForGpuJepa(spec: WorkloadSpec, gpuId: string): SizeResult {
	const gpu = GPUS_BY_ID.get(gpuId)!;
	const targetCps = Math.max(0.0001, spec.targetClipsPerSec);
	const maxSec = spec.maxSecPerClip;
	const tpMax = Math.min(spec.gpusPerNode, 8);

	let best: {
		tp: number;
		pp: number;
		dp: number;
		B: number;
		replicaGpus: number;
		numGpus: number;
		cps: number;
		secPerClip: number;
	} | null = null;
	let sawFit = false;
	let sawLatency = false;

	for (const tp of POW2.filter((n) => n <= tpMax)) {
		for (const pp of POW2.filter((n) => n <= PP_MAX)) {
			const replicaGpus = tp * pp;
			const base = makeConfig(spec, gpuId, replicaGpus, tp, pp, 1, 1);
			const p1 = computeProfile(base);
			if (!p1.perGpu.fits || !p1.jepa) continue;
			sawFit = true;
			if (p1.jepa.secPerClip > maxSec) continue; // batch 1 already too slow
			sawLatency = true;

			// pick the batch that maximizes replica clips/sec under the latency SLO
			let B = 1;
			let cps = p1.jepa.clipsPerSec;
			let secAt = p1.jepa.secPerClip;
			for (const b of BATCHES.filter((n) => n > 1)) {
				const pb = computeProfile({ ...base, batchSize: b });
				if (!pb.perGpu.fits || !pb.jepa || pb.jepa.secPerClip > maxSec) break;
				if (pb.jepa.clipsPerSec > cps) {
					B = b;
					cps = pb.jepa.clipsPerSec;
					secAt = pb.jepa.secPerClip;
				}
			}
			const dp = Math.max(1, Math.ceil(targetCps / cps));
			const numGpus = replicaGpus * dp;
			if (!best || numGpus < best.numGpus)
				best = { tp, pp, dp, B, replicaGpus, numGpus, cps, secPerClip: secAt };
		}
	}

	if (!best) {
		const reason = !sawFit
			? 'model does not fit even at max tensor+pipeline parallelism'
			: !sawLatency
				? 'cannot meet the per-clip latency target on this GPU'
				: 'no feasible configuration';
		return {
			gpuId,
			gpuName: gpu.name,
			feasible: false,
			reason,
			tp: 0,
			pp: 0,
			ep: 0,
			dp: 0,
			replicaGpus: 0,
			batchPerReplica: 0,
			provisionedConcurrency: 0,
			numGpus: 0,
			numNodes: 0,
			perUserTps: 0,
			ttftMs: 0,
			fullReqMs: 0,
			clusterTps: 0,
			config: makeConfig(spec, gpuId, tpMax * PP_MAX, tpMax, PP_MAX, 1, 1) // best-attempt (max-parallel) layout so the UI can render die/memory with overflow
		};
	}

	const config = makeConfig(spec, gpuId, best.numGpus, best.tp, best.pp, 1, best.B);
	const p = computeProfile(config);
	return {
		gpuId,
		gpuName: gpu.name,
		feasible: true,
		tp: best.tp,
		pp: best.pp,
		ep: 1,
		dp: best.dp,
		replicaGpus: best.replicaGpus,
		batchPerReplica: best.B,
		provisionedConcurrency: 0,
		numGpus: best.numGpus,
		numNodes: Math.ceil(best.numGpus / spec.gpusPerNode),
		perUserTps: 0,
		ttftMs: 0,
		fullReqMs: 0,
		clusterTps: 0,
		clipsPerSec: p.jepa?.clipsPerSec ?? best.cps * best.dp,
		secPerClip: best.secPerClip,
		config
	};
}

/** Encoder sizing (embeddings, rerankers): smallest cluster that produces the
 * target docs/sec while keeping each batch under the latency SLO. Same shape as
 * JEPA sizing — TP/PP + DP replicas, walk the batch. */
function sizeForGpuEncoder(spec: WorkloadSpec, gpuId: string): SizeResult {
	const gpu = GPUS_BY_ID.get(gpuId)!;
	const targetDps = Math.max(0.0001, spec.targetDocsPerSec);
	const maxSec = spec.maxSecPerDoc;
	const tpMax = Math.min(spec.gpusPerNode, 8);

	let best: {
		tp: number;
		pp: number;
		dp: number;
		B: number;
		replicaGpus: number;
		numGpus: number;
		dps: number;
		secPerDoc: number;
	} | null = null;
	let sawFit = false;
	let sawLatency = false;

	for (const tp of POW2.filter((n) => n <= tpMax)) {
		for (const pp of POW2.filter((n) => n <= PP_MAX)) {
			const replicaGpus = tp * pp;
			const base = {
				...makeConfig(spec, gpuId, replicaGpus, tp, pp, 1, 1),
				phase: 'prefill' as const
			};
			const p1 = computeProfile(base);
			if (!p1.perGpu.fits || !p1.encoder) continue;
			sawFit = true;
			if (p1.encoder.secPerDoc > maxSec) continue;
			sawLatency = true;

			let B = 1;
			let dps = p1.encoder.docsPerSec;
			let secAt = p1.encoder.secPerDoc;
			for (const b of BATCHES.filter((n) => n > 1)) {
				const pb = computeProfile({ ...base, batchSize: b });
				if (!pb.perGpu.fits || !pb.encoder || pb.encoder.secPerDoc > maxSec) break;
				if (pb.encoder.docsPerSec > dps) {
					B = b;
					dps = pb.encoder.docsPerSec;
					secAt = pb.encoder.secPerDoc;
				}
			}
			const dp = Math.max(1, Math.ceil(targetDps / dps));
			const numGpus = replicaGpus * dp;
			if (!best || numGpus < best.numGpus)
				best = { tp, pp, dp, B, replicaGpus, numGpus, dps, secPerDoc: secAt };
		}
	}

	if (!best) {
		const reason = !sawFit
			? 'encoder does not fit even at max tensor+pipeline parallelism'
			: !sawLatency
				? 'cannot meet the per-doc latency target on this GPU'
				: 'no feasible configuration';
		return {
			gpuId,
			gpuName: gpu.name,
			feasible: false,
			reason,
			tp: 0,
			pp: 0,
			ep: 0,
			dp: 0,
			replicaGpus: 0,
			batchPerReplica: 0,
			provisionedConcurrency: 0,
			numGpus: 0,
			numNodes: 0,
			perUserTps: 0,
			ttftMs: 0,
			fullReqMs: 0,
			clusterTps: 0,
			config: makeConfig(spec, gpuId, tpMax * PP_MAX, tpMax, PP_MAX, 1, 1) // best-attempt (max-parallel) layout so the UI can render die/memory with overflow
		};
	}

	const config = {
		...makeConfig(spec, gpuId, best.numGpus, best.tp, best.pp, 1, best.B),
		phase: 'prefill' as const
	};
	const p = computeProfile(config);
	return {
		gpuId,
		gpuName: gpu.name,
		feasible: true,
		tp: best.tp,
		pp: best.pp,
		ep: 1,
		dp: best.dp,
		replicaGpus: best.replicaGpus,
		batchPerReplica: best.B,
		provisionedConcurrency: 0,
		numGpus: best.numGpus,
		numNodes: Math.ceil(best.numGpus / spec.gpusPerNode),
		perUserTps: 0,
		ttftMs: 0,
		fullReqMs: 0,
		clusterTps: 0,
		docsPerSec: p.encoder?.docsPerSec ?? best.dps * best.dp,
		secPerDoc: best.secPerDoc,
		config
	};
}

/** VLA sizing: smallest cluster that drives the target number of robots at the
 * model's control frequency, with each chunk under the latency SLO. */
function sizeForGpuVla(spec: WorkloadSpec, gpuId: string): SizeResult {
	const gpu = GPUS_BY_ID.get(gpuId)!;
	const model = MODELS_BY_ID.get(spec.modelId)!;
	const controlHz = model.vla?.controlHz ?? 50;
	const targetActionsPerSec = Math.max(0.0001, spec.targetRobots * controlHz);
	const maxSec = spec.maxSecPerControl;
	const tpMax = Math.min(spec.gpusPerNode, 8);

	let best: {
		tp: number;
		pp: number;
		dp: number;
		B: number;
		replicaGpus: number;
		numGpus: number;
		aps: number;
		secPerControl: number;
	} | null = null;
	let sawFit = false;
	let sawLatency = false;

	for (const tp of POW2.filter((n) => n <= tpMax)) {
		for (const pp of POW2.filter((n) => n <= PP_MAX)) {
			const replicaGpus = tp * pp;
			const base = makeConfig(spec, gpuId, replicaGpus, tp, pp, 1, 1);
			const p1 = computeProfile(base);
			if (!p1.perGpu.fits || !p1.vla) continue;
			sawFit = true;
			if (p1.vla.secPerControl > maxSec) continue;
			sawLatency = true;

			let B = 1;
			let aps = p1.vla.actionsPerSec;
			let secAt = p1.vla.secPerControl;
			for (const b of BATCHES.filter((n) => n > 1 && n <= 16)) {
				const pb = computeProfile({ ...base, batchSize: b });
				if (!pb.perGpu.fits || !pb.vla || pb.vla.secPerControl > maxSec) break;
				if (pb.vla.actionsPerSec > aps) {
					B = b;
					aps = pb.vla.actionsPerSec;
					secAt = pb.vla.secPerControl;
				}
			}
			const dp = Math.max(1, Math.ceil(targetActionsPerSec / aps));
			const numGpus = replicaGpus * dp;
			if (!best || numGpus < best.numGpus)
				best = { tp, pp, dp, B, replicaGpus, numGpus, aps, secPerControl: secAt };
		}
	}

	if (!best) {
		const reason = !sawFit
			? 'VLA does not fit even at max tensor+pipeline parallelism'
			: !sawLatency
				? 'cannot meet the per-chunk latency SLO on this GPU'
				: 'no feasible configuration';
		return {
			gpuId,
			gpuName: gpu.name,
			feasible: false,
			reason,
			tp: 0,
			pp: 0,
			ep: 0,
			dp: 0,
			replicaGpus: 0,
			batchPerReplica: 0,
			provisionedConcurrency: 0,
			numGpus: 0,
			numNodes: 0,
			perUserTps: 0,
			ttftMs: 0,
			fullReqMs: 0,
			clusterTps: 0,
			config: makeConfig(spec, gpuId, tpMax * PP_MAX, tpMax, PP_MAX, 1, 1) // best-attempt (max-parallel) layout so the UI can render die/memory with overflow
		};
	}

	const config = makeConfig(spec, gpuId, best.numGpus, best.tp, best.pp, 1, best.B);
	const p = computeProfile(config);
	return {
		gpuId,
		gpuName: gpu.name,
		feasible: true,
		tp: best.tp,
		pp: best.pp,
		ep: 1,
		dp: best.dp,
		replicaGpus: best.replicaGpus,
		batchPerReplica: best.B,
		provisionedConcurrency: 0,
		numGpus: best.numGpus,
		numNodes: Math.ceil(best.numGpus / spec.gpusPerNode),
		perUserTps: 0,
		ttftMs: 0,
		fullReqMs: 0,
		clusterTps: 0,
		robotsDriven: (p.vla?.actionsPerSec ?? best.aps * best.dp) / controlHz,
		secPerControl: best.secPerControl,
		config
	};
}

/** ASR sizing: smallest cluster that transcribes `targetStreams` concurrent
 * real-time audio streams with per-replica RTF ≥ minRtf. */
function sizeForGpuAsr(spec: WorkloadSpec, gpuId: string): SizeResult {
	const gpu = GPUS_BY_ID.get(gpuId)!;
	const target = Math.max(1, spec.targetStreams);
	const minRtf = Math.max(0.1, spec.minRtf);
	const tpMax = Math.min(spec.gpusPerNode, 8);

	let best: {
		tp: number;
		pp: number;
		dp: number;
		B: number;
		replicaGpus: number;
		numGpus: number;
		rtf: number;
		streams: number;
	} | null = null;
	let sawFit = false;
	let sawRtf = false;

	for (const tp of POW2.filter((n) => n <= tpMax)) {
		for (const pp of POW2.filter((n) => n <= PP_MAX)) {
			const replicaGpus = tp * pp;
			const base = makeConfig(spec, gpuId, replicaGpus, tp, pp, 1, 1);
			const p1 = computeProfile(base);
			if (!p1.perGpu.fits || !p1.asr) continue;
			sawFit = true;
			if (p1.asr.rtf < minRtf) continue;
			sawRtf = true;

			// walk the batch: bigger batches process more streams per replica but
			// each stream's individual RTF shrinks linearly (compute-bound path)
			let B = 1;
			let rtfAt = p1.asr.rtf;
			let streamsAt = p1.asr.rtf; // streams a single replica serves = its RTF
			for (const b of BATCHES.filter((n) => n > 1 && n <= 64)) {
				const pb = computeProfile({ ...base, batchSize: b });
				if (!pb.perGpu.fits || !pb.asr) break;
				// per-stream RTF = replica-RTF-at-batch-b / batch. If it drops below minRtf, stop.
				const perStreamRtf = (pb.asr.rtf * b) / b; // pb.asr.rtf already per replica; each of B streams gets rtf-per-batch of pb.asr.rtf/B
				// Actually with batched decode, all B streams share the wall time — per-stream RTF equals the replica's RTF for one full window done in parallel:
				const replicaStreams = pb.asr.audioSecPerSec / 30; /* window sec, safe fallback */
				if (perStreamRtf < minRtf) break;
				B = b;
				rtfAt = pb.asr.rtf;
				streamsAt = replicaStreams; // streams per replica that can be served at ≥ real-time
			}
			const dp = Math.max(1, Math.ceil(target / streamsAt));
			const numGpus = replicaGpus * dp;
			if (!best || numGpus < best.numGpus)
				best = { tp, pp, dp, B, replicaGpus, numGpus, rtf: rtfAt, streams: streamsAt };
		}
	}

	if (!best) {
		const reason = !sawFit
			? 'ASR model does not fit even at max tensor+pipeline parallelism'
			: !sawRtf
				? 'cannot meet the min real-time factor SLO on this GPU'
				: 'no feasible configuration';
		return {
			gpuId,
			gpuName: gpu.name,
			feasible: false,
			reason,
			tp: 0,
			pp: 0,
			ep: 0,
			dp: 0,
			replicaGpus: 0,
			batchPerReplica: 0,
			provisionedConcurrency: 0,
			numGpus: 0,
			numNodes: 0,
			perUserTps: 0,
			ttftMs: 0,
			fullReqMs: 0,
			clusterTps: 0,
			config: makeConfig(spec, gpuId, tpMax * PP_MAX, tpMax, PP_MAX, 1, 1) // best-attempt (max-parallel) layout so the UI can render die/memory with overflow
		};
	}

	const config = makeConfig(spec, gpuId, best.numGpus, best.tp, best.pp, 1, best.B);
	const p = computeProfile(config);
	return {
		gpuId,
		gpuName: gpu.name,
		feasible: true,
		tp: best.tp,
		pp: best.pp,
		ep: 1,
		dp: best.dp,
		replicaGpus: best.replicaGpus,
		batchPerReplica: best.B,
		provisionedConcurrency: 0,
		numGpus: best.numGpus,
		numNodes: Math.ceil(best.numGpus / spec.gpusPerNode),
		perUserTps: 0,
		ttftMs: 0,
		fullReqMs: 0,
		clusterTps: 0,
		streamsServed: best.streams * best.dp,
		rtf: p.asr?.rtf ?? best.rtf,
		config
	};
}

/** Smallest cluster of one GPU type that meets fit + TTFT + per-user throughput
 * for the provisioned concurrency. Searches tensor/pipeline/expert parallelism
 * and scales data-parallel replicas across as many nodes as needed. Returns
 * feasible=false with a reason if no cluster works. */
export function sizeForGpu(spec: WorkloadSpec, gpuId: string): SizeResult {
	const gpu = GPUS_BY_ID.get(gpuId)!;
	const model = MODELS_BY_ID.get(spec.modelId)!;
	if (model.kind === 'diffusion') return sizeForGpuDiffusion(spec, gpuId);
	if (model.kind === 'jepa') return sizeForGpuJepa(spec, gpuId);
	if (model.kind === 'encoder') return sizeForGpuEncoder(spec, gpuId);
	if (model.kind === 'vla') return sizeForGpuVla(spec, gpuId);
	if (model.kind === 'asr') return sizeForGpuAsr(spec, gpuId);
	const provisioned = Math.ceil(spec.concurrency * BASIS_FACTOR[spec.basis]);
	// TP stays within a node (NVLink); PP and EP may span nodes.
	const tpMax = Math.min(spec.gpusPerNode, 8);
	const epOptions = model.moe ? POW2.filter((n) => n <= EP_MAX) : [1];

	let best: {
		tp: number;
		pp: number;
		ep: number;
		dp: number;
		B: number;
		replicaGpus: number;
		numGpus: number;
		ttftMs: number;
	} | null = null;
	let sawFit = false;
	let sawTtft = false;
	let bestPerUser1 = 0; // best achievable per-user tok/s at batch 1 (single sequence), for the reason string
	// A representative single-replica config so the UI can still render the die/memory (with
	// overflow) when infeasible. Prefer the max-effort config that FITS; else the most-sharded
	// attempt (which shows the memory overflow that makes it infeasible).
	const maxEp = epOptions[epOptions.length - 1];
	let probe = makeConfig(spec, gpuId, PP_MAX * Math.max(tpMax, maxEp), tpMax, PP_MAX, maxEp, 1);

	for (const tp of POW2.filter((n) => n <= tpMax)) {
		for (const pp of POW2.filter((n) => n <= PP_MAX)) {
			for (const ep of epOptions) {
				// One replica: pp pipeline stages, each holding tp tensor shards and
				// ep expert shards co-located (the larger dimension sets the width).
				const replicaGpus = pp * Math.max(tp, ep);
				// Per-GPU memory / TTFT depend only on tp/pp/ep, not dp — probe one
				// replica (numGpus = replicaGpus so calc keeps our tp/pp/ep).
				const base = makeConfig(spec, gpuId, replicaGpus, tp, pp, ep, 1);
				const p1 = computeProfile(base);
				if (!p1.perGpu.fits) continue;
				if (!sawFit) probe = base; // first config that fits — keep as the representative view
				sawFit = true;
				if (p1.ttftMs > spec.ttftTargetMs) continue;
				sawTtft = true;
				bestPerUser1 = Math.max(bestPerUser1, p1.perUserTps);

				// largest batch/replica that still meets the per-user throughput SLO and fits in KV
				let B = 0;
				for (const b of BATCHES) {
					const pb = computeProfile({ ...base, batchSize: b });
					if (!pb.perGpu.fits || b > pb.maxConcurrentSeqs) break;
					if (pb.perUserTps < spec.throughputTargetTps) break;
					B = b;
				}
				if (B < 1) continue;

				const dp = Math.max(1, Math.ceil(provisioned / B));
				const numGpus = replicaGpus * dp;
				// prefer fewer GPUs; tie-break toward fewer replica GPUs (simpler topology)
				if (
					!best ||
					numGpus < best.numGpus ||
					(numGpus === best.numGpus && replicaGpus < best.replicaGpus)
				)
					best = { tp, pp, ep, dp, B, replicaGpus, numGpus, ttftMs: p1.ttftMs };
			}
		}
	}

	if (!best) {
		const reason = !sawFit
			? 'model does not fit even at max tensor+pipeline+expert parallelism'
			: !sawTtft
				? 'cannot meet the TTFT target on this GPU'
				: `min throughput ${spec.throughputTargetTps} tok/s/user exceeds the hardware ceiling (best ${bestPerUser1.toFixed(0)} tok/s even at batch 1, full tensor-parallel) — a per-user decode limit, not a cluster-size cap`;
		// Load the die for the architecture view: profile the probe at the largest batch
		// that fits, so the compute die shows the GPU under real load (batch-1 decode
		// leaves the tensor cores idle). If nothing fits (memory overflow), batch 1 stays
		// and the memory box shows the overflow.
		let probeB = 1;
		for (const b of BATCHES) {
			const pb = computeProfile({ ...probe, batchSize: b });
			if (!pb.perGpu.fits || b > pb.maxConcurrentSeqs) break;
			probeB = b;
		}
		return {
			gpuId,
			gpuName: gpu.name,
			feasible: false,
			reason,
			tp: 0,
			pp: 0,
			ep: 0,
			dp: 0,
			replicaGpus: 0,
			batchPerReplica: 0,
			provisionedConcurrency: provisioned,
			numGpus: 0,
			numNodes: 0,
			perUserTps: 0,
			ttftMs: 0,
			fullReqMs: 0,
			clusterTps: 0,
			config: { ...probe, batchSize: probeB }
		};
	}

	// final config sized to the whole cluster (dp replicas) for the visuals
	const config = makeConfig(spec, gpuId, best.numGpus, best.tp, best.pp, best.ep, best.B);
	const p = computeProfile(config);
	const fullReqMs = best.ttftMs + (spec.outputTokens / p.perUserTps) * 1000;
	const servedConcurrency = best.B * best.dp;
	return {
		gpuId,
		gpuName: gpu.name,
		feasible: true,
		tp: best.tp,
		pp: best.pp,
		ep: best.ep,
		dp: best.dp,
		replicaGpus: best.replicaGpus,
		batchPerReplica: best.B,
		provisionedConcurrency: provisioned,
		numGpus: best.numGpus,
		numNodes: Math.ceil(best.numGpus / spec.gpusPerNode),
		perUserTps: p.perUserTps,
		ttftMs: best.ttftMs,
		fullReqMs,
		clusterTps: p.perUserTps * servedConcurrency,
		config
	};
}

// ---------- Disaggregated prefill/decode sizing ----------
//
// Modern serving separates prefill (compute-bound, TTFT-driven) from decode
// (memory-bound, throughput-driven). Two independent pools with different
// parallelism + batch shapes, joined by a KV-cache hand-off over the fabric.
// Sizes each pool as the smallest fit, then meets both flow rates.

/** Time to move one request's KV cache from a prefill replica to a decode
 * replica over the fabric. Approximate: kv bytes for the input, split by TP on
 * both sides and shipped once. */
function kvTransferSeconds(
	spec: WorkloadSpec,
	m: ReturnType<typeof MODELS_BY_ID.get>,
	prefillTp: number,
	decodeTp: number
): number {
	if (!m) return 0;
	const fab = FABRICS_BY_ID.get(spec.fabricId) ?? FABRICS_BY_ID.get('ib-ndr')!;
	const kvHeads = m.numKvHeads ?? m.numHeads;
	const headDim = m.headDim;
	const layers = m.kvLayers ?? m.numLayers;
	const bytes = 2 * spec.inputTokens * layers * kvHeads * headDim * (spec.kvBits / 8);
	const perGpuBytes = bytes / Math.max(prefillTp, decodeTp); // TP shards KV
	const bw = (fab.gbps * 1e9) / 8; // bytes/s
	return perGpuBytes / (bw * fab.allReduceEff);
}

/** Find the smallest prefill replica (tp/pp) that meets the TTFT budget on a
 * single prompt. Returns null if no configuration fits and is fast enough. */
function bestPrefillReplica(
	spec: WorkloadSpec,
	gpuId: string,
	ttftBudgetMs: number
): {
	tp: number;
	pp: number;
	ep: number;
	replicaGpus: number;
	msPerPrompt: number;
	config: Config;
} | null {
	const tpMax = Math.min(spec.gpusPerNode, 8);
	const model = MODELS_BY_ID.get(spec.modelId)!;
	const epOptions = model.moe ? POW2.filter((n) => n <= EP_MAX) : [1];
	let best: {
		tp: number;
		pp: number;
		ep: number;
		replicaGpus: number;
		msPerPrompt: number;
		config: Config;
	} | null = null;
	for (const tp of POW2.filter((n) => n <= tpMax)) {
		for (const pp of POW2.filter((n) => n <= PP_MAX)) {
			for (const ep of epOptions) {
				const replicaGpus = pp * Math.max(tp, ep);
				const cfg = makeConfig(spec, gpuId, replicaGpus, tp, pp, ep, 1);
				cfg.phase = 'prefill';
				const p = computeProfile(cfg);
				if (!p.perGpu.fits) continue;
				if (p.ttftMs > ttftBudgetMs) continue;
				if (!best || replicaGpus < best.replicaGpus)
					best = { tp, pp, ep, replicaGpus, msPerPrompt: p.ttftMs, config: cfg };
			}
		}
	}
	return best;
}

/** Find the smallest decode replica (tp/pp/ep, batch) that meets the per-user
 * throughput SLO — same as the aggregated solver's inner loop, minus TTFT. */
function bestDecodeReplica(
	spec: WorkloadSpec,
	gpuId: string
): {
	tp: number;
	pp: number;
	ep: number;
	replicaGpus: number;
	batch: number;
	perUserTps: number;
	config: Config;
} | null {
	const tpMax = Math.min(spec.gpusPerNode, 8);
	const model = MODELS_BY_ID.get(spec.modelId)!;
	const epOptions = model.moe ? POW2.filter((n) => n <= EP_MAX) : [1];
	let best: {
		tp: number;
		pp: number;
		ep: number;
		replicaGpus: number;
		batch: number;
		perUserTps: number;
		config: Config;
	} | null = null;
	for (const tp of POW2.filter((n) => n <= tpMax)) {
		for (const pp of POW2.filter((n) => n <= PP_MAX)) {
			for (const ep of epOptions) {
				const replicaGpus = pp * Math.max(tp, ep);
				const base = makeConfig(spec, gpuId, replicaGpus, tp, pp, ep, 1);
				const p1 = computeProfile(base);
				if (!p1.perGpu.fits) continue;
				if (p1.perUserTps < spec.throughputTargetTps) continue;
				let batch = 0;
				let perUser = 0;
				for (const b of BATCHES) {
					const pb = computeProfile({ ...base, batchSize: b });
					if (!pb.perGpu.fits || b > pb.maxConcurrentSeqs) break;
					if (pb.perUserTps < spec.throughputTargetTps) break;
					batch = b;
					perUser = pb.perUserTps;
				}
				if (batch < 1) continue;
				if (!best || replicaGpus < best.replicaGpus)
					best = {
						tp,
						pp,
						ep,
						replicaGpus,
						batch,
						perUserTps: perUser,
						config: { ...base, batchSize: batch }
					};
			}
		}
	}
	return best;
}

/** Size a disaggregated prefill/decode deployment on one GPU type. */
export function sizeDisaggregated(spec: WorkloadSpec, gpuId: string): DisaggResult {
	const gpu = GPUS_BY_ID.get(gpuId);
	if (!gpu) return emptyDisagg('unknown GPU');

	const provisioned = Math.ceil(spec.concurrency * BASIS_FACTOR[spec.basis]);
	// Approximate steady-state prompt rate: each active user generates
	// outputTokens at their perUserTps, so one prompt every outputTokens/perUserTps
	// seconds. Use the throughput SLO as the floor for perUserTps.
	const promptsPerSec = (provisioned * spec.throughputTargetTps) / Math.max(1, spec.outputTokens);

	// Reserve half the TTFT budget for KV transfer at first; the model may adjust.
	// Prefill replica is picked to leave a healthy margin for the actual transfer.
	const ttftPrefillBudget = spec.ttftTargetMs * 0.7;
	const prefillPick = bestPrefillReplica(spec, gpuId, ttftPrefillBudget);
	if (!prefillPick) return emptyDisagg('prefill pool cannot meet TTFT on this GPU');
	const decodePick = bestDecodeReplica(spec, gpuId);
	if (!decodePick) return emptyDisagg('decode pool cannot meet per-user throughput on this GPU');

	// KV transfer over the fabric between the two pools.
	const model = MODELS_BY_ID.get(spec.modelId);
	const kvSec = kvTransferSeconds(spec, model, prefillPick.tp, decodePick.tp);
	const kvTransferMs = kvSec * 1000;
	const ttftMs = prefillPick.msPerPrompt + kvTransferMs;
	if (ttftMs > spec.ttftTargetMs)
		return emptyDisagg('prefill + KV transfer exceeds the TTFT target');

	// Prefill replicas: enough to sustain promptsPerSec.
	const promptsPerSecPerPrefillReplica = 1000 / prefillPick.msPerPrompt;
	const prefillReplicas = Math.max(1, Math.ceil(promptsPerSec / promptsPerSecPerPrefillReplica));

	// Decode replicas: each holds `batch` active users concurrently.
	const decodeReplicas = Math.max(1, Math.ceil(provisioned / decodePick.batch));

	const prefillGpus = prefillPick.replicaGpus * prefillReplicas;
	const decodeGpus = decodePick.replicaGpus * decodeReplicas;
	const prefillNodes = Math.ceil(prefillGpus / spec.gpusPerNode);
	const decodeNodes = Math.ceil(decodeGpus / spec.gpusPerNode);

	// Achieved figures at the sized decode batch.
	const decodeAtBatch = computeProfile({ ...decodePick.config, numGpus: decodeGpus });
	const clusterTps = decodeAtBatch.perUserTps * decodePick.batch * decodeReplicas;

	const prefillPool: PoolSizing & { msPerPrompt: number; promptsPerSecPerReplica: number } = {
		tp: prefillPick.tp,
		pp: prefillPick.pp,
		ep: prefillPick.ep,
		replicas: prefillReplicas,
		replicaGpus: prefillPick.replicaGpus,
		gpus: prefillGpus,
		nodes: prefillNodes,
		msPerPrompt: prefillPick.msPerPrompt,
		promptsPerSecPerReplica: promptsPerSecPerPrefillReplica,
		config: { ...prefillPick.config, numGpus: prefillGpus }
	};
	const decodePool: PoolSizing & {
		batchPerReplica: number;
		perUserTps: number;
		clusterTps: number;
	} = {
		tp: decodePick.tp,
		pp: decodePick.pp,
		ep: decodePick.ep,
		replicas: decodeReplicas,
		replicaGpus: decodePick.replicaGpus,
		gpus: decodeGpus,
		nodes: decodeNodes,
		batchPerReplica: decodePick.batch,
		perUserTps: decodeAtBatch.perUserTps,
		clusterTps,
		config: { ...decodePick.config, numGpus: decodeGpus }
	};

	return {
		feasible: true,
		provisionedConcurrency: provisioned,
		promptsPerSec,
		prefill: prefillPool,
		decode: decodePool,
		kvTransferMs,
		ttftMs,
		totalGpus: prefillGpus + decodeGpus,
		totalNodes: Math.ceil((prefillGpus + decodeGpus) / spec.gpusPerNode)
	};
}

function emptyDisagg(reason: string): DisaggResult {
	const zeroPool = {
		tp: 0,
		pp: 0,
		ep: 0,
		replicas: 0,
		replicaGpus: 0,
		gpus: 0,
		nodes: 0
	} as PoolSizing;
	return {
		feasible: false,
		reason,
		provisionedConcurrency: 0,
		promptsPerSec: 0,
		prefill: { ...zeroPool, msPerPrompt: 0, promptsPerSecPerReplica: 0, config: {} as Config },
		decode: { ...zeroPool, batchPerReplica: 0, perUserTps: 0, clusterTps: 0, config: {} as Config },
		kvTransferMs: 0,
		ttftMs: 0,
		totalGpus: 0,
		totalNodes: 0
	};
}

/** Size the chosen GPU and rank every GPU type by cluster size. */
export function solveWorkload(spec: WorkloadSpec): {
	chosen: SizeResult;
	comparison: SizeResult[];
} {
	const comparison = GPUS.map((g) => sizeForGpu(spec, g.id)).sort((a, b) => {
		if (a.feasible !== b.feasible) return a.feasible ? -1 : 1;
		return a.numGpus - b.numGpus;
	});
	const chosen = sizeForGpu(spec, spec.gpuId);
	return { chosen, comparison };
}
