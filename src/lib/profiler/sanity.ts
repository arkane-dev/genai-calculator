// Config sanity advisories: configurations that are valid (they compute) but are
// likely mistakes a non-expert wouldn't spot. These are soft hints, not the hard
// "doesn't fit" / infeasible checks — they nudge toward a sane setup.

import type { GpuSpec, WeightFormat } from './types';
import { FP8_NATIVE_GPUS } from './data';

export interface SanityInput {
	tp: number;
	gpusPerNode: number;
	fabricLabel: string;
	gpu: GpuSpec;
	formatTier: WeightFormat['tier'];
}

/** Soft warnings for a valid-but-questionable config. Empty when nothing's off. */
export function configWarnings(i: SanityInput): string[] {
	const w: string[] = [];

	// Tensor parallel that spills past the node runs over the inter-node fabric,
	// not the NVLink it assumes — usually a big, silent slowdown.
	if (i.tp > i.gpusPerNode) {
		w.push(
			`Tensor parallel (TP=${i.tp}) is larger than GPUs per node (${i.gpusPerNode}), so it spans nodes over ${i.fabricLabel} — far slower than the in-node NVLink that TP assumes. Keep TP ≤ node size and scale across nodes with pipeline or expert parallel.`
		);
	}

	if (i.formatTier === 'fp8' && !FP8_NATIVE_GPUS.has(i.gpu.id)) {
		w.push(
			`FP8 needs Hopper, Ada, or Blackwell tensor cores; ${i.gpu.name} would emulate it with no speed-up. Use BF16 here, or pick an FP8-capable GPU.`
		);
	}

	// fp4Tflops === 0 is the "no native FP4 path" convention on GpuSpec.
	if (i.formatTier === 'fp4' && i.gpu.fp4Tflops === 0) {
		w.push(
			`FP4 (NVFP4/MXFP4) needs Blackwell (B200/B300); ${i.gpu.name} can't run it natively — the model will still fit smaller, but not run faster.`
		);
	}

	return w;
}
