// Calculation chains for the Math tab: each computed quantity as a symbolic formula, the
// same formula with the current config's numbers substituted, and the result. Finals are
// sourced from computeProfile / computeTraining (so the tab can never drift from the tool);
// intermediates are recomputed here with the SAME exported constants + helpers.
import {
	computeProfile,
	MBU,
	MFU,
	FAMILY_MFU,
	ACT_BYTES,
	OVERHEAD_BYTES,
	moeSplit,
	nonExpertParams,
	activeExpertParams,
	computePeak
} from '$lib/profiler/calc';
import {
	computeTraining,
	MFU_TRAIN,
	GRAD_BYTES,
	QLORA_COMPUTE_TAX,
	OPT_BYTES,
	activeParamsOf
} from '$lib/training/calc';
import { MODELS_BY_ID, GPUS_BY_ID, WEIGHT_FORMATS_BY_ID } from '$lib/profiler/data';
import type { Config, ModelSpec, GpuSpec, WeightFormat } from '$lib/profiler/types';
import type { TrainConfig } from '$lib/training/types';

export interface Step {
	label: string;
	formula: string; // symbolic LaTeX
	substituted: string; // LaTeX with the current numbers
	result: string; // formatted result (plain text)
	note?: string;
}
export interface Section {
	title: string;
	intro?: string;
	steps: Step[];
}

const GiB = 1024 ** 3;
// --- number formatting for LaTeX / results ---
const sig = (x: number, n = 3) => {
	if (!isFinite(x) || x === 0) return '0';
	const d = Math.ceil(Math.log10(Math.abs(x)));
	const p = Math.max(0, n - d);
	return x.toFixed(Math.min(p, 6)).replace(/\.?0+$/, '');
};
const params = (p: number) =>
	p >= 1e12
		? `${sig(p / 1e12)}\\text{T}`
		: p >= 1e9
			? `${sig(p / 1e9)}\\text{B}`
			: `${sig(p / 1e6)}\\text{M}`;
const gb = (bytes: number) => `${sig(bytes / 1e9)}\\text{ GB}`;
const gbTxt = (bytes: number) => `${sig(bytes / 1e9)} GB`;
const ms = (s: number) => `${sig(s * 1000)}\\text{ ms}`;

/** Inference (Modelling / Workload) chain for the transformer LLM path. Non-transformer
 *  kinds return a single placeholder section (their math views are a later addition). */
export function inferenceChain(cfg: Config): Section[] {
	const m = MODELS_BY_ID.get(cfg.modelId)!;
	const g = GPUS_BY_ID.get(cfg.gpuId)!;
	const fmt = WEIGHT_FORMATS_BY_ID.get(cfg.weightFormatId) ?? WEIGHT_FORMATS_BY_ID.get('bf16')!;
	const kind = m.kind ?? 'transformer';
	if (kind === 'diffusion' && m.diffusion) return diffusionChain(cfg, m, g, fmt);
	if (kind === 'jepa' && m.jepa) return jepaChain(cfg, m, g, fmt);
	if (kind === 'encoder' && m.encoder) return encoderChain(cfg, m, g, fmt);
	if (kind === 'vla' && m.vla) return vlaChain(cfg, m, g, fmt);
	if (kind === 'asr' && m.asr) return asrChain(cfg, m, g, fmt);

	const p = computeProfile(cfg);
	const spec = p.specSpeedup ?? 1;
	const { tp, pp, ep, dp } = p.cluster;
	const cachedFrac = Math.max(0, Math.min(1, cfg.cachedPrefixFrac ?? 0));
	const imageTokens =
		m.vlm && (cfg.imagesPerRequest ?? 0) > 0
			? (cfg.imagesPerRequest ?? 0) * m.vlm.tokensPerImage
			: 0;
	const inputTokens = Math.max(1, cfg.inputTokens + imageTokens);
	const outputTokens = Math.max(1, cfg.outputTokens);
	const prefillTokens = Math.max(1, Math.round(inputTokens * (1 - cachedFrac)));
	const seqLen = inputTokens + outputTokens;
	const batch = Math.max(1, cfg.batchSize);
	const tokensPerSeq = cfg.phase === 'prefill' ? prefillTokens : 1;
	const tokensInStep = batch * tokensPerSeq;
	const wb = fmt.bitsPerWeight / 8;
	const kvb = cfg.kvBits / 8;
	const layerShard = tp * pp;

	// intermediates (recomputed with the shared constants)
	const kvPerTokenLayer = m.mla ? m.mla.kvLatentDim * kvb : 2 * m.numKvHeads * m.headDim * kvb;
	const kvLayers = m.kvLayers ?? m.numLayers;
	const kvShard = (m.mla ? 1 : Math.min(tp, m.numKvHeads)) * pp;
	const bytesPerStep = p.perGpu.weights + p.perGpu.kv; // (+2·mamba, folded into the tool's value)
	const peakBW = g.memBandwidthTBs * 1e12;
	const memTime = bytesPerStep / (peakBW * MBU);
	const ne = nonExpertParams(m);
	const ae = activeExpertParams(m);
	const flopsPerStep = 2 * tokensInStep * (ne / (tp * pp) + ae / (ep * pp));
	const peakFlops = computePeak(g, fmt.tier);
	const computeTime = flopsPerStep / (peakFlops * MFU * (g.mfuFactor ?? 1));
	const prefillFlops = 2 * batch * prefillTokens * (ne / (tp * pp) + ae / (ep * pp));

	const setup: Step[] = [
		{
			label: 'Model parameters',
			formula: `P`,
			substituted:
				params(m.params) + (m.moe ? ` \\;(\\text{active } ${params(p.activeParams)})` : ''),
			result: m.moe
				? `${sig(m.params / 1e9)}B total, ${sig(p.activeParams / 1e9)}B active`
				: `${sig(m.params / 1e9)}B`
		},
		{
			label: 'GPU peak (compute · bandwidth)',
			formula: `\\pi_{\\text{flops}},\\; \\pi_{\\text{bw}}`,
			substituted: `${sig(peakFlops / 1e12)}\\text{ TFLOP/s},\\; ${sig(g.memBandwidthTBs)}\\text{ TB/s}`,
			result: `${sig(peakFlops / 1e12)} TFLOP/s (${fmt.tier}), ${sig(g.memBandwidthTBs)} TB/s`
		},
		{
			label: 'Parallelism',
			formula: `t_p,\\; p_p,\\; e_p,\\; d_p`,
			substituted: `${tp},\\; ${pp},\\; ${ep},\\; ${dp}`,
			result: `TP ${tp} · PP ${pp} · EP ${ep} · DP ${dp} (of ${p.cluster.numGpus} GPUs)`
		},
		{
			label: 'Sequence',
			formula: `L = s_{\\text{in}} + s_{\\text{out}}`,
			substituted: `${inputTokens} + ${outputTokens} = ${seqLen}`,
			result: `${seqLen} tokens${imageTokens ? ` (incl. ${imageTokens} image tokens)` : ''}`
		}
	];

	const memWeights: Step = m.moe
		? {
				label: 'Weights (MoE: non-expert + experts)',
				formula: `W = \\frac{P_{\\text{ne}}\\, b_w}{t_p p_p} + \\frac{P_{\\text{exp}}\\, b_w}{e_p p_p}`,
				substituted: `\\frac{${params(moeSplit(m).nonExpert)}\\cdot ${wb}}{${tp}\\cdot${pp}} + \\frac{${params(moeSplit(m).expert)}\\cdot ${wb}}{${ep}\\cdot${pp}} = ${gb(p.perGpu.weights)}`,
				result: gbTxt(p.perGpu.weights)
			}
		: {
				label: 'Weights',
				formula: `W = \\dfrac{P \\cdot b_w}{t_p \\cdot p_p}`,
				substituted: `\\dfrac{${params(m.params)} \\cdot ${wb}}{${tp} \\cdot ${pp}} = ${gb(p.perGpu.weights)}`,
				result: gbTxt(p.perGpu.weights),
				note: `b_w = ${wb} bytes/param (${fmt.label})`
			};

	const memory: Step[] = [
		memWeights,
		{
			label: 'KV per token · layer',
			formula: m.mla
				? `k_{tl} = d_{\\text{lat}} \\cdot b_{kv}`
				: `k_{tl} = 2 \\, n_{kv} \\, d_h \\, b_{kv}`,
			substituted: m.mla
				? `${m.mla.kvLatentDim} \\cdot ${kvb} = ${sig(kvPerTokenLayer)}\\text{ B}`
				: `2 \\cdot ${m.numKvHeads} \\cdot ${m.headDim} \\cdot ${kvb} = ${sig(kvPerTokenLayer)}\\text{ B}`,
			result: `${sig(kvPerTokenLayer)} B`,
			note: m.mla ? 'MLA: one compressed latent (no per-head K/V)' : `GQA: ${m.numKvHeads} KV heads`
		},
		{
			label: 'KV cache',
			formula: `\\text{KV} = \\frac{n_{\\text{kvL}} \\, L \\, k_{tl}}{s_{kv}} \\cdot w_{\\text{pad}} \\cdot B`,
			substituted: `\\frac{${kvLayers}\\cdot ${seqLen}\\cdot ${sig(kvPerTokenLayer)}}{${kvShard}} \\cdot w_{\\text{pad}} \\cdot ${batch} = ${gb(p.perGpu.kv)}`,
			result: gbTxt(p.perGpu.kv),
			note: `${cfg.kvAllocation} allocation, KV on ${kvLayers}/${m.numLayers} layers, sharded ${kvShard}×`
		},
		{
			label: 'Activations',
			formula: `A = \\dfrac{2 B \\, L \\, h \\, b_a}{t_p p_p}`,
			substituted: `\\dfrac{2\\cdot ${batch}\\cdot ${seqLen}\\cdot ${m.hiddenSize}\\cdot ${ACT_BYTES}}{${layerShard}} = ${gb(p.perGpu.activations)}`,
			result: gbTxt(p.perGpu.activations)
		},
		{
			label: 'Overhead',
			formula: `O`,
			substituted: `${sig(OVERHEAD_BYTES / GiB)}\\text{ GiB}`,
			result: gbTxt(p.perGpu.overhead),
			note: 'CUDA context + framework + fragmentation'
		},
		{
			label: 'Total vs capacity',
			formula: `U = W + \\text{KV} + A + O \\;\\lessgtr\\; C`,
			substituted: `${gb(p.perGpu.used)} \\;${p.perGpu.fits ? '\\leq' : '>'}\\; ${gb(p.perGpu.capacity)}`,
			result: `${gbTxt(p.perGpu.used)} / ${gbTxt(p.perGpu.capacity)} — ${p.perGpu.fits ? 'fits' : 'does NOT fit'}`
		}
	];

	const throughput: Step[] = [
		{
			label: 'Bytes read per step (decode: weights + KV)',
			formula: `b_{\\text{step}} = W + \\text{KV}`,
			substituted: `${gb(p.perGpu.weights)} + ${gb(p.perGpu.kv)} = ${gb(bytesPerStep)}`,
			result: gbTxt(bytesPerStep)
		},
		{
			label: 'Memory time',
			formula: `t_{\\text{mem}} = \\dfrac{b_{\\text{step}}}{\\pi_{\\text{bw}} \\cdot \\text{MBU}}`,
			substituted: `\\dfrac{${gb(bytesPerStep)}}{${sig(g.memBandwidthTBs)}\\text{ TB/s} \\cdot ${MBU}} = ${ms(memTime)}`,
			result: `${sig(memTime * 1000)} ms`,
			note: `MBU = ${MBU}`
		},
		{
			label: 'FLOPs per step',
			formula: `f_{\\text{step}} = 2\\, n_{\\text{tok}} \\left(\\frac{P_{\\text{ne}}}{t_p p_p} + \\frac{P_{\\text{exp,act}}}{e_p p_p}\\right)`,
			substituted: `2\\cdot ${tokensInStep}\\cdot (\\ldots) = ${sig(flopsPerStep / 1e9)}\\text{ GFLOP}`,
			result: `${sig(flopsPerStep / 1e12, 3)} TFLOP (${tokensInStep} tokens/step)`
		},
		{
			label: 'Compute time',
			formula: `t_{\\text{cmp}} = \\dfrac{f_{\\text{step}}}{\\pi_{\\text{flops}} \\cdot \\text{MFU} \\cdot \\phi}`,
			substituted: `\\dfrac{${sig(flopsPerStep / 1e12)}\\text{ T}}{${sig(peakFlops / 1e12)}\\cdot ${MFU}\\cdot ${sig(g.mfuFactor ?? 1)}} = ${ms(computeTime)}`,
			result: `${sig(computeTime * 1000)} ms`,
			note: `MFU = ${MFU}, per-GPU factor φ = ${sig(g.mfuFactor ?? 1)}`
		},
		{
			label: 'Step time (roofline + comm)',
			formula: `t_{\\text{step}} = \\max(t_{\\text{mem}}, t_{\\text{cmp}}) + t_{\\text{comm}}`,
			substituted: `\\max(${ms(memTime)}, ${ms(computeTime)}) + t_{\\text{comm}} = ${ms(p.stepTimeMs / 1000)}`,
			result: `${sig(p.stepTimeMs)} ms`,
			note: `bottleneck: ${p.bottleneck}`
		},
		{
			label: 'Aggregate throughput',
			formula: `T = \\dfrac{n_{\\text{tok}}}{t_{\\text{step}}} \\cdot d_p${spec > 1 ? ' \\cdot \\sigma' : ''}`,
			substituted: `\\dfrac{${tokensInStep}}{${sig(p.stepTimeMs / 1000)}\\text{ s}} \\cdot ${dp}${spec > 1 ? ` \\cdot ${sig(spec)}` : ''} = ${sig(p.throughputTps)}\\text{ tok/s}`,
			result: `${sig(p.throughputTps)} tok/s${spec > 1 ? ` (spec-decode ${sig(spec)}×)` : ''}`
		},
		{
			label: 'Per-user throughput',
			formula: `T_u = \\dfrac{s_{\\text{tok}}}{p_p \\, t_{\\text{step}} + (p_p-1) t_{pp}}${spec > 1 ? '\\cdot \\sigma' : ''}`,
			substituted: `\\approx ${sig(p.perUserTps)}\\text{ tok/s}`,
			result: `${sig(p.perUserTps)} tok/s per user`
		}
	];

	const latency: Step[] = [
		{
			label: 'Prefill FLOPs (uncached input)',
			formula: `f_{\\text{pre}} = 2 B \\, s_{\\text{pre}} \\left(\\frac{P_{\\text{ne}}}{t_p p_p} + \\frac{P_{\\text{exp,act}}}{e_p p_p}\\right)`,
			substituted: `2\\cdot ${batch}\\cdot ${prefillTokens}\\cdot(\\ldots) = ${sig(prefillFlops / 1e12)}\\text{ TFLOP}`,
			result: `${sig(prefillFlops / 1e12)} TFLOP${cachedFrac > 0 ? ` (${Math.round(cachedFrac * 100)}% prefix cached)` : ''}`
		},
		{
			label: 'Time to first token',
			formula: `\\text{TTFT} = \\dfrac{f_{\\text{pre}}}{\\pi_{\\text{flops}} \\cdot \\text{MFU} \\cdot \\phi}`,
			substituted: `\\dfrac{${sig(prefillFlops / 1e12)}\\text{ T}}{${sig(peakFlops / 1e12)}\\cdot ${MFU}\\cdot ${sig(g.mfuFactor ?? 1)}} = ${sig(p.ttftMs)}\\text{ ms}`,
			result: `${sig(p.ttftMs)} ms`
		}
	];

	return [
		{ title: 'Setup', steps: setup },
		{
			title: 'Memory (per GPU)',
			intro: 'What one GPU must hold for this replica.',
			steps: memory
		},
		{
			title: 'Throughput (roofline)',
			intro:
				'A step is bounded by the slower of memory traffic and compute, plus in-stage collectives.',
			steps: throughput
		},
		{ title: 'Latency (TTFT)', steps: latency }
	];
}

/** Training / fine-tuning memory + step-time chain. */
export function trainingChain(cfg: TrainConfig): Section[] {
	const m = MODELS_BY_ID.get(cfg.modelId)!;
	const fmt = WEIGHT_FORMATS_BY_ID.get(cfg.weightFormatId) ?? WEIGHT_FORMATS_BY_ID.get('bf16')!;
	const t = computeTraining(cfg);
	const isLora = cfg.method === 'lora' || cfg.method === 'qlora';
	const { dp, replicaGpus, trainableParams } = t;
	const active = activeParamsOf(m);
	let flopsFactor = isLora ? 5 : 6;
	if (cfg.activationCheckpointing) flopsFactor += 2;

	const setup: Step[] = [
		{
			label: 'Method · trainable params',
			formula: `P_{\\text{train}}`,
			substituted: isLora
				? `\\text{LoRA}(r=${cfg.loraRank}) = ${params(trainableParams)}`
				: `\\text{full} = ${params(trainableParams)}`,
			result: `${cfg.method.toUpperCase()} — ${sig(trainableParams / 1e6)}M trainable of ${sig(m.params / 1e9)}B`
		},
		{
			label: 'Parallelism',
			formula: `t_p \\cdot p_p = g_{\\text{rep}},\\; d_p`,
			substituted: `${cfg.tp}\\cdot ${cfg.pp} = ${replicaGpus},\\; d_p = ${dp}`,
			result: `replica ${replicaGpus} GPUs · DP ${dp} · ZeRO-${cfg.zeroStage}`
		}
	];

	const memory: Step[] = [
		{
			label: isLora ? 'Base weights (frozen)' : 'Weights',
			formula: `W`,
			substituted: `${gb(t.perGpu.weights)}`,
			result: gbTxt(t.perGpu.weights),
			note:
				cfg.method === 'qlora'
					? 'QLoRA: base frozen in 4-bit'
					: `${fmt.label}, sharded ${replicaGpus}×${cfg.zeroStage >= 3 ? `·${dp} (ZeRO-3)` : ''}`
		},
		{
			label: 'Gradients (trainable only)',
			formula: `G = \\dfrac{P_{\\text{train}} \\cdot ${GRAD_BYTES}}{g_{\\text{rep}}${cfg.zeroStage >= 2 ? ' \\cdot d_p' : ''}}`,
			substituted: `\\dfrac{${params(trainableParams)}\\cdot ${GRAD_BYTES}}{${replicaGpus}${cfg.zeroStage >= 2 ? `\\cdot ${dp}` : ''}} = ${gb(t.perGpu.gradients)}`,
			result: gbTxt(t.perGpu.gradients)
		},
		{
			label: 'Optimizer states',
			formula: `S = \\dfrac{P_{\\text{train}} \\cdot o}{g_{\\text{rep}}${cfg.zeroStage >= 1 ? ' \\cdot d_p' : ''}}`,
			substituted: cfg.cpuOffload
				? `0 \\;(\\text{CPU offload})`
				: `\\dfrac{${params(trainableParams)}\\cdot ${OPT_BYTES[cfg.optimizer]}}{${replicaGpus}${cfg.zeroStage >= 1 ? `\\cdot ${dp}` : ''}} = ${gb(t.perGpu.optimizer)}`,
			result: gbTxt(t.perGpu.optimizer),
			note: `${cfg.optimizer} = ${OPT_BYTES[cfg.optimizer]} B/param${cfg.cpuOffload ? ', offloaded to host RAM' : ''}`
		},
		{
			label: 'Activations',
			formula: `A = \\dfrac{a_{\\text{layer}} \\cdot n_L / p_p}{t_p}${cfg.activationCheckpointing ? ' + 3\\,a_{\\text{full}}' : ''}`,
			substituted: `${gb(t.perGpu.activations)}`,
			result: gbTxt(t.perGpu.activations),
			note: cfg.activationCheckpointing
				? 'checkpointing: stored inputs + recompute peak'
				: 'no checkpointing (full activations)'
		},
		{
			label: 'Total vs capacity',
			formula: `U = W + G + S + A + O \\;\\lessgtr\\; C`,
			substituted: `${gb(t.perGpu.used)} \\;${t.perGpu.fits ? '\\leq' : '>'}\\; ${gb(t.perGpu.capacity)}`,
			result: `${gbTxt(t.perGpu.used)} / ${gbTxt(t.perGpu.capacity)} — ${t.perGpu.fits ? 'fits' : 'does NOT fit'}`
		}
	];

	const time: Step[] = [
		{
			label: 'FLOPs per token (fwd + bwd)',
			formula: `f_{\\text{tok}} = ${flopsFactor} \\cdot P_{\\text{act}}${cfg.method === 'qlora' ? ` \\cdot ${QLORA_COMPUTE_TAX}` : ''}`,
			substituted: `${flopsFactor}\\cdot ${params(active)}${cfg.method === 'qlora' ? `\\cdot ${QLORA_COMPUTE_TAX}` : ''}`,
			result: `${flopsFactor}·N/token${isLora ? ' (LoRA skips frozen weight-grads)' : ''}${cfg.activationCheckpointing ? ', +2 for recompute' : ''}`
		},
		{
			label: 'Global batch',
			formula: `n_{\\text{gb}} = b \\cdot s \\cdot g_{\\text{acc}} \\cdot d_p`,
			substituted: `${cfg.microBatchSize}\\cdot ${cfg.seqLen}\\cdot ${cfg.gradAccum}\\cdot ${dp} = ${sig(t.globalBatchTokens / 1e3)}\\text{k}`,
			result: `${sig(t.globalBatchTokens / 1e3)}k tokens/step`
		},
		{
			label: 'Step time',
			formula: `t_{\\text{step}} = t_{\\text{cmp}} + t_{\\text{sync}} + t_{\\text{offload}}`,
			substituted: `${sig(t.stepTimeMs)}\\text{ ms}`,
			result: `${sig(t.stepTimeMs)} ms`,
			note: `bottleneck: ${t.bottleneck}; MFU ≈ ${sig(t.mfu)} (train peak ${MFU_TRAIN})`
		},
		{
			label: 'Throughput',
			formula: `T = \\dfrac{n_{\\text{gb}}}{t_{\\text{step}}}`,
			substituted: `\\dfrac{${sig(t.globalBatchTokens / 1e3)}\\text{k}}{${sig(t.stepTimeMs)}\\text{ ms}} = ${sig(t.tokensPerSec)}\\text{ tok/s}`,
			result: `${sig(t.tokensPerSec)} tok/s`
		},
		{
			label: 'Time to train',
			formula: `t_{\\text{train}} = \\dfrac{D \\cdot E}{T}`,
			substituted: `\\dfrac{${sig(cfg.datasetTokens / 1e9)}\\text{B}\\cdot ${cfg.epochs}}{${sig(t.tokensPerSec)}\\text{ tok/s}} = ${sig(t.timeToTrainHours)}\\text{ h}`,
			result: `${sig(t.timeToTrainHours)} hours (${sig(cfg.datasetTokens / 1e9)}B tokens × ${cfg.epochs} epochs)`
		}
	];

	return [
		{ title: 'Setup', steps: setup },
		{
			title: 'Memory (per GPU)',
			intro:
				'Training holds weights + gradients + optimizer states + activations — the 16 B/param story for full fine-tuning.',
			steps: memory
		},
		{ title: 'Compute & time', steps: time }
	];
}

// ---- non-transformer families (dense TP/PP, one forward pass; finals from the tool) ----
const DIFFUSION_FLOP_CORRECTION = 1.6; // matches calc.ts (UNet/DiT linear-FLOP undercount)

function commonSetup(
	m: ModelSpec,
	g: GpuSpec,
	fmt: WeightFormat,
	peakFlops: number,
	tp: number,
	pp: number,
	dp: number,
	numGpus: number
): Step[] {
	return [
		{
			label: 'Parameters',
			formula: `P`,
			substituted: params(m.params),
			result: `${sig(m.params / 1e9)}B`
		},
		{
			label: 'GPU peak (compute · bandwidth)',
			formula: `\\pi_{\\text{flops}},\\;\\pi_{\\text{bw}}`,
			substituted: `${sig(peakFlops / 1e12)}\\text{ TFLOP/s},\\;${sig(g.memBandwidthTBs)}\\text{ TB/s}`,
			result: `${sig(peakFlops / 1e12)} TFLOP/s (${fmt.tier}), ${sig(g.memBandwidthTBs)} TB/s`
		},
		{
			label: 'Parallelism',
			formula: `t_p,\\;p_p,\\;d_p`,
			substituted: `${tp},\\;${pp},\\;${dp}`,
			result: `TP ${tp} · PP ${pp} · DP ${dp} (of ${numGpus} GPUs)`
		}
	];
}

function totalStep(pg: { used: number; capacity: number; fits: boolean }, terms: string): Step {
	return {
		label: 'Total vs capacity',
		formula: `U = ${terms} \\;\\lessgtr\\; C`,
		substituted: `${gb(pg.used)} \\;${pg.fits ? '\\leq' : '>'}\\; ${gb(pg.capacity)}`,
		result: `${gbTxt(pg.used)} / ${gbTxt(pg.capacity)} — ${pg.fits ? 'fits' : 'does NOT fit'}`
	};
}

/** Diffusion (UNet / DiT / MMDiT): `steps` denoising passes over the latent, no KV. */
function diffusionChain(cfg: Config, m: ModelSpec, g: GpuSpec, fmt: WeightFormat): Section[] {
	const ds = m.diffusion!;
	const p = computeProfile(cfg);
	const { tp, pp, dp, numGpus } = p.cluster;
	const layerShard = tp * pp;
	const steps = Math.max(1, cfg.steps ?? ds.defaultSteps);
	const res = Math.max(64, cfg.resolution ?? ds.defaultResolution);
	const frames = Math.max(1, cfg.frames ?? ds.frames ?? 1);
	const cfgOn = cfg.guidance ?? ds.cfg;
	const cfgFactor = cfgOn ? 2 : 1;
	const batch = Math.max(1, cfg.batchSize);
	const side = res / ds.vaeDownsample / ds.patch;
	const latentTokens = Math.max(1, Math.round(side * side * (ds.aspect ?? 1) * frames));
	const effTokens = batch * latentTokens * cfgFactor;
	const wb = fmt.bitsPerWeight / 8;
	const peakFlops = computePeak(g, fmt.tier);
	const linCorr = ds.arch === 'mmdit' ? 1 : DIFFUSION_FLOP_CORRECTION;
	const attnFlops =
		ds.arch === 'unet'
			? 0
			: 4 * m.numLayers * latentTokens * latentTokens * m.hiddenSize * (batch * cfgFactor);
	const flopsPerStep = (2 * m.params * effTokens * linCorr + attnFlops) / layerShard;
	const computeTime = flopsPerStep / (peakFlops * MFU * (g.mfuFactor ?? 1));
	const memTime = (p.perGpu.weights + p.perGpu.activations) / (g.memBandwidthTBs * 1e12 * MBU);
	const d = p.diffusion!;

	return [
		{
			title: 'Setup',
			steps: [
				...commonSetup(m, g, fmt, peakFlops, tp, pp, dp, numGpus),
				{
					label: 'Latent tokens per image',
					formula: `n_{\\text{lat}} = \\left(\\frac{r}{f_{\\text{vae}}\\, p}\\right)^2 \\cdot a \\cdot F`,
					substituted: `\\left(\\frac{${res}}{${ds.vaeDownsample}\\cdot ${ds.patch}}\\right)^2 \\cdot ${ds.aspect ?? 1}\\cdot ${frames} = ${latentTokens}`,
					result: `${latentTokens} tokens (${res}px, ${ds.arch})`,
					note: cfgOn ? 'CFG on → cond + uncond (×2 forwards)' : 'no CFG'
				}
			]
		},
		{
			title: 'Memory (per GPU)',
			steps: [
				{
					label: 'Weights (denoiser + encoders)',
					formula: `W = \\frac{P\\, b_w}{t_p p_p} + \\frac{(P_{\\text{txt}}+P_{\\text{vae}})\\, b_w}{t_p}`,
					substituted: `\\frac{${params(m.params)}\\cdot ${wb}}{${layerShard}} + \\frac{${params((ds.textEncoderParams ?? 0) + (ds.vaeParams ?? 0))}\\cdot ${wb}}{${tp}} = ${gb(p.perGpu.weights)}`,
					result: gbTxt(p.perGpu.weights)
				},
				{
					label: 'Activations',
					formula: `A = \\frac{2\\, n_{\\text{lat}}\\, c\\, h\\, b_a}{t_p p_p} \\cdot B`,
					substituted: `\\frac{2\\cdot ${latentTokens}\\cdot ${cfgFactor}\\cdot ${m.hiddenSize}\\cdot ${ACT_BYTES}}{${layerShard}}\\cdot ${batch} = ${gb(p.perGpu.activations)}`,
					result: gbTxt(p.perGpu.activations),
					note: 'no KV cache (not autoregressive)'
				},
				totalStep(p.perGpu, 'W + A + O')
			]
		},
		{
			title: 'Roofline & throughput',
			intro: `Each of the ${steps} denoising steps is a full forward over the latent.`,
			steps: [
				{
					label: 'FLOPs per step',
					formula: `f_{\\text{step}} = \\frac{2 P\\, n_{\\text{tok}}\\, \\kappa + f_{\\text{attn}}}{t_p p_p}`,
					substituted: `\\approx ${sig(flopsPerStep / 1e12)}\\text{ TFLOP}`,
					result: `${sig(flopsPerStep / 1e12)} TFLOP/step`,
					note: `linear-FLOP correction κ = ${linCorr}${ds.arch !== 'unet' ? ' + quadratic attention' : ' (UNet: attention in effective patch)'}`
				},
				{
					label: 'Compute vs memory time',
					formula: `t_{\\text{step}} = \\max\\!\\left(\\frac{f_{\\text{step}}}{\\pi_{\\text{flops}}\\text{MFU}\\,\\phi},\\;\\frac{W+A}{\\pi_{\\text{bw}}\\text{MBU}}\\right)`,
					substituted: `\\max(${ms(computeTime)}, ${ms(memTime)}) = ${ms(d.stepTimeMs / 1000)}`,
					result: `${sig(d.stepTimeMs)} ms/step — ${p.bottleneck}-bound`
				},
				{
					label: 'Seconds per image',
					formula: `t_{\\text{img}} = \\text{steps} \\cdot p_p\\, t_{\\text{step}}`,
					substituted: `${steps}\\cdot ${ms(d.stepTimeMs / 1000)} = ${sig(d.secPerImage)}\\text{ s}`,
					result: `${sig(d.secPerImage)} s/image`
				},
				{
					label: 'Throughput',
					formula: `R = \\frac{B\\, d_p}{t_{\\text{img}}}`,
					substituted: `\\frac{${batch}\\cdot ${dp}}{${sig(d.secPerImage)}} = ${sig(d.imagesPerSec)}\\text{ img/s}`,
					result: `${sig(d.imagesPerSec)} images/s (${sig(d.imagesPerSec * 60)}/min)`
				}
			]
		}
	];
}

/** JEPA (V-JEPA / V-JEPA 2): one ViT encoder forward over video patches, no KV. */
function jepaChain(cfg: Config, m: ModelSpec, g: GpuSpec, fmt: WeightFormat): Section[] {
	const js = m.jepa!;
	const p = computeProfile(cfg);
	const { tp, pp, dp, numGpus } = p.cluster;
	const layerShard = tp * pp;
	const res = Math.max(64, cfg.resolution ?? js.defaultResolution);
	const frames = Math.max(js.tubelet, cfg.frames ?? js.defaultFrames);
	const batch = Math.max(1, cfg.batchSize);
	const spatial = Math.round(res / js.patch) ** 2;
	const temporal = Math.max(1, Math.round(frames / js.tubelet));
	const tokens = Math.max(1, spatial * temporal);
	const effTokens = batch * tokens;
	const peakFlops = computePeak(g, fmt.tier);
	const flopsFwd =
		(2 * m.params * effTokens + 4 * m.numLayers * tokens * tokens * m.hiddenSize * batch) /
		layerShard;
	const rollout = js.rolloutSteps ?? 0;
	const computeTime = flopsFwd / (peakFlops * MFU * (g.mfuFactor ?? 1) * FAMILY_MFU.jepa);
	const memTime = (p.perGpu.weights + p.perGpu.activations) / (g.memBandwidthTBs * 1e12 * MBU);
	const j = p.jepa!;

	return [
		{
			title: 'Setup',
			steps: [
				...commonSetup(m, g, fmt, peakFlops, tp, pp, dp, numGpus),
				{
					label: 'Video patch tokens',
					formula: `n = \\left(\\frac{r}{p}\\right)^2 \\cdot \\frac{F}{\\tau}`,
					substituted: `${spatial}\\cdot ${temporal} = ${tokens}`,
					result: `${tokens} tokens (${frames} frames @ ${res}px, tubelet ${js.tubelet})`
				}
			]
		},
		{
			title: 'Memory (per GPU)',
			steps: [
				{
					label: 'Weights (encoder + predictor)',
					formula: `W = \\frac{(P + P_{\\text{pred}})\\, b_w}{t_p p_p}`,
					substituted: `\\frac{${params(m.params + (js.predictorParams ?? 0))}\\cdot ${fmt.bitsPerWeight / 8}}{${layerShard}} = ${gb(p.perGpu.weights)}`,
					result: gbTxt(p.perGpu.weights)
				},
				{
					label: 'Activations',
					formula: `A = \\frac{2\\, n\\, h\\, b_a}{t_p p_p} \\cdot B`,
					substituted: `= ${gb(p.perGpu.activations)}`,
					result: gbTxt(p.perGpu.activations)
				},
				totalStep(p.perGpu, 'W + A + O')
			]
		},
		{
			title: 'Roofline & throughput',
			steps: [
				{
					label: 'FLOPs per clip (linear + attention)',
					formula: `f = \\frac{2 P n + 4 n_L\\, n^2 h\\, B${rollout ? ' + \\text{rollout}' : ''}}{t_p p_p}`,
					substituted: `\\approx ${sig(flopsFwd / 1e12)}\\text{ TFLOP}`,
					result: `${sig(flopsFwd / 1e12)} TFLOP${rollout ? ` (+${rollout}-step predictor rollout)` : ''}`,
					note: 'quadratic attention dominates at 64 frames'
				},
				{
					label: 'Compute time (with JEPA family factor)',
					formula: `t_{\\text{cmp}} = \\frac{f}{\\pi_{\\text{flops}}\\,\\text{MFU}\\,\\phi\\cdot \\mu_{\\text{jepa}}}`,
					substituted: `\\frac{${sig(flopsFwd / 1e12)}\\text{ T}}{${sig(peakFlops / 1e12)}\\cdot ${MFU}\\cdot ${sig(g.mfuFactor ?? 1)}\\cdot ${FAMILY_MFU.jepa}} = ${ms(computeTime)}`,
					result: `${sig(computeTime * 1000)} ms`,
					note: `μ_jepa = ${FAMILY_MFU.jepa} (ViT forwards run ~2× slower than the LLM roofline)`
				},
				{
					label: 'Seconds per clip',
					formula: `t_{\\text{clip}} = \\max(t_{\\text{cmp}}, t_{\\text{mem}}) \\cdot p_p`,
					substituted: `\\max(${ms(computeTime)}, ${ms(memTime)}) = ${sig(j.secPerClip * 1000)}\\text{ ms}`,
					result: `${sig(j.secPerClip * 1000)} ms/clip`
				},
				{
					label: 'Throughput',
					formula: `R = \\frac{B\\, d_p}{t_{\\text{clip}}}`,
					substituted: `= ${sig(j.clipsPerSec)}\\text{ clips/s}`,
					result: `${sig(j.clipsPerSec)} clips/s`
				}
			]
		}
	];
}

/** Encoder (embeddings / rerankers): one bidirectional forward, no KV. */
function encoderChain(cfg: Config, m: ModelSpec, g: GpuSpec, fmt: WeightFormat): Section[] {
	const es = m.encoder!;
	const p = computeProfile(cfg);
	const { tp, pp, dp, numGpus } = p.cluster;
	const layerShard = tp * pp;
	const tokens = Math.max(1, Math.min(cfg.inputTokens || es.defaultSeqLen, es.maxSeqLen));
	const batch = Math.max(1, cfg.batchSize);
	const effTokens = batch * tokens;
	const peakFlops = computePeak(g, fmt.tier);
	const flopsFwd =
		(2 * m.params * effTokens + 4 * m.numLayers * tokens * tokens * m.hiddenSize * batch) /
		layerShard;
	const computeTime = flopsFwd / (peakFlops * MFU * (g.mfuFactor ?? 1) * FAMILY_MFU.encoder);
	const e = p.encoder!;

	return [
		{
			title: 'Setup',
			steps: [
				...commonSetup(m, g, fmt, peakFlops, tp, pp, dp, numGpus),
				{
					label: 'Sequence length',
					formula: `n`,
					substituted: `${tokens}`,
					result: `${tokens} tokens (${es.task})`
				}
			]
		},
		{
			title: 'Memory (per GPU)',
			steps: [
				{
					label: 'Weights',
					formula: `W = \\frac{P\\, b_w}{t_p p_p}`,
					substituted: `\\frac{${params(m.params)}\\cdot ${fmt.bitsPerWeight / 8}}{${layerShard}} = ${gb(p.perGpu.weights)}`,
					result: gbTxt(p.perGpu.weights)
				},
				{
					label: 'Activations',
					formula: `A = \\frac{2\\, n\\, h\\, b_a}{t_p p_p} \\cdot B`,
					substituted: `= ${gb(p.perGpu.activations)}`,
					result: gbTxt(p.perGpu.activations),
					note: 'no KV cache (single bidirectional pass)'
				},
				totalStep(p.perGpu, 'W + A + O')
			]
		},
		{
			title: 'Roofline & throughput',
			steps: [
				{
					label: 'FLOPs per forward (linear + attention)',
					formula: `f = \\frac{2 P n_{\\text{tok}} + 4 n_L\\, n^2 h\\, B}{t_p p_p}`,
					substituted: `\\approx ${sig(flopsFwd / 1e12)}\\text{ TFLOP}`,
					result: `${sig(flopsFwd / 1e12)} TFLOP`
				},
				{
					label: 'Compute time (with encoder family factor)',
					formula: `t_{\\text{cmp}} = \\frac{f}{\\pi_{\\text{flops}}\\,\\text{MFU}\\,\\phi\\cdot \\mu_{\\text{enc}}}`,
					substituted: `\\frac{${sig(flopsFwd / 1e12)}\\text{ T}}{${sig(peakFlops / 1e12)}\\cdot ${MFU}\\cdot ${sig(g.mfuFactor ?? 1)}\\cdot ${FAMILY_MFU.encoder}} = ${ms(computeTime)}`,
					result: `${sig(computeTime * 1000)} ms`,
					note: `μ_enc = ${FAMILY_MFU.encoder} (small bidirectional GEMMs run hotter than LLM prefill)`
				},
				{
					label: 'Seconds per document',
					formula: `t_{\\text{doc}} = \\max(t_{\\text{cmp}}, t_{\\text{mem}}) \\cdot p_p`,
					substituted: `= ${sig(e.secPerDoc * 1000)}\\text{ ms}`,
					result: `${sig(e.secPerDoc * 1000)} ms/doc`
				},
				{
					label: 'Throughput',
					formula: `R = \\frac{B\\, d_p}{t_{\\text{doc}}}`,
					substituted: `= ${sig(e.docsPerSec)}\\text{ docs/s}`,
					result: `${sig(e.docsPerSec)} docs/s (${sig(p.throughputTps)} tok/s)`
				}
			]
		}
	];
}

/** VLA (robotics): vision encode + backbone prefill + flow-matching action rollout. */
function vlaChain(cfg: Config, m: ModelSpec, g: GpuSpec, fmt: WeightFormat): Section[] {
	const vs = m.vla!;
	const p = computeProfile(cfg);
	const { tp, pp, dp, numGpus } = p.cluster;
	const layerShard = tp * pp;
	const batch = Math.max(1, cfg.batchSize);
	const imgTokens = vs.camerasPerObs * vs.tokensPerImage;
	const inputTokens = imgTokens + 1 + 48;
	const peakFlops = computePeak(g, fmt.tier);
	const backboneLinear = 2 * m.params * batch * inputTokens;
	const backboneAttn = 4 * m.numLayers * inputTokens * inputTokens * m.hiddenSize * batch;
	const visionForward = 2 * vs.visionParams * imgTokens * batch;
	const expertForward =
		vs.flowSteps *
		(2 * vs.expertParams * vs.chunkSize * batch +
			4 * m.numLayers * vs.chunkSize * vs.chunkSize * m.hiddenSize * batch);
	const flops = (backboneLinear + backboneAttn + visionForward + expertForward) / layerShard;
	const computeTime = flops / (peakFlops * MFU * (g.mfuFactor ?? 1));
	const v = p.vla!;

	return [
		{
			title: 'Setup',
			steps: [
				...commonSetup(m, g, fmt, peakFlops, tp, pp, dp, numGpus),
				{
					label: 'Input tokens per observation',
					formula: `n = c\\, n_{\\text{img}} + 1 + 48`,
					substituted: `${vs.camerasPerObs}\\cdot ${vs.tokensPerImage} + 1 + 48 = ${inputTokens}`,
					result: `${inputTokens} tokens (${vs.camerasPerObs} cameras + state + text)`
				}
			]
		},
		{
			title: 'Memory (per GPU)',
			steps: [
				{
					label: 'Weights (backbone + expert + vision)',
					formula: `W = \\frac{(P + P_{\\text{exp}})\\, b_w}{t_p p_p} + \\frac{P_{\\text{vis}}\\, b_w}{t_p}`,
					substituted: `= ${gb(p.perGpu.weights)}`,
					result: gbTxt(p.perGpu.weights)
				},
				totalStep(p.perGpu, 'W + A + O')
			]
		},
		{
			title: 'Roofline & control rate',
			intro: 'One cycle = vision encode + backbone prefill + a flow-matching action rollout.',
			steps: [
				{
					label: 'FLOPs per observation',
					formula: `f = \\frac{f_{\\text{bb}} + f_{\\text{vis}} + s_{\\text{flow}}\\, f_{\\text{exp}}}{t_p p_p}`,
					substituted: `\\approx ${sig(flops / 1e12)}\\text{ TFLOP}`,
					result: `${sig(flops / 1e12)} TFLOP (${vs.flowSteps} flow steps, chunk ${vs.chunkSize})`
				},
				{
					label: 'Seconds per control chunk',
					formula: `t_{\\text{ctl}} = \\max(t_{\\text{cmp}}, t_{\\text{mem}}) \\cdot p_p`,
					substituted: `= ${sig(v.secPerControl * 1000)}\\text{ ms}`,
					result: `${sig(v.secPerControl * 1000)} ms/chunk`,
					note: `compute alone ≈ ${sig(computeTime * 1000)} ms`
				},
				{
					label: 'Closed-loop control rate',
					formula: `\\text{Hz} = \\frac{\\text{chunkSize}}{t_{\\text{ctl}}}`,
					substituted: `\\frac{${vs.chunkSize}}{${sig(v.secPerControl)}} = ${sig(v.effectiveHz)}\\text{ Hz}`,
					result: `${sig(v.effectiveHz)} Hz per replica (${sig(v.actionsPerSec)} actions/s cluster)`,
					note: 'compute roofline only — real robot latency is control-loop/camera-IO bound (a floor)'
				}
			]
		}
	];
}

/** ASR (Whisper): one encoder pass over the audio window + autoregressive text decode. */
function asrChain(cfg: Config, m: ModelSpec, g: GpuSpec, fmt: WeightFormat): Section[] {
	const as = m.asr!;
	const p = computeProfile(cfg);
	const { tp, pp, dp, numGpus } = p.cluster;
	const layerShard = tp * pp;
	const batch = Math.max(1, cfg.batchSize);
	const peakFlops = computePeak(g, fmt.tier);
	const encFlops =
		(2 * as.encoderParams * as.audioTokens * batch +
			4 * (m.numLayers / 2) * as.audioTokens * as.audioTokens * m.hiddenSize * batch) /
		layerShard;
	const a = p.asr!;

	return [
		{
			title: 'Setup',
			steps: [
				...commonSetup(m, g, fmt, peakFlops, tp, pp, dp, numGpus),
				{
					label: 'Audio window',
					formula: `n_{\\text{aud}},\\;n_{\\text{txt}}`,
					substituted: `${as.audioTokens}\\text{ audio},\\;${as.avgTextTokens}\\text{ text}`,
					result: `${as.audioTokens} audio tokens (${as.audioWindowSec}s) → ${as.avgTextTokens} text tokens`
				}
			]
		},
		{
			title: 'Memory (per GPU)',
			steps: [
				{
					label: 'Weights (encoder + decoder)',
					formula: `W = \\frac{(P_{\\text{enc}} + P_{\\text{dec}})\\, b_w}{t_p p_p}`,
					substituted: `= ${gb(p.perGpu.weights)}`,
					result: gbTxt(p.perGpu.weights)
				},
				{
					label: 'KV cache (self + cross)',
					formula: `\\text{KV} = \\frac{n_{L}^{\\text{dec}} \\cdot 2 (n_{\\text{txt}}+n_{\\text{aud}})\\, n_h\\, d_h\\, b_{kv}}{t_p p_p} B`,
					substituted: `= ${gb(p.perGpu.kv)}`,
					result: gbTxt(p.perGpu.kv),
					note: 'decoder self-attention KV + cross-attention to the encoded audio'
				},
				totalStep(p.perGpu, 'W + \\text{KV} + A + O')
			]
		},
		{
			title: 'Two-phase timing',
			intro: 'Compute-bound encoder pass, then a memory-bound autoregressive decode.',
			steps: [
				{
					label: 'Encoder pass (compute-bound)',
					formula: `t_{\\text{enc}} = \\frac{2 P_{\\text{enc}} n_{\\text{aud}} + 4 \\frac{n_L}{2} n_{\\text{aud}}^2 h}{\\pi_{\\text{flops}}\\text{MFU}\\,\\phi}`,
					substituted: `f_{\\text{enc}} \\approx ${sig(encFlops / 1e12)}\\text{ TFLOP} \\Rightarrow ${sig(a.encoderMs)}\\text{ ms}`,
					result: `${sig(a.encoderMs)} ms (= TTFT)`
				},
				{
					label: 'Decoder (memory-bound, autoregressive)',
					formula: `t_{\\text{dec}} = n_{\\text{txt}} \\cdot \\frac{W_{\\text{dec}} + \\text{KV}/B}{\\pi_{\\text{bw}}\\text{MBU}}`,
					substituted: `${as.avgTextTokens}\\text{ tokens} \\Rightarrow ${sig(a.decoderMs)}\\text{ ms}`,
					result: `${sig(a.decoderMs)} ms`
				},
				{
					label: 'Per window / real-time factor',
					formula: `\\text{RTF} = \\frac{t_{\\text{audio}}}{t_{\\text{enc}} + t_{\\text{dec}}}`,
					substituted: `\\frac{${as.audioWindowSec}\\text{ s}}{${sig(a.secPerWindow)}\\text{ s}} = ${sig(a.rtf)}\\times`,
					result: `${sig(a.rtf)}× real-time per replica · ${sig(a.audioSecPerSec)} audio-s/s cluster`
				}
			]
		}
	];
}
