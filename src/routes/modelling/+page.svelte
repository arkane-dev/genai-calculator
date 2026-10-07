<script lang="ts">
	import ComputeDie from '$lib/components/ComputeDie.svelte';
	import BatchCurve, { type BatchPoint } from '$lib/components/BatchCurve.svelte';
	import Controls from '$lib/components/Controls.svelte';
	import CostPanel from '$lib/components/CostPanel.svelte';
	import EnergyPanel from '$lib/components/EnergyPanel.svelte';
	import LoadPanel from '$lib/components/LoadPanel.svelte';
	import HowTo from '$lib/components/HowTo.svelte';
	import InfoTip from '$lib/components/InfoTip.svelte';
	import MemoryStack from '$lib/components/MemoryStack.svelte';
	import Topology from '$lib/components/Topology.svelte';
	import Pipe from '$lib/components/Pipe.svelte';
	import StatCard from '$lib/components/StatCard.svelte';
	import { computeProfile, fmtBytes, fmtMs, fmtTps } from '$lib/profiler/calc';
	import {
		COLORS,
		FABRICS_BY_ID,
		GPUS_BY_ID,
		MODELS,
		MODELS_BY_ID,
		WEIGHT_FORMATS_BY_ID
	} from '$lib/profiler/data';
	import { configWarnings } from '$lib/profiler/sanity';
	import SanityNotes from '$lib/components/SanityNotes.svelte';
	import { parseModelUpload } from '$lib/profiler/upload';
	import type { Config, ModelSpec } from '$lib/profiler/types';
	import ShareButton from '$lib/components/ShareButton.svelte';
	import ExportButton from '$lib/components/ExportButton.svelte';
	import CompareTable, { type CompareRow } from '$lib/components/CompareTable.svelte';
	import type { SizingReport } from '$lib/report/markdown';
	import { applyShared, readSharedState } from '$lib/share/url';
	import { loadShared, saveShared, readPricing, type PricingState } from '$lib/state/shared.svelte';
	import { defaultPurchasing, estimateCost, PRICE_AS_OF } from '$lib/cost/pricing';
	import { DEFAULT_GRID, type EmissionsMethod } from '$lib/energy/estimate';
	import { base } from '$app/paths';
	import { onMount } from 'svelte';

	// shared region: one selector (in EnergyPanel) drives both energy CO2 and cost context
	let region = $state(DEFAULT_GRID);
	let emissionsMethod = $state<EmissionsMethod>('location');

	let config = $state<Config>({
		modelId: 'llama31-70b',
		gpuId: 'h100-sxm',
		numGpus: 8,
		gpusPerNode: 8,
		fabricId: 'mp-400',
		tp: 8,
		pp: 1,
		ppEnabled: false,
		ep: 1,
		epEnabled: false,
		batchSize: 32,
		inputTokens: 4096,
		outputTokens: 4096,
		weightFormatId: 'bf16',
		kvBits: 16,
		phase: 'decode',
		kvAllocation: 'paged',
		specDecode: false,
		draftAcceptRate: 0.7,
		specTokens: 4,
		cachedPrefixFrac: 0,
		steps: 50,
		resolution: 1024,
		guidance: true,
		frames: 1,
		imagesPerRequest: 0
	});

	let uploaded = $state<ModelSpec[]>([]);
	let uploadError = $state('');
	const allModels = $derived([...MODELS, ...uploaded]);

	// shared pricing controls (owned here, bound into <CostPanel>, synced across tabs)
	let pricing = $state<PricingState>({
		purchasing: 'on-demand',
		useEstimatedPricing: true,
		manualPerGpuHour: 0
	});

	// reopen a shared setup from the URL (?c=...)
	let syncLoaded = $state(false);
	onMount(() => {
		applyShared(config, loadShared());
		applyShared(config, readSharedState(location.search));
		readPricing(loadShared(), pricing);
		readPricing(readSharedState(location.search), pricing);
		syncLoaded = true;
	});
	$effect(() => {
		if (syncLoaded) saveShared({ ...(config as unknown as Record<string, unknown>), ...pricing });
	});

	async function onModelFile(e: Event) {
		const file = (e.target as HTMLInputElement).files?.[0];
		if (!file) return;
		try {
			const spec = parseModelUpload(JSON.parse(await file.text()));
			MODELS_BY_ID.set(spec.id, spec); // register so computeProfile can find it
			uploaded = [...uploaded.filter((m) => m.id !== spec.id), spec];
			config.modelId = spec.id;
			uploadError = '';
		} catch (err) {
			uploadError = err instanceof Error ? err.message : String(err);
		}
	}

	const bottleneckColor = {
		memory: COLORS.memPipe,
		compute: COLORS.tensor,
		network: COLORS.network
	};
	const fmtSeq = (n: number) =>
		n >= 1048576
			? `${(n / 1048576).toFixed(n % 1048576 ? 1 : 0)}M`
			: n >= 1024
				? `${(n / 1024).toFixed(0)}k`
				: String(n);

	const model = $derived(MODELS_BY_ID.get(config.modelId)!);
	const gpu = $derived(GPUS_BY_ID.get(config.gpuId)!);
	const sanity = $derived(
		configWarnings({
			tp: config.tp,
			gpusPerNode: config.gpusPerNode,
			fabricLabel: FABRICS_BY_ID.get(config.fabricId)?.label ?? config.fabricId,
			gpu,
			formatTier: WEIGHT_FORMATS_BY_ID.get(config.weightFormatId)?.tier ?? 'fp16'
		})
	);
	const isMoe = $derived(!!model.moe);
	const isVideoDiff = $derived(model.kind === 'diffusion' && (model.diffusion?.frames ?? 1) > 1);

	// cost per unit of work: tokens for LLMs, images/clips for the generative kinds
	const costMetric = $derived.by(() => {
		if (p.diffusion)
			return {
				kind: 'rate' as const,
				perSec: p.diffusion.imagesPerSec,
				unit: (isVideoDiff ? 'clips' : 'images') as 'clips' | 'images'
			};
		if (p.jepa)
			return { kind: 'rate' as const, perSec: p.jepa.clipsPerSec, unit: 'clips' as const };
		if (p.encoder)
			return { kind: 'rate' as const, perSec: p.encoder.docsPerSec, unit: 'docs' as const };
		if (p.vla) return undefined; // VLA best expressed per-hour
		if (p.asr)
			return {
				kind: 'rate' as const,
				perSec: p.asr.audioSecPerSec / 3600,
				unit: 'audio-hours' as const
			};
		return { kind: 'rate' as const, perSec: p.throughputTps, unit: 'tokens' as const };
	});
	const p = $derived(computeProfile(config));

	// Vector/CUDA cores carry norms, softmax and elementwise work — a blend of
	// bandwidth and compute pressure, always below the tensor-core load.
	const cudaFrac = $derived(Math.min(1, p.memBwFrac * 0.35 + p.computeFrac * 0.25));

	// Map the GPU's die blocks to what actually drives each during decode.
	//  - RT cores exist on Ada parts but LLM inference never touches them (0%).
	//  - mem  (memory controllers + L2) tracks the achieved HBM bandwidth.
	//  - link (on-die NVLink / PCIe I/O) tracks the interconnect fill.
	//  - sched (scheduler / uncore) is genuinely structural, no estimate.
	const dieUtil: Record<string, number | undefined> = $derived({
		tensor: p.computeFrac,
		cuda: cudaFrac,
		rt: 0,
		hbm: p.memBwFrac,
		l2: p.l2Frac,
		link: p.nvlinkFrac,
		sched: undefined
	});
	// which die block is the limiter — marked so you can see what you're bound by
	// (informational; a fitting config being memory- or compute-bound is normal, not a fault)
	const bottleneckKind = $derived(
		p.bottleneck === 'memory'
			? 'hbm'
			: p.bottleneck === 'compute'
				? 'tensor'
				: p.bottleneck === 'network'
					? 'link'
					: ''
	);
	const dieUnits = $derived(
		gpu.die.map((u) => ({
			label: u.label,
			areaFrac: u.areaFrac,
			color: COLORS[u.kind],
			isStatic: u.kind === 'sched',
			bottleneck: (u.kind === bottleneckKind ? 'info' : undefined) as 'info' | undefined,
			util: dieUtil[u.kind]
		}))
	);

	// Batch sweep: how throughput, per-user speed and $/token trade off as batch grows.
	// Only meaningful for token-serving (transformer) models; the generative kinds have
	// their own latency story on the Workload tab.
	const isTokenModel = $derived((model.kind ?? 'transformer') === 'transformer');
	const SWEEP_BATCHES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256];
	const batchClusterPerHour = $derived(
		estimateCost(config.gpuId, config.numGpus, defaultPurchasing(config.gpuId)).clusterPerHour
	);
	const batchPoints = $derived.by<BatchPoint[]>(() => {
		if (!isTokenModel) return [];
		const pts: BatchPoint[] = [];
		for (const b of SWEEP_BATCHES) {
			const pb = computeProfile({ ...config, batchSize: b });
			if (!pb.perGpu.fits || b > pb.maxConcurrentSeqs) break;
			const per1M =
				batchClusterPerHour != null && pb.throughputTps > 0
					? (batchClusterPerHour / 3600 / pb.throughputTps) * 1e6
					: null;
			pts.push({
				batch: b,
				throughputTps: pb.throughputTps,
				perUserTps: pb.perUserTps,
				ttftMs: pb.ttftMs,
				per1M
			});
		}
		return pts;
	});
	const fmtUsdSmall = (n: number) =>
		n >= 1 ? `$${n.toFixed(2)}` : n >= 0.01 ? `$${n.toFixed(3)}` : `$${n.toFixed(5)}`;
	const fmtUsd = (n: number) =>
		n >= 100 ? `$${n.toLocaleString('en-US', { maximumFractionDigits: 0 })}` : `$${n.toFixed(2)}`;

	// --- scenario A/B compare: pin a config, compare the live one against it ---
	let pinned = $state<Config | null>(null);
	const per1MOf = (cph: number | null, tps: number) =>
		cph != null && tps > 0 ? (cph / 3600 / tps) * 1e6 : null;
	const compareRows = $derived.by<CompareRow[]>(() => {
		if (!pinned) return [];
		const pP = computeProfile(pinned);
		const pGpu = GPUS_BY_ID.get(pinned.gpuId)!;
		const pModel = MODELS_BY_ID.get(pinned.modelId);
		const aCph = estimateCost(
			pinned.gpuId,
			pinned.numGpus,
			defaultPurchasing(pinned.gpuId)
		).clusterPerHour;
		const bCph = batchClusterPerHour;
		const aPer1M = per1MOf(aCph, pP.throughputTps);
		const bPer1M = per1MOf(bCph, p.throughputTps);
		return [
			{ label: 'Model', a: pModel?.name ?? pinned.modelId, b: model.name },
			{ label: 'GPUs', a: `${pinned.numGpus}× ${pGpu.name}`, b: `${config.numGpus}× ${gpu.name}` },
			{
				label: 'Parallelism',
				a: `TP${pinned.tp}·PP${pinned.pp}${pinned.ep > 1 ? `·EP${pinned.ep}` : ''}`,
				b: `TP${config.tp}·PP${config.pp}${config.ep > 1 ? `·EP${config.ep}` : ''}`
			},
			{ label: 'Batch', a: String(pinned.batchSize), b: String(config.batchSize) },
			{ label: 'Fits / GPU', a: pP.perGpu.fits ? 'Yes' : 'No', b: p.perGpu.fits ? 'Yes' : 'No' },
			{
				label: 'Aggregate throughput',
				a: `${fmtTps(pP.throughputTps)} tok/s`,
				b: `${fmtTps(p.throughputTps)} tok/s`,
				aNum: pP.throughputTps,
				bNum: p.throughputTps,
				better: 'higher'
			},
			{
				label: 'Per-user speed',
				a: `${pP.perUserTps.toFixed(1)} tok/s`,
				b: `${p.perUserTps.toFixed(1)} tok/s`,
				aNum: pP.perUserTps,
				bNum: p.perUserTps,
				better: 'higher'
			},
			{
				label: 'TTFT',
				a: fmtMs(pP.ttftMs),
				b: fmtMs(p.ttftMs),
				aNum: pP.ttftMs,
				bNum: p.ttftMs,
				better: 'lower'
			},
			{
				label: 'Memory / GPU',
				a: fmtBytes(pP.perGpu.used),
				b: fmtBytes(p.perGpu.used),
				aNum: pP.perGpu.used,
				bNum: p.perGpu.used,
				better: 'lower'
			},
			...(aCph != null && bCph != null
				? ([
						{
							label: 'Cluster / hour',
							a: fmtUsd(aCph),
							b: fmtUsd(bCph),
							aNum: aCph,
							bNum: bCph,
							better: 'lower'
						}
					] as CompareRow[])
				: []),
			...(aPer1M != null && bPer1M != null
				? ([
						{
							label: 'Cost / 1M tokens',
							a: fmtUsdSmall(aPer1M),
							b: fmtUsdSmall(bPer1M),
							aNum: aPer1M,
							bNum: bPer1M,
							better: 'lower'
						}
					] as CompareRow[])
				: [])
		];
	});
	const pinnedLabel = $derived(
		pinned
			? `${MODELS_BY_ID.get(pinned.modelId)?.name ?? pinned.modelId} · ${pinned.numGpus}× ${GPUS_BY_ID.get(pinned.gpuId)?.name}`
			: ''
	);

	// assemble a downloadable Markdown summary of the current scenario
	function buildReport(): SizingReport {
		const wf = WEIGHT_FORMATS_BY_ID.get(config.weightFormatId)?.label ?? config.weightFormatId;
		const cph = estimateCost(
			config.gpuId,
			config.numGpus,
			defaultPurchasing(config.gpuId)
		).clusterPerHour;
		const sections: SizingReport['sections'] = [
			{
				heading: 'Configuration',
				rows: [
					['Model', `${model.name} (${(model.params / 1e9).toFixed(1)}B params)`],
					['GPU', `${config.numGpus}× ${gpu.name}`],
					[
						'Parallelism',
						`TP${config.tp} · PP${config.pp}${config.ep > 1 ? ` · EP${config.ep}` : ''}`
					],
					['Batch size', String(config.batchSize)],
					['Weight format', wf],
					['KV precision', config.kvBits === 8 ? 'FP8' : 'FP16'],
					['Tokens', `${config.inputTokens} in / ${config.outputTokens} out`],
					['Phase', config.phase]
				]
			},
			{
				heading: 'Results (first-order roofline)',
				rows: [
					['Fits per GPU', p.perGpu.fits ? 'Yes' : 'No — does not fit'],
					['Bottleneck', `${p.bottleneck}-bound`],
					['Aggregate throughput', `${fmtTps(p.throughputTps)} tok/s`],
					['Per-user throughput', `${p.perUserTps.toFixed(1)} tok/s`],
					['TTFT', fmtMs(p.ttftMs)],
					['Memory', `${fmtBytes(p.perGpu.used)} / ${fmtBytes(p.perGpu.capacity)} per GPU`]
				]
			}
		];
		if (cph != null) {
			sections.push({
				heading: 'Cost (market estimate)',
				rows: [
					['Cluster / hour', fmtUsd(cph)],
					['Cluster / day', fmtUsd(cph * 24)],
					['Cluster / month', fmtUsd(cph * 730)],
					...(p.throughputTps > 0
						? ([['Cost / 1M output tokens', fmtUsdSmall((cph / 3600 / p.throughputTps) * 1e6)]] as [
								string,
								string
							][])
						: [])
				],
				note: `GPU cost: median on-demand market rate (${PRICE_AS_OF}).`
			});
		}
		return {
			title: `GPU sizing — ${model.name} on ${gpu.name}`,
			subtitle: `${config.numGpus}× ${gpu.name} · TP${config.tp}·PP${config.pp}${config.ep > 1 ? `·EP${config.ep}` : ''} · batch ${config.batchSize}`,
			generatedAt: new Date().toISOString().slice(0, 10),
			sections,
			disclaimer:
				'First-order roofline estimates, not a benchmark. GPU prices are a point-in-time market median. Get a real quote before you decide.'
		};
	}
</script>

<svelte:head><title>GenAI Calculator — Modelling</title></svelte:head>

<div>
	<div class="page">
		<header class="page-head">
			<div>
				<h1 class="page-title">Modelling</h1>
				<p class="lede">
					Transformer inference sizing. Will it fit, and how fast will it decode. First-order
					roofline estimates.
				</p>
			</div>
			<div class="row">
				<button
					type="button"
					onclick={() => (pinned = { ...config })}
					class="tool-btn"
					title="Pin this config as scenario A, then change controls to compare B against it"
				>
					{pinned ? 'Re-pin' : 'Pin to compare'}
				</button>
				<ExportButton report={buildReport} filename="gpu-sizing-modelling" />
				<ShareButton payload={config} />
			</div>
		</header>

		<HowTo>
			<p>
				This tab answers two questions about running an AI model on GPUs: <strong
					>will it fit?</strong
				>
				and <strong>how fast will it be?</strong> You choose a model and some hardware, and the app works
				out the answer instantly. It's a smart estimate, not a real benchmark, so treat the numbers as
				a close guide.
			</p>

			<p class="sub-title">Try it in three steps</p>
			<ol class="bullets numbered">
				<li>
					On the left, pick a <strong>model</strong> (the AI) and a <strong>GPU</strong> (the chip
					that runs it). Every control has an
					<span class="i-badge" aria-hidden="true">i</span> button that explains it.
				</li>
				<li>
					Set how many GPUs you have and how you split the model across them, then how much work you
					send it (batch size and how long the questions and answers are).
				</li>
				<li>
					Watch the pictures and numbers on the right change as you drag the sliders. Nothing is
					saved or sent anywhere — experiment freely.
				</li>
			</ol>

			<p class="sub-title">What the pictures mean</p>
			<ul class="bullets">
				<li>
					<strong>Memory box</strong> — everything that must fit in the GPU's memory: the model's weights,
					the KV cache (its memory of the conversation), and working space. If the box overflows, it won't
					fit.
				</li>
				<li>
					<strong>Pipe</strong> — how hard the GPU is reading its memory. This is usually the speed limit
					when writing an answer.
				</li>
				<li>
					<strong>Chip (die)</strong> — how busy the GPU's calculators are. This is the speed limit when
					reading a long question.
				</li>
				<li>
					<strong>Interconnect</strong> — the cables and network the GPUs use to talk to each other when
					a model is shared across many of them.
				</li>
			</ul>

			<p class="space-top">
				A model always has one thing slowing it down most — the <strong>bottleneck</strong>. It's
				either waiting on memory, waiting on math, or waiting to talk to other GPUs. The tool tells
				you which, so you know what to fix.
			</p>

			<p class="sub-title">Supported model types</p>
			<ul class="bullets">
				<li>
					<strong>Language models</strong> — the familiar chatbots and text models (Llama, Qwen, DeepSeek,
					GLM, Nemotron, and more). Dense and mixture-of-experts (MoE), including latent-attention (MLA)
					and hybrid Mamba designs. They read a question and write an answer one token at a time.
				</li>
				<li>
					<strong>Diffusion models</strong> — image generators (Stable Diffusion XL, FLUX.1, SD 3.5, PixArt-Σ,
					DiT-XL) and video generators (CogVideoX). They start from noise and clean it up over many steps.
					Sized by images (or clips) per second and time per image, not tokens. Video adds a frames control.
				</li>
				<li>
					<strong>Autoregressive image (VAR)</strong> — image generators built like a language model,
					predicting an image coarse-to-fine one scale at a time (VAR-d30). Modelled as a transformer:
					set Output tokens to the image's token count (about 680 for 256²).
				</li>
				<li>
					<strong>Vision-language (VLM)</strong> — LLMs that also read images (Qwen2.5-VL, Pixtral, LLaVA-OneVision).
					Each image adds a few hundred to a few thousand visual patch tokens to the prompt. Sized like
					an LLM; images inflate prefill and KV memory.
				</li>
				<li>
					<strong>Vision-language-action (VLA)</strong> — robotics policies (pi-zero, SmolVLA). A VLM
					backbone with a flow-matching action expert that produces a chunk of continuous actions per
					observation. Sized by robots driven at the control frequency (typically 50 Hz).
				</li>
				<li>
					<strong>Speech (ASR)</strong> — Whisper family. Encoder-decoder: 30-second audio window through
					the encoder, autoregressive text decode with cross-attention. Reports real-time factor (RTF),
					audio-sec/sec, and encoder-vs-decoder time. Sized by concurrent real-time streams and minimum
					RTF.
				</li>
				<li>
					<strong>Embeddings &amp; rerankers</strong> — text encoders that turn a document into a vector
					(BGE-M3, Jina, E5-Mistral) or score a query-doc pair (BGE reranker). A single bidirectional
					forward pass, no generation. Sized by docs per second and time per batch. Encoders love batching.
				</li>
				<li>
					<strong>World models (JEPA)</strong> — self-supervised video models (V-JEPA and V-JEPA 2).
					They watch a clip and turn it into an embedding in a single pass, no words in or out.
					Sized by clips per second. Because a clip is thousands of space-time patches, attention
					cost grows with the square of the patch count. The <strong>-AC</strong> variant is an action-conditioned
					world model: it also rolls a predictor forward over a planning horizon.
				</li>
			</ul>
			<p class="hint space-top-s">
				Pick the type from the grouped model menu on the left. The controls and result cards change
				to match the type you chose.
			</p>

			<p class="sub-title">Word list</p>
			<ul class="bullets">
				<li>
					<strong>Token</strong> — a chunk of text, roughly a short word or word-piece. Models read and
					write in tokens.
				</li>
				<li>
					<strong>Weights</strong> — the model's learned knowledge, stored as billions of numbers. Bigger
					models have more and need more memory.
				</li>
				<li>
					<strong>GPU</strong> — the specialized chip that does the heavy math. <strong>HBM</strong> is
					its fast on-chip memory.
				</li>
				<li>
					<strong>KV cache</strong> — short notes the model keeps about the conversation so it doesn't
					re-read everything for each new word. It grows with the length of the chat.
				</li>
				<li><strong>Batch</strong> — how many requests are handled together at once.</li>
				<li>
					<strong>Prefill vs decode</strong> — prefill is reading the whole question (math-heavy); decode
					is writing the answer one word at a time (memory-heavy).
				</li>
				<li>
					<strong>Parallelism (TP / PP / EP)</strong> — different ways to split one model across several
					GPUs when it's too big or too slow for one.
				</li>
				<li>
					<strong>Weight format / quantization</strong> — how precisely each number is stored. Fewer bits
					saves memory and can run faster, with a small accuracy cost.
				</li>
			</ul>

			<p class="space-top">
				Want the opposite — you know your users and speed goals and want to know how many GPUs to
				buy? Use the <a href="{base}/">Workload</a> tab.
			</p>
		</HowTo>

		<div class="layout">
			<!-- controls -->
			<aside
				class="sidebar"
			>
				<div class="panel">
					<Controls bind:config {isMoe} models={allModels} />
					<div class="group upload">
						<label class="upload-btn">
							+ Add model from JSON
							<input
								type="file"
								accept=".json,application/json,text/plain"
								class="visually-hidden"
								onchange={onModelFile}
							/>
						</label>
						<p class="hint centered">
							a HuggingFace config.json or a ModelSpec
						</p>
						{#if uploadError}
							<p class="hint bad">{uploadError}</p>
						{/if}
					</div>
				</div>
			</aside>

			<!-- results -->
			<main class="results">
				<SanityNotes notes={sanity} />
				<!-- headline numbers -->
				{#if p.diffusion}
					<div class="tiles cols-6">
						<StatCard
							label="Fits on device"
							value={p.perGpu.fits ? 'Yes' : 'No'}
							accent={p.perGpu.fits ? '#3ff0b8' : '#ff3b52'}
							sub={p.perGpu.fits
								? `${fmtBytes(p.perGpu.headroom)} free / GPU`
								: `${fmtBytes(-p.perGpu.headroom)} over / GPU`}
							info="Does the denoiser plus its text encoders and working space fit in each GPU's memory? If No, use more GPUs, tensor-parallel, or a smaller weight format."
						/>
						<StatCard
							label={isVideoDiff ? 'Clips / sec' : 'Images / sec'}
							value={p.diffusion.imagesPerSec.toFixed(p.diffusion.imagesPerSec < 1 ? 2 : 1)}
							unit={isVideoDiff ? 'clip/s' : 'img/s'}
							accent={COLORS.memPipe}
							sub="aggregate, {config.numGpus} GPU"
							info="How many {isVideoDiff
								? 'clips'
								: 'images'} the whole cluster finishes per second. Higher serves more users. Bigger batches and more GPUs raise this."
						/>
						<StatCard
							label={config.batchSize > 1
								? 'Batch latency'
								: isVideoDiff
									? 'Sec / clip'
									: 'Sec / image'}
							value={p.diffusion.secPerImage.toFixed(2)}
							unit="s"
							accent={bottleneckColor[p.bottleneck]}
							sub="{config.batchSize > 1
								? `${config.batchSize} ${isVideoDiff ? 'clip' : 'img'}/batch`
								: isVideoDiff
									? 'one clip'
									: 'one image'} · {p.diffusion.steps} steps"
							info="Wall-clock time to finish this run: denoising steps × time per step. A bigger batch takes longer overall (it's the whole batch's time), but produces more {isVideoDiff
								? 'clips'
								: 'images'} — see the rate. Fewer steps or tensor-parallel across GPUs lowers it."
						/>
						<StatCard
							label="Step time"
							value={fmtMs(p.diffusion.stepTimeMs)}
							sub="one denoising pass"
							info="Time for a single denoising step — one full pass of the model over the image. The total is this × the number of steps."
						/>
						<StatCard
							label="Latent tokens"
							value={fmtTps(p.diffusion.latentTokens)}
							sub="{config.resolution}²{isVideoDiff ? ` · ${config.frames}f` : ''} · {p.diffusion
								.cfg
								? 'CFG on'
								: 'no CFG'}"
							info="How many latent patches the model denoises per {isVideoDiff
								? 'clip (area × frames)'
								: 'image'} (area ÷ VAE downsample ÷ patch). More tokens = more compute. Guidance (CFG) doubles the work per step."
						/>
						<StatCard
							label={isVideoDiff ? 'Max clips / GPU' : 'Max images / GPU'}
							value={fmtTps(p.maxConcurrentSeqs)}
							accent={config.batchSize <= p.maxConcurrentSeqs ? '#3ff0b8' : '#ff3b52'}
							sub="in leftover memory"
							info="How many {isVideoDiff
								? 'clips'
								: 'images'} one GPU can generate at once with the memory left after weights. If your batch is larger than this, they won't all fit."
						/>
					</div>
				{:else if p.asr}
					<div class="tiles cols-6">
						<StatCard
							label="Fits on device"
							value={p.perGpu.fits ? 'Yes' : 'No'}
							accent={p.perGpu.fits ? '#3ff0b8' : '#ff3b52'}
							sub={p.perGpu.fits
								? `${fmtBytes(p.perGpu.headroom)} free / GPU`
								: `${fmtBytes(-p.perGpu.headroom)} over / GPU`}
							info="Does the encoder + decoder + KV cache fit in each GPU's memory?"
						/>
						<StatCard
							label="Real-time factor"
							value={p.asr.rtf.toFixed(p.asr.rtf < 10 ? 1 : 0)}
							unit="×"
							accent={p.asr.rtf >= 1 ? '#3ff0b8' : '#ff3b52'}
							sub={p.asr.rtf >= 1 ? 'faster than real-time' : 'slower than real-time'}
							info="Audio duration ÷ processing time. RTF ≥ 1 means the model transcribes faster than the audio plays — you can serve live speech. RTF 20× on a 30s clip means the model finishes it in 1.5s."
						/>
						<StatCard
							label="Audio-sec / sec"
							value={fmtTps(p.asr.audioSecPerSec)}
							sub="{config.numGpus} GPU · batch {config.batchSize}"
							info="Seconds of audio the whole cluster transcribes per real second. A batch of 16 streams on one GPU with RTF 10× = 160 audio-sec/sec throughput."
						/>
						<StatCard
							label="Sec / window"
							value={fmtMs(p.asr.secPerWindow * 1000)}
							accent={bottleneckColor[p.bottleneck]}
							sub="one {model.asr?.audioWindowSec ?? 30}s window · batch {config.batchSize}"
							info="Total wall-clock to transcribe one audio window: encoder pass + autoregressive text decode."
						/>
						<StatCard
							label="Encoder"
							value={fmtMs(p.asr.encoderMs)}
							sub="fixed audio pass"
							info="Time to process the mel spectrogram through the encoder. Independent of what's said — pure compute over a fixed number of audio tokens."
						/>
						<StatCard
							label="Decoder"
							value={fmtMs(p.asr.decoderMs)}
							sub="~{model.asr?.avgTextTokens ?? 128} tokens"
							info="Autoregressive text generation over the encoder's output. Memory-bound like an LLM decode; scales with output length."
						/>
					</div>
				{:else if p.vla}
					<div class="tiles cols-6">
						<StatCard
							label="Fits on device"
							value={p.perGpu.fits ? 'Yes' : 'No'}
							accent={p.perGpu.fits ? '#3ff0b8' : '#ff3b52'}
							sub={p.perGpu.fits
								? `${fmtBytes(p.perGpu.headroom)} free / GPU`
								: `${fmtBytes(-p.perGpu.headroom)} over / GPU`}
							info="Does the backbone + vision encoder + action expert fit in each GPU's memory?"
						/>
						<StatCard
							label="Sustained control"
							value={p.vla.effectiveHz.toFixed(0)}
							unit="Hz"
							accent={p.vla.effectiveHz >= (model.vla?.controlHz ?? 50) ? '#3ff0b8' : '#ff3b52'}
							sub="target {model.vla?.controlHz ?? 50} Hz · open-loop chunks"
							info="Closed-loop control frequency this GPU can sustain per replica: chunk size ÷ chunk latency. Green if it meets the deployed control rate."
						/>
						<StatCard
							label="Sec / chunk"
							value={fmtMs(p.vla.secPerControl * 1000)}
							accent={bottleneckColor[p.bottleneck]}
							sub="{model.vla?.chunkSize ?? 50}-action chunk · {model.vla?.flowSteps ??
								10} flow steps"
							info="Wall-clock time to produce one chunk of actions from one observation: vision encode + backbone prefill + flow-matching action rollout."
						/>
						<StatCard
							label="Actions / sec"
							value={fmtTps(p.vla.actionsPerSec)}
							sub="{config.numGpus} GPU · batch {config.batchSize}"
							info="Total actions produced across the cluster. In practice each robot consumes ~controlHz actions/sec (50 Hz for pi-zero); this figure divided by that shows how many robots one cluster can drive."
						/>
						<StatCard
							label="Robots served"
							value={fmtTps(p.vla.actionsPerSec / (model.vla?.controlHz ?? 50))}
							sub="at {model.vla?.controlHz ?? 50} Hz control"
							info="How many robots this cluster can drive at the target control frequency. actions/sec ÷ controlHz."
						/>
						<StatCard
							label="Chunks / sec"
							value={p.vla.controlsPerSec.toFixed(p.vla.controlsPerSec < 10 ? 1 : 0)}
							sub="cluster-wide inference calls"
							info="Number of full inference cycles (one observation → chunkSize actions) the cluster completes per second."
						/>
					</div>
				{:else if p.encoder}
					<div class="tiles cols-6">
						<StatCard
							label="Fits on device"
							value={p.perGpu.fits ? 'Yes' : 'No'}
							accent={p.perGpu.fits ? '#3ff0b8' : '#ff3b52'}
							sub={p.perGpu.fits
								? `${fmtBytes(p.perGpu.headroom)} free / GPU`
								: `${fmtBytes(-p.perGpu.headroom)} over / GPU`}
							info="Does the encoder fit in each GPU's memory? Encoders are small; if No, raise TP or a smaller weight format."
						/>
						<StatCard
							label={model.encoder?.task === 'reranker' ? 'Scores / sec' : 'Docs / sec'}
							value={p.encoder.docsPerSec.toFixed(p.encoder.docsPerSec < 10 ? 1 : 0)}
							unit={model.encoder?.task === 'reranker' ? 'score/s' : 'doc/s'}
							accent={COLORS.memPipe}
							sub="aggregate, {config.numGpus} GPU"
							info="How many documents (or query-doc pairs for a reranker) the whole cluster encodes per second."
						/>
						<StatCard
							label={config.batchSize > 1 ? 'Batch latency' : 'Sec / doc'}
							value={fmtMs(p.encoder.secPerDoc * 1000)}
							accent={bottleneckColor[p.bottleneck]}
							sub={config.batchSize > 1 ? `${config.batchSize} in batch` : 'one document'}
							info="Wall-clock time for one forward pass. A bigger batch takes longer overall but produces more results, see docs/sec."
						/>
						<StatCard
							label="Tokens / sec"
							value={fmtTps(p.throughputTps)}
							unit="tok/s"
							sub="tokens processed"
							info="Total tokens the encoder ingests per second across all GPUs. Handy to compare an encoder against an LLM's prefill throughput."
						/>
						<StatCard
							label="Sequence length"
							value={fmtSeq(p.encoder.tokens)}
							sub={model.encoder?.embedDim
								? `${model.encoder.embedDim}-dim vector`
								: 'reranker score'}
							info="Tokens processed per document. Attention is quadratic in this: doubling the sequence quadruples attention cost."
						/>
						<StatCard
							label="Max batch / GPU"
							value={fmtTps(p.maxConcurrentSeqs)}
							accent={config.batchSize <= p.maxConcurrentSeqs ? '#3ff0b8' : '#ff3b52'}
							sub="in leftover memory"
							info="How many docs one GPU can process at once with the memory left after weights."
						/>
					</div>
				{:else if p.jepa}
					<div class="tiles cols-6">
						<StatCard
							label="Fits on device"
							value={p.perGpu.fits ? 'Yes' : 'No'}
							accent={p.perGpu.fits ? '#3ff0b8' : '#ff3b52'}
							sub={p.perGpu.fits
								? `${fmtBytes(p.perGpu.headroom)} free / GPU`
								: `${fmtBytes(-p.perGpu.headroom)} over / GPU`}
							info="Does the encoder plus its predictor and working space fit in each GPU's memory? If No, use more GPUs, tensor-parallel, or a smaller weight format."
						/>
						<StatCard
							label="Clips / sec"
							value={p.jepa.clipsPerSec.toFixed(p.jepa.clipsPerSec < 1 ? 2 : 1)}
							unit="clip/s"
							accent={COLORS.memPipe}
							sub="aggregate, {config.numGpus} GPU"
							info="How many video clips the whole cluster encodes into embeddings per second. Higher serves more downstream work. Bigger batches and more GPUs raise this."
						/>
						<StatCard
							label={config.batchSize > 1 ? 'Batch latency' : 'Sec / clip'}
							value={p.jepa.secPerClip.toFixed(3)}
							unit="s"
							accent={bottleneckColor[p.bottleneck]}
							sub="{config.batchSize > 1 ? `${config.batchSize} clip/batch` : 'one clip'} · {p.jepa
								.frames}f"
							info="Wall-clock time to encode this run — one forward pass over the video patches. A bigger batch takes longer overall but produces more clips, see clips/sec."
						/>
						<StatCard
							label="Forward pass"
							value={fmtMs(p.jepa.forwardTimeMs)}
							sub="single encode"
							info="Time for one encoder forward over the clip's patches. There are no denoising steps or generated tokens — JEPA encodes in a single pass."
						/>
						<StatCard
							label="Patch tokens"
							value={fmtTps(p.jepa.tokens)}
							sub="{config.resolution}² · {p.jepa.frames}f"
							info="How many space-time patches the encoder processes per clip: (resolution ÷ patch)² spatial × (frames ÷ tubelet) temporal. Attention cost grows with the square of this."
						/>
						<StatCard
							label="Max clips / GPU"
							value={fmtTps(p.maxConcurrentSeqs)}
							accent={config.batchSize <= p.maxConcurrentSeqs ? '#3ff0b8' : '#ff3b52'}
							sub="in leftover memory"
							info="How many clips one GPU can encode at once with the memory left after weights. If your batch is larger than this, they won't all fit."
						/>
					</div>
				{:else}
					<div class="tiles cols-6">
						<StatCard
							label="Fits on device"
							value={p.perGpu.fits ? 'Yes' : 'No'}
							accent={p.perGpu.fits ? '#3ff0b8' : '#ff3b52'}
							sub={p.perGpu.fits
								? `${fmtBytes(p.perGpu.headroom)} free / GPU`
								: `${fmtBytes(-p.perGpu.headroom)} over / GPU`}
							info="Does everything the model needs fit in each GPU's memory? That's the model's weights, the KV cache (its running memory of the conversation), and some working space. If it says No, the model won't run as set up — use fewer/smaller pieces, more GPUs, or a smaller weight format."
						/>
						<StatCard
							label="Max concurrent seqs"
							value={fmtTps(p.maxConcurrentSeqs)}
							accent={config.batchSize <= p.maxConcurrentSeqs ? '#3ff0b8' : '#ff3b52'}
							sub="{fmtSeq(
								config.inputTokens + config.outputTokens
							)} ctx · {config.kvAllocation} KV"
							info="How many conversations one GPU can hold at the same time with the leftover memory. Each conversation needs its own KV cache, so longer chats mean fewer fit. If your batch size is bigger than this number, they won't all fit at once."
						/>
						<StatCard
							label="{p.phase === 'prefill' ? 'Prefill' : 'Decode'} throughput"
							value={fmtTps(p.throughputTps)}
							unit="tok/s"
							accent={COLORS.memPipe}
							sub="aggregate, {config.numGpus} GPU"
							info="Total words-per-second (in 'tokens') the whole cluster produces across all users at once. Higher means you serve more people. It's the headline speed number, found by dividing the work by the time each step takes."
						/>
						<StatCard
							label="Per sequence"
							value={fmtTps(p.perUserTps)}
							unit="tok/s"
							sub={p.specSpeedup && p.specSpeedup > 1
								? `single user · ${p.specSpeedup.toFixed(1)}× spec-decode`
								: 'single user decode'}
							accent={p.specSpeedup && p.specSpeedup > 1 ? '#3ff0b8' : undefined}
							info="How fast the answer appears for ONE user, in words per second. This is what a single person feels. Bigger batches raise total throughput but can lower this, because the GPU is sharing its time among more users. Speculative decoding raises it directly (decode only)."
						/>
						<StatCard
							label="Time to first token"
							value={fmtMs(p.ttftMs)}
							sub="prefill · {p.bottleneck}-bound"
							accent={bottleneckColor[p.bottleneck]}
							info="How long a user waits before the very first word appears. Lower feels snappier. It's set by how long the model takes to read the whole question (the 'prefill' step) before it can start answering."
						/>
						<StatCard
							label="Full request"
							value={fmtMs(p.ttftMs + (config.outputTokens / p.perUserTps) * 1000)}
							sub="TTFT + {config.outputTokens} tok decode"
							info="The total wait for a whole answer: the time to the first word, plus the time to write all the remaining words. This is the end-to-end experience for one user."
						/>
					</div>
				{/if}

				<!-- memory -> pipe -> compute -->
				<section class="panel">
					<div class="mem-split three">
						<div class="mem-chart">
							<MemoryStack
								segments={p.segments}
								capacity={p.perGpu.capacity}
								used={p.perGpu.used}
								fits={p.perGpu.fits}
							/>
						</div>

						<div class="pipe-col">
							<Pipe
								frac={p.memBwFrac}
								color={COLORS.memPipe}
								label="HBM bandwidth"
								detail="{p.bandwidthUsedGBs.toFixed(0)} / {p.bandwidthPeakGBs.toFixed(0)} GB/s"
							/>
							<div class="readout">
								{Math.round(p.memBwFrac * 100)}% of {gpu.memBandwidthTBs} TB/s
								<InfoTip
									text="How fast the GPU reads its memory, versus the fastest it possibly could. Writing each new word means re-reading the model's weights, so during decode this pipe is usually the real speed limit. When it's near 100%, the GPU is 'memory-bound' — waiting on memory, not on math."
								/>
							</div>
						</div>

						<div class="die-col">
							<ComputeDie units={dieUnits} />
							<div class="readout">
								{p.computeAchievedTflops.toFixed(0)} / {p.computePeakTflops.toFixed(0)} TFLOPS
								<InfoTip
									text="How much math the GPU is doing, versus the most it can (TFLOPS = trillions of math operations per second). Reading a long question ('prefill') is heavy on math and lights this up; writing words one at a time ('decode') uses little. When it's near 100%, the GPU is 'compute-bound' — limited by math speed."
								/>
							</div>
							<div class="hint centered">
								HBM I/O tracks memory bandwidth, L2 tracks the hotter of memory/compute (so it stays
								busy in prefill), link tracks interconnect; scheduler shown structural
							</div>
						</div>
					</div>
				</section>

				<!-- interconnect: nodes + fabric -->
				<section>
					<div class="field-label">
						Interconnect
						<InfoTip
							text="When a model is split across several GPUs, they must swap results after every step. This shows those links: fast NVLink cables between GPUs inside one server, and the slower network (InfiniBand or RDMA Ethernet) between servers. If a link fills up, the GPUs spend time waiting to talk instead of working — 'network-bound'."
						/>
					</div>
					<Topology {p} hasNvlink={gpu.hasNvlink} />
				</section>

				{#if pinned}
					<CompareTable
						labelA={pinnedLabel}
						labelB={`${model.name} · ${config.numGpus}× ${gpu.name}`}
						rows={compareRows}
						onclear={() => (pinned = null)}
					/>
				{/if}

				{#if isTokenModel && batchPoints.length >= 2}
					<BatchCurve points={batchPoints} currentBatch={config.batchSize} {fmtTps} {fmtUsdSmall} />
				{/if}

				<CostPanel
					gpuId={config.gpuId}
					numGpus={config.numGpus}
					metric={costMetric}
					bind:useEstimated={pricing.useEstimatedPricing}
					bind:manualPerGpuHour={pricing.manualPerGpuHour}
					bind:purchasing={pricing.purchasing}
				/>

				<EnergyPanel
					gpuId={config.gpuId}
					numGpus={config.numGpus}
					util={Math.max(p.memBwFrac, p.computeFrac)}
					metric={costMetric}
					bind:region
					bind:method={emissionsMethod}
				/>

				<LoadPanel
					gpuId={config.gpuId}
					numGpus={config.numGpus}
					weightBytes={(model.params *
						(WEIGHT_FORMATS_BY_ID.get(config.weightFormatId)?.bitsPerWeight ?? 16)) /
						8}
				/>

				<p class="small mute">
					First-order roofline (MBU 0.8, MFU 0.7) with 1.5 GB runtime overhead per GPU. MoE/EP
					sharding, activation memory and KV allocation waste are approximate; projected models are
					estimates from the current generation.
				</p>
			</main>
		</div>
	</div>
</div>

<style>
	.pipe-col { display: flex; flex-direction: column; justify-content: center; }
	.die-col { display: flex; flex-direction: column; }
	.readout {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.375rem;
		margin-top: var(--nd-space-2);
		color: var(--nd-text-mute);
		font-size: var(--nd-text-xs);
		text-align: center;
	}
	.upload { margin-top: var(--nd-space-4); }
</style>
