<script lang="ts">
	import ComputeDie from '$lib/components/ComputeDie.svelte';
	import CostPanel from '$lib/components/CostPanel.svelte';
	import EnergyPanel from '$lib/components/EnergyPanel.svelte';
	import LoadPanel from '$lib/components/LoadPanel.svelte';
	import HowTo from '$lib/components/HowTo.svelte';
	import MemoryStack from '$lib/components/MemoryStack.svelte';
	import Segmented from '$lib/components/Segmented.svelte';
	import Slider from '$lib/components/Slider.svelte';
	import Toggle from '$lib/components/Toggle.svelte';
	import StatCard from '$lib/components/StatCard.svelte';
	import Topology from '$lib/components/Topology.svelte';
	import { computeProfile, fmtMs, fmtTps } from '$lib/profiler/calc';
	import {
		COLORS,
		defaultFabricFor,
		FABRICS,
		GPUS,
		GPUS_BY_ID,
		gpuSupportsFormat,
		MODELS,
		MODELS_BY_ID,
		weightFormatOptions,
		WEIGHT_FORMATS_BY_ID
	} from '$lib/profiler/data';
	import { parseModelUpload } from '$lib/profiler/upload';
	import type { ModelSpec } from '$lib/profiler/types';
	import { sizeDisaggregated, solveWorkload } from '$lib/sizing/solve';
	import { base } from '$app/paths';
	import type { WorkloadSpec } from '$lib/sizing/types';
	import InfoTip from '$lib/components/InfoTip.svelte';
	import ShareButton from '$lib/components/ShareButton.svelte';
	import HandoffMenu from '$lib/components/HandoffMenu.svelte';
	import ExportButton from '$lib/components/ExportButton.svelte';
	import type { SizingReport } from '$lib/report/markdown';
	import { applyShared, readSharedState } from '$lib/share/url';
	import { loadShared, saveShared, readPricing, type PricingState } from '$lib/state/shared';
	import { defaultPurchasing, estimateCost, PRICE_AS_OF } from '$lib/cost/pricing';
	import { DEFAULT_GRID, type EmissionsMethod } from '$lib/energy/estimate';
	import { onMount } from 'svelte';

	// shared region: one selector (in EnergyPanel) drives both energy CO2 and cost context
	let region = $state(DEFAULT_GRID);
	let emissionsMethod = $state<EmissionsMethod>('location');

	let spec = $state<WorkloadSpec>({
		modelId: 'llama31-70b',
		gpuId: 'h100-sxm',
		gpusPerNode: 8,
		fabricId: 'mp-400',
		weightFormatId: 'bf16',
		kvBits: 16,
		kvAllocation: 'paged',
		concurrency: 256,
		basis: 'average',
		inputTokens: 2048,
		outputTokens: 512,
		ttftTargetMs: 1500,
		throughputTargetTps: 30,
		steps: 50,
		resolution: 1024,
		guidance: true,
		targetImagesPerSec: 5,
		maxSecPerImage: 5,
		frames: 64,
		targetClipsPerSec: 10,
		maxSecPerClip: 2,
		targetDocsPerSec: 500,
		maxSecPerDoc: 0.5,
		imagesPerRequest: 0,
		targetRobots: 5,
		maxSecPerControl: 1,
		targetStreams: 100,
		minRtf: 1,
		specDecode: false,
		draftAcceptRate: 0.7,
		specTokens: 4,
		cachedPrefixFrac: 0
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
	// Cross-tab sync: persisted store first, then the URL (?c=) overrides for share links.
	let syncLoaded = $state(false);
	onMount(() => {
		applyShared(spec, loadShared());
		applyShared(spec, readSharedState(location.search));
		readPricing(loadShared(), pricing);
		readPricing(readSharedState(location.search), pricing);
		syncLoaded = true;
	});
	// Sync the SOLVER-CHOSEN cluster (tp/pp/ep/numGpus/batch), not the bare spec, so
	// Modelling / Economics inherit the exact fitting layout and agree on "fits".
	$effect(() => {
		if (syncLoaded)
			saveShared({ ...(chosen.config as unknown as Record<string, unknown>), ...pricing });
	});
	async function onModelFile(e: Event) {
		const file = (e.target as HTMLInputElement).files?.[0];
		if (!file) return;
		try {
			const m = parseModelUpload(JSON.parse(await file.text()));
			MODELS_BY_ID.set(m.id, m);
			uploaded = [...uploaded.filter((x) => x.id !== m.id), m];
			spec.modelId = m.id;
			uploadError = '';
		} catch (err) {
			uploadError = err instanceof Error ? err.message : String(err);
		}
	}

	const cpu = [1, 2, 4, 8].map((n) => ({ value: n, label: String(n) }));
	const fmtSeq = (n: number) =>
		n >= 1024 ? `${(n / 1024).toFixed(n % 1024 ? 1 : 0)}k` : String(n);

	// disaggregated prefill/decode: only valid for LLM workloads
	let disaggEnabled = $state(false);
	const result = $derived(solveWorkload(spec));
	const chosen = $derived(result.chosen);
	const p = $derived(chosen.feasible ? computeProfile(chosen.config) : null);
	// Even when infeasible, profile the solver's best-attempt config so the die + memory
	// stay visible (with overflow / >100%), mirroring the Modelling tab.
	const viz = $derived(computeProfile(chosen.config));
	const gpu = $derived(GPUS_BY_ID.get(spec.gpuId)!);
	const wfOpts = $derived(weightFormatOptions(gpu));
	// Clamp to BF16 if the GPU can't run the selected format (e.g. FP8 on a T4).
	$effect(() => {
		const f = WEIGHT_FORMATS_BY_ID.get(spec.weightFormatId);
		if (gpu && f && !gpuSupportsFormat(gpu, f)) spec.weightFormatId = 'bf16';
	});

	const modelOf = $derived(allModels.find((mm) => mm.id === spec.modelId));
	const isDiffusion = $derived(modelOf?.kind === 'diffusion');
	const isVideoDiff = $derived(isDiffusion && (modelOf?.diffusion?.frames ?? 1) > 1);
	const isJepa = $derived(modelOf?.kind === 'jepa');
	const isEncoder = $derived(modelOf?.kind === 'encoder');
	const isVla = $derived(modelOf?.kind === 'vla');
	const isAsr = $derived(modelOf?.kind === 'asr');
	const isVlm = $derived(!isVla && !!modelOf?.vlm);
	const disagg = $derived(
		disaggEnabled && !isDiffusion && !isJepa && !isEncoder && !isVla && !isAsr
			? sizeDisaggregated(spec, spec.gpuId)
			: null
	);
	const VID_FRAMES = [5, 9, 13, 17, 21];
	const modelGroups = $derived(
		[
			{
				label: 'Language models',
				items: allModels.filter(
					(mm) =>
						(mm.kind ?? 'transformer') === 'transformer' &&
						!mm.visualAR &&
						!mm.vlm &&
						!mm.reasoning &&
						!mm.tts
				)
			},
			{ label: 'Reasoning models', items: allModels.filter((mm) => mm.reasoning) },
			{ label: 'Speech synthesis (TTS)', items: allModels.filter((mm) => mm.tts) },
			{
				label: 'Vision-language (VLM)',
				items: allModels.filter((mm) => (mm.kind ?? 'transformer') === 'transformer' && mm.vlm)
			},
			{ label: 'Vision-language-action (VLA)', items: allModels.filter((mm) => mm.kind === 'vla') },
			{ label: 'Autoregressive image', items: allModels.filter((mm) => mm.visualAR) },
			{ label: 'Diffusion models', items: allModels.filter((mm) => mm.kind === 'diffusion') },
			{ label: 'Embeddings & rerankers', items: allModels.filter((mm) => mm.kind === 'encoder') },
			{ label: 'Speech (ASR)', items: allModels.filter((mm) => mm.kind === 'asr') },
			{ label: 'World models (JEPA)', items: allModels.filter((mm) => mm.kind === 'jepa') }
		].filter((gp) => gp.items.length)
	);
	const RES = [256, 512, 768, 1024, 1536, 2048];
	const costMetric = $derived.by(() => {
		if (isDiffusion)
			return {
				kind: 'rate' as const,
				perSec: chosen.imagesPerSec ?? 0,
				unit: (isVideoDiff ? 'clips' : 'images') as 'clips' | 'images'
			};
		if (isJepa)
			return { kind: 'rate' as const, perSec: chosen.clipsPerSec ?? 0, unit: 'clips' as const };
		if (isEncoder)
			return { kind: 'rate' as const, perSec: chosen.docsPerSec ?? 0, unit: 'docs' as const };
		if (isVla) return undefined; // VLA cost is best expressed per-hour, not per-work-unit
		if (isAsr)
			return {
				kind: 'rate' as const,
				perSec: (chosen.streamsServed ?? 0) / 3600,
				unit: 'audio-hours' as const
			};
		return { kind: 'rate' as const, perSec: chosen.clusterTps, unit: 'tokens' as const };
	});
	const JEPA_RES = [224, 256, 384, 512];
	const FRAMES = [8, 16, 32, 64];

	// --- auto-recommend: rank the GPU comparison by an objective ---
	type Objective = 'gpus' | 'hourly' | 'unit';
	let objective = $state<Objective>('gpus');
	const workUnit = $derived(
		isDiffusion ? (isVideoDiff ? 'clip' : 'image') : isJepa ? 'clip' : 'tok'
	);
	const rowRate = (r: (typeof result.comparison)[number]) =>
		isDiffusion ? (r.imagesPerSec ?? 0) : isJepa ? (r.clipsPerSec ?? 0) : r.clusterTps;
	// cost basis: the on-demand market median for each GPU
	const rowCostHr = (r: (typeof result.comparison)[number]) =>
		r.feasible ? estimateCost(r.gpuId, r.numGpus, defaultPurchasing(r.gpuId)).clusterPerHour : null;
	const rowCostPerUnit = (r: (typeof result.comparison)[number]) => {
		const hr = rowCostHr(r);
		const rate = rowRate(r);
		if (hr == null || rate <= 0) return null;
		const perSec = hr / 3600;
		// tokens → per 1M; images/clips → per 1k
		return workUnit === 'tok' ? (perSec / rate) * 1e6 : (perSec / rate) * 1e3;
	};
	const ranked = $derived.by(() => {
		const withCost = result.comparison.map((r) => ({
			r,
			hr: rowCostHr(r),
			unit: rowCostPerUnit(r)
		}));
		const feas = withCost.filter((x) => x.r.feasible);
		const infeas = withCost.filter((x) => !x.r.feasible);
		const key =
			objective === 'gpus'
				? (x: (typeof feas)[number]) => x.r.numGpus
				: objective === 'hourly'
					? (x: (typeof feas)[number]) => x.hr ?? Infinity
					: (x: (typeof feas)[number]) => x.unit ?? Infinity;
		feas.sort((a, b) => key(a) - key(b));
		return { rows: [...feas, ...infeas], bestGpuId: feas[0]?.r.gpuId };
	});
	const fmtUsd = (n: number) => (n >= 100 ? `$${n.toFixed(0)}` : `$${n.toFixed(2)}`);
	const fmtUsdSmall = (n: number) =>
		n >= 1 ? `$${n.toFixed(2)}` : n >= 0.01 ? `$${n.toFixed(3)}` : `$${n.toFixed(5)}`;
	const unitCostLabel = $derived(workUnit === 'tok' ? '$/1M tok' : `$/1k ${workUnit}`);

	// downloadable Markdown summary of the sized workload
	function buildReport(): SizingReport {
		const name = modelOf?.name ?? spec.modelId;
		const demand: [string, string][] = [
			['Model', name],
			['Candidate GPU', gpu.name],
			['GPUs per node', String(spec.gpusPerNode)]
		];
		if (isDiffusion)
			demand.push(
				['Target', `${spec.targetImagesPerSec} ${isVideoDiff ? 'clip' : 'img'}/s`],
				['Max sec/' + (isVideoDiff ? 'clip' : 'image'), `${spec.maxSecPerImage} s`]
			);
		else if (isJepa)
			demand.push(
				['Target', `${spec.targetClipsPerSec} clip/s`],
				['Max sec/clip', `${spec.maxSecPerClip} s`]
			);
		else if (isEncoder)
			demand.push(
				['Target', `${spec.targetDocsPerSec} doc/s`],
				['Max sec/batch', `${spec.maxSecPerDoc} s`]
			);
		else if (isAsr)
			demand.push(['Streams', String(spec.targetStreams)], ['Min RTF', `${spec.minRtf}×`]);
		else if (isVla)
			demand.push(
				['Robots', String(spec.targetRobots)],
				['Max sec/chunk', `${spec.maxSecPerControl} s`]
			);
		else
			demand.push(
				[
					'Concurrency',
					`${spec.concurrency} (${spec.basis} → ${chosen.provisionedConcurrency} peak)`
				],
				['Tokens', `${spec.inputTokens} in / ${spec.outputTokens} out`],
				['Max TTFT', `${spec.ttftTargetMs} ms`],
				['Min throughput', `${spec.throughputTargetTps} tok/s/user`]
			);

		const sections: SizingReport['sections'] = [{ heading: 'Demand & SLOs', rows: demand }];

		if (!chosen.feasible) {
			sections.push({ heading: 'Result', rows: [['Feasible', 'No']], note: chosen.reason });
		} else {
			const res: [string, string][] = [
				[
					'Cluster',
					`${chosen.numGpus}× ${gpu.name} (${chosen.numNodes} node${chosen.numNodes > 1 ? 's' : ''})`
				],
				[
					'Parallelism',
					`TP${chosen.tp} · PP${chosen.pp}${chosen.ep > 1 ? ` · EP${chosen.ep}` : ''} · DP${chosen.dp}`
				],
				['Batch / replica', String(chosen.batchPerReplica)]
			];
			if (isDiffusion)
				res.push(
					[isVideoDiff ? 'Clips/sec' : 'Images/sec', (chosen.imagesPerSec ?? 0).toFixed(2)],
					['Sec/' + (isVideoDiff ? 'clip' : 'image'), (chosen.secPerImage ?? 0).toFixed(2)]
				);
			else if (isJepa)
				res.push(
					['Clips/sec', (chosen.clipsPerSec ?? 0).toFixed(2)],
					['Sec/clip', (chosen.secPerClip ?? 0).toFixed(3)]
				);
			else if (isEncoder) res.push(['Docs/sec', (chosen.docsPerSec ?? 0).toFixed(0)]);
			else if (isAsr)
				res.push(
					['Streams served', (chosen.streamsServed ?? 0).toFixed(0)],
					['RTF/replica', (chosen.rtf ?? 0).toFixed(1) + '×']
				);
			else if (isVla)
				res.push(
					['Robots driven', (chosen.robotsDriven ?? 0).toFixed(0)],
					['Sec/chunk', ((chosen.secPerControl ?? 0) * 1000).toFixed(0) + ' ms']
				);
			else
				res.push(
					['Cluster throughput', `${fmtTps(chosen.clusterTps)} tok/s`],
					['Per-user throughput', `${fmtTps(chosen.perUserTps)} tok/s`],
					['TTFT', fmtMs(chosen.ttftMs)]
				);
			sections.push({ heading: 'Sized cluster (first-order)', rows: res });

			const cph = estimateCost(
				spec.gpuId,
				chosen.numGpus,
				defaultPurchasing(spec.gpuId)
			).clusterPerHour;
			if (cph != null)
				sections.push({
					heading: 'Cost (market estimate)',
					rows: [
						['Cluster / hour', fmtUsd(cph)],
						['Cluster / day', fmtUsd(cph * 24)],
						['Cluster / month', fmtUsd(cph * 730)]
					],
					note: `GPU cost: median on-demand market rate (${PRICE_AS_OF}).`
				});
		}

		return {
			title: `Workload sizing — ${name} on ${gpu.name}`,
			subtitle: chosen.feasible ? `${chosen.numGpus}× ${gpu.name}` : 'infeasible on this GPU',
			generatedAt: new Date().toISOString().slice(0, 10),
			sections,
			disclaimer:
				'First-order sizing estimates, not a benchmark. GPU prices are a point-in-time market median. Get a real quote before you decide.'
		};
	}
	// seed diffusion / JEPA runtime knobs from the model's defaults on switch
	let seededFor = $state('');
	$effect(() => {
		const dsm = modelOf?.diffusion;
		const jm = modelOf?.jepa;
		if (dsm && spec.modelId !== seededFor) {
			spec.steps = dsm.defaultSteps;
			spec.resolution = dsm.defaultResolution;
			spec.guidance = dsm.cfg;
			spec.frames = dsm.frames ?? 1;
			seededFor = spec.modelId;
		} else if (jm && spec.modelId !== seededFor) {
			spec.resolution = jm.defaultResolution;
			spec.frames = jm.defaultFrames;
			seededFor = spec.modelId;
		} else if (modelOf?.encoder && spec.modelId !== seededFor) {
			spec.inputTokens = modelOf.encoder.defaultSeqLen;
			seededFor = spec.modelId;
		} else if (modelOf?.kind === 'vla' && spec.modelId !== seededFor) {
			seededFor = spec.modelId;
		} else if (modelOf?.kind === 'asr' && spec.modelId !== seededFor) {
			seededFor = spec.modelId;
		} else if (modelOf?.vlm && spec.modelId !== seededFor) {
			spec.imagesPerRequest = 1;
			seededFor = spec.modelId;
		} else if (!modelOf?.vlm && !modelOf?.vla && spec.modelId !== seededFor) {
			if (spec.imagesPerRequest > 0) spec.imagesPerRequest = 0;
			seededFor = spec.modelId;
		}
	});

	const cudaFrac = $derived(Math.min(1, viz.memBwFrac * 0.35 + viz.computeFrac * 0.25));
	// which die block is the binding resource — highlighted red when the workload is infeasible
	const bottleneckKind = $derived(
		viz.bottleneck === 'memory'
			? 'hbm'
			: viz.bottleneck === 'compute'
				? 'tensor'
				: viz.bottleneck === 'network'
					? 'link'
					: ''
	);
	const dieUnits = $derived(
		gpu.die.map((u) => ({
			label: u.label,
			areaFrac: u.areaFrac,
			color: COLORS[u.kind],
			isStatic: u.kind === 'sched',
			bottleneck:
				u.kind === bottleneckKind
					? ((chosen.feasible ? 'info' : 'alert') as 'info' | 'alert')
					: undefined,
			util:
				u.kind === 'tensor'
					? viz.computeFrac
					: u.kind === 'cuda'
						? cudaFrac
						: u.kind === 'rt'
							? 0
							: u.kind === 'hbm' || u.kind === 'l2'
								? viz.memBwFrac
								: u.kind === 'link'
									? viz.nvlinkFrac
									: undefined
		}))
	);
	// plain-language bottleneck label for the infeasible header badge
	const bottleneckLabel = $derived(
		viz.bottleneck === 'memory'
			? 'memory bandwidth (HBM)'
			: viz.bottleneck === 'compute'
				? 'compute (tensor cores)'
				: viz.bottleneck === 'network'
					? 'the interconnect'
					: viz.bottleneck
	);
</script>

<svelte:head><title>GenAI Calculator — Workload sizing</title></svelte:head>

<div class="min-h-screen bg-slate-950 text-slate-100">
	<div class="mx-auto max-w-7xl px-6 py-8">
		<header class="mb-6 flex items-start justify-between gap-4">
			<div>
				<h1 class="text-2xl font-semibold">Workload Sizing</h1>
				<p class="mt-1 text-sm text-slate-400">
					Describe the demand and per-user SLOs. The smallest cluster that meets them is sized live.
				</p>
			</div>
			<div class="flex items-center gap-2">
				<ExportButton report={buildReport} filename="gpu-sizing-workload" />
				<ShareButton payload={spec} />
				<HandoffMenu
					payload={spec}
					targets={[
						{ path: 'modelling', label: 'Modelling' },
						{ path: 'economics', label: 'Self-host vs API' },
						{ path: 'training', label: 'Training' }
					]}
				/>
			</div>
		</header>

		<HowTo>
			<p>
				This tab answers "<strong>how much hardware do I need to buy?</strong>". It is the reverse
				of the <a href="{base}/">Modelling</a> tab. There you pick the hardware and see how it runs.
				Here you say how busy your service will be and the promises you want to keep, and the app
				finds the <strong>smallest cluster of GPUs</strong> that can do it. Every box has an
				<span class="text-teal-400">i</span> button — tap it to learn what that box means.
			</p>
			<ol class="mt-2 ml-4 list-decimal space-y-1">
				<li>
					Pick your <strong>model</strong> (the AI brain) and a <strong>GPU</strong> to try (the chip
					that runs it). You can upload a model's config.json to add your own.
				</li>
				<li>
					Describe the <strong>demand</strong>: how many people use it at once (concurrency), and
					whether that number is the busiest moment or a calmer average. The app always plans for
					the busiest moment.
				</li>
				<li>
					Set the <strong>average length</strong> of questions (input) and answers (output), in tokens
					(word-pieces).
				</li>
				<li>
					Set your <strong>promises (SLOs)</strong>: how long a user waits for the first word
					(TTFT), and how fast words then stream (throughput).
				</li>
			</ol>
			<p class="mt-2">
				The app then shows the smallest cluster that both <strong>fits the model in memory</strong>
				and <strong>keeps every promise</strong>, plus how other GPUs would compare. The numbers are
				quick physics-based estimates, not a real benchmark.
			</p>
			<p class="mt-3 font-medium text-slate-300">Supported model types</p>
			<ul class="mt-1 ml-4 list-disc space-y-1">
				<li>
					<strong>Language models</strong> — chatbots and text models (Llama, Qwen, DeepSeek, GLM, Nemotron).
					Sized by concurrent users and per-user speed promises (TTFT and throughput).
				</li>
				<li>
					<strong>Diffusion models</strong> — image generators (SDXL, FLUX.1, SD 3.5, PixArt-Σ, DiT-XL)
					and video (CogVideoX). Sized by target images or clips per second and a max time per image/clip.
				</li>
				<li>
					<strong>Vision-language (VLM)</strong> — LLMs that also read images (Qwen2.5-VL, Pixtral, LLaVA-OneVision).
					Sized like an LLM plus an "images per request" knob that adds visual-patch tokens.
				</li>
				<li>
					<strong>Vision-language-action (VLA)</strong> — robotics policies (pi-zero, SmolVLA). Sized
					by target robots × control frequency and a max per-chunk latency.
				</li>
				<li>
					<strong>Speech (ASR)</strong> — Whisper family (large-v3, large-v3-turbo, medium). Sized by
					concurrent real-time streams and minimum RTF.
				</li>
				<li>
					<strong>Embeddings &amp; rerankers</strong> — text encoders (BGE-M3, Jina v3, E5-Mistral) and
					rerankers (BGE reranker). Sized by target docs (or query-doc pairs) per second and a max time
					per batch.
				</li>
				<li>
					<strong>World models (JEPA)</strong> — video models (V-JEPA, V-JEPA 2, and the action-conditioned
					V-JEPA 2-AC) that turn a clip into an embedding. Sized by target clips per second and a max
					time per clip.
				</li>
				<li>
					<strong>Autoregressive image (VAR)</strong> — image generators built like a language model (VAR-d30).
					Sized like a language model; set output tokens to the image's token count.
				</li>
			</ul>
			<p class="mt-2 text-xs text-slate-500">
				Pick the type from the grouped model menu. The demand and SLO controls change to match:
				users and tokens for language, images or clips per second for the others.
			</p>
			<div class="mt-3 rounded-lg border border-slate-700 bg-slate-800/40 p-3">
				<div class="mb-1 text-xs font-medium text-slate-300">Word list</div>
				<ul class="ml-4 list-disc space-y-1 text-xs text-slate-400">
					<li>
						<strong>Token</strong> — a word-piece. Models read and write tokens, not whole words. Roughly
						¾ of a word each.
					</li>
					<li>
						<strong>Concurrency</strong> — how many requests are being answered at the same moment.
					</li>
					<li>
						<strong>P90</strong> — the busy level that only the busiest 10% of moments go above. Higher
						than the average, lower than the all-time peak.
					</li>
					<li>
						<strong>TTFT</strong> — "time to first token": how long a user waits before the first word
						appears.
					</li>
					<li><strong>Throughput</strong> — how many tokens per second the answer streams at.</li>
					<li>
						<strong>SLO</strong> — a service promise you commit to keeping (like "under 1.5 seconds to
						first word").
					</li>
					<li>
						<strong>TP / PP</strong> — two ways to split one model across several GPUs: TP splits each
						layer's math, PP puts different layers on different GPUs.
					</li>
				</ul>
			</div>
		</HowTo>

		<div class="grid grid-cols-1 gap-6 lg:grid-cols-[320px_1fr]">
			<!-- inputs -->
			<aside
				class="lg:sticky lg:top-6 lg:max-h-[calc(100vh-3rem)] lg:self-start lg:overflow-y-auto lg:pr-1"
			>
				<div class="flex flex-col gap-5 rounded-xl border border-slate-700 bg-slate-900/60 p-5">
					<div>
						<div class="mb-1 flex items-center gap-1.5 text-sm text-slate-300">
							Model
							<InfoTip
								text="The AI model you want to serve. Bigger models are smarter but need more memory and more GPUs. This is what you're sizing the hardware for."
							/>
						</div>
						<select
			aria-label="Model"
							bind:value={spec.modelId}
							class="w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100"
						>
							{#each modelGroups as grp (grp.label)}
								<optgroup label={grp.label}>
									{#each grp.items as m (m.id)}<option value={m.id}
											>{m.name}{m.projected ? ' (projected)' : ''}</option
										>{/each}
								</optgroup>
							{/each}
						</select>
					</div>
					<div>
						<div class="mb-1 flex items-center gap-1.5 text-sm text-slate-300">
							Candidate GPU
							<InfoTip
								text="The chip you'd like to run on. The app figures out how many of these you'd need. Try a few — the comparison table below shows how each one stacks up."
							/>
						</div>
						<select
			aria-label="Candidate GPU"
							bind:value={spec.gpuId}
							onchange={() => (spec.fabricId = defaultFabricFor(spec.gpuId))}
							class="w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100"
						>
							{#each GPUS as g (g.id)}<option value={g.id}>{g.name}</option>{/each}
						</select>
					</div>
					<Segmented
						label="GPUs per node"
						bind:value={spec.gpusPerNode}
						options={cpu}
						info="How many GPUs sit inside one server. GPUs in the same server talk over a super-fast link (NVLink); GPUs in different servers talk over the slower network. Usually 8."
					/>
					<div>
						<div class="mb-1 flex items-center gap-1.5 text-sm text-slate-300">
							Network fabric
							<InfoTip
								text="The network that connects the servers to each other. It only matters when one model is split across more than one server — a faster fabric means less waiting when GPUs share data."
							/>
						</div>
						<select
			aria-label="Network fabric"
							bind:value={spec.fabricId}
							class="w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100"
						>
							{#each FABRICS as f (f.id)}<option value={f.id}>{f.label}</option>{/each}
						</select>
					</div>

					{#if isDiffusion}
						<div class="border-t border-slate-700 pt-4">
							<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
								Demand
							</div>
							<Slider
								label={isVideoDiff ? 'Target clips / sec' : 'Target images / sec'}
								bind:value={spec.targetImagesPerSec}
								min={0.1}
								max={200}
								step={0.1}
								display="{spec.targetImagesPerSec} {isVideoDiff ? 'clip' : 'img'}/s"
								info="How many {isVideoDiff
									? 'clips'
									: 'images'} the whole service must finish per second at peak. This is the main driver of cluster size — the app adds data-parallel replicas until it's met."
							/>
							<div class="mt-3"></div>
							<Segmented
								label="Resolution"
								bind:value={spec.resolution}
								options={RES.map((v) => ({ value: v, label: `${v}²` }))}
								info="Frame size in pixels per side. Larger frames have far more latent patches to denoise, so time and memory rise with the area — doubling the side is roughly 4× the work."
							/>
							<div class="mt-3"></div>
							{#if isVideoDiff}
								<Segmented
									label="Video length (latent frames)"
									bind:value={spec.frames}
									options={VID_FRAMES.map((v) => ({ value: v, label: String(v) }))}
									info="How long each clip is, in the model's compressed 'latent' frames. More frames means far more space-time patches, and because attention is quadratic, the compute climbs fast."
								/>
								<div class="mt-3"></div>
							{/if}
							<Slider
								label="Denoising steps"
								bind:value={spec.steps}
								min={1}
								max={100}
								step={1}
								info="How many denoising passes to run per {isVideoDiff
									? 'clip'
									: 'image'}. Total time is steps × step time. Fewer steps are faster but can look rougher."
							/>
							<div class="mt-3"></div>
							<Toggle
								label="Classifier-free guidance"
								bind:checked={spec.guidance}
								hint="cond + uncond pass"
								info="Runs two forward passes per step (with and without the prompt) to sharpen prompt-following, doubling the work. Distilled models (e.g. FLUX.1-dev) skip this."
							/>
						</div>

						<div class="border-t border-slate-700 pt-4">
							<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
								{isVideoDiff ? 'Per-clip SLO' : 'Per-image SLO'}
							</div>
							<Slider
								label={isVideoDiff ? 'Max sec / clip' : 'Max sec / image'}
								bind:value={spec.maxSecPerImage}
								min={0.5}
								max={120}
								step={0.5}
								display="{spec.maxSecPerImage} s"
								info="Your promise for the longest a single {isVideoDiff
									? 'clip'
									: 'image'} (or batch) may take. Lower is snappier but caps how large a batch the app can use, so it may need more GPUs. The app rejects any config slower than this."
							/>
						</div>
					{:else if isJepa}
						<div class="border-t border-slate-700 pt-4">
							<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
								Demand
							</div>
							<Slider
								label="Target clips / sec"
								bind:value={spec.targetClipsPerSec}
								min={0.1}
								max={500}
								step={0.1}
								display="{spec.targetClipsPerSec} clip/s"
								info="How many video clips the service must encode into embeddings per second at peak. The main driver of cluster size — the app adds data-parallel replicas until it's met."
							/>
							<div class="mt-3"></div>
							<Segmented
								label="Resolution"
								bind:value={spec.resolution}
								options={JEPA_RES.map((v) => ({ value: v, label: `${v}²` }))}
								info="Frame size in pixels per side. Larger frames mean more patches to encode, so time and memory rise with the area."
							/>
							<div class="mt-3"></div>
							<Segmented
								label="Frames per clip"
								bind:value={spec.frames}
								options={FRAMES.map((v) => ({ value: v, label: String(v) }))}
								info="How many video frames each clip carries. Frames group in twos (tubelets), so more frames mean more patches — and, because attention is quadratic, more than proportionally more compute."
							/>
						</div>

						<div class="border-t border-slate-700 pt-4">
							<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
								Per-clip SLO
							</div>
							<Slider
								label="Max sec / clip"
								bind:value={spec.maxSecPerClip}
								min={0.1}
								max={30}
								step={0.1}
								display="{spec.maxSecPerClip} s"
								info="Your promise for the longest a single clip (or batch) may take to encode. Lower is snappier but caps how large a batch the app can use, so it may need more GPUs. The app rejects any config slower than this."
							/>
						</div>
					{:else if isEncoder}
						<div class="border-t border-slate-700 pt-4">
							<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
								Demand
							</div>
							<Slider
								label="Target {modelOf?.encoder?.task === 'reranker' ? 'scores' : 'docs'} / sec"
								bind:value={spec.targetDocsPerSec}
								min={1}
								max={100000}
								step={1}
								display="{spec.targetDocsPerSec} {modelOf?.encoder?.task === 'reranker'
									? 'score'
									: 'doc'}/s"
								info="How many documents (or query-doc pairs, for a reranker) the service must encode per second at peak. The main driver of cluster size — the app adds replicas until it's met."
							/>
							<div class="mt-3"></div>
							<Slider
								label="Avg sequence length"
								bind:value={spec.inputTokens}
								min={64}
								max={modelOf?.encoder?.maxSeqLen ?? 8192}
								log
								display="{fmtSeq(spec.inputTokens)} tok"
								info="Typical tokens per document. Attention is quadratic in this, so short passages are much cheaper than long ones."
							/>
						</div>

						<div class="border-t border-slate-700 pt-4">
							<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
								Per-doc SLO
							</div>
							<Slider
								label="Max sec / doc (or batch)"
								bind:value={spec.maxSecPerDoc}
								min={0.01}
								max={10}
								step={0.01}
								display="{spec.maxSecPerDoc} s"
								info="Longest acceptable latency for one document or one batch pass. Lower caps how big a batch can be, so it may need more GPUs. The app rejects any config slower than this."
							/>
						</div>
					{:else if isAsr}
						<div class="border-t border-slate-700 pt-4">
							<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
								Demand
							</div>
							<Slider
								label="Concurrent audio streams"
								bind:value={spec.targetStreams}
								min={1}
								max={10000}
								step={1}
								display="{spec.targetStreams} streams"
								info="How many simultaneous real-time audio streams the cluster must transcribe (call centre, live captioning, meeting bots)."
							/>
						</div>

						<div class="border-t border-slate-700 pt-4">
							<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
								Per-stream SLO
							</div>
							<Slider
								label="Min real-time factor"
								bind:value={spec.minRtf}
								min={0.5}
								max={20}
								step={0.1}
								display="{spec.minRtf}×"
								info="Minimum RTF per stream (audio duration ÷ processing time). 1× keeps up with live audio; higher gives headroom for jitter and re-tries. Under 1× means the transcription falls behind."
							/>
						</div>
					{:else if isVla}
						<div class="border-t border-slate-700 pt-4">
							<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
								Demand
							</div>
							<Slider
								label="Robots to drive"
								bind:value={spec.targetRobots}
								min={1}
								max={500}
								step={1}
								display="{spec.targetRobots} robots"
								info="How many physical robots the cluster must serve, each running at the model's control frequency ({modelOf
									?.vla?.controlHz ?? 50} Hz). Each robot consumes ~controlHz actions per second."
							/>
						</div>

						<div class="border-t border-slate-700 pt-4">
							<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
								Per-chunk SLO
							</div>
							<Slider
								label="Max sec / chunk"
								bind:value={spec.maxSecPerControl}
								min={0.05}
								max={5}
								step={0.05}
								display="{spec.maxSecPerControl} s"
								info="Longest acceptable time for one inference cycle (observation → {modelOf?.vla
									?.chunkSize ?? 50}-action chunk). Must be at most chunkSize/controlHz = {(
									(modelOf?.vla?.chunkSize ?? 50) / (modelOf?.vla?.controlHz ?? 50)
								).toFixed(1)} s or the robot outruns the model."
							/>
						</div>
					{:else}
						<div class="border-t border-slate-700 pt-4">
							<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
								Demand
							</div>
							<Slider
								label="Concurrent requests"
								bind:value={spec.concurrency}
								min={1}
								max={8192}
								step={1}
								info="How many people are using the service at the same time. More users at once means more work, so more GPUs. This is the single biggest driver of cluster size."
							/>
							<div class="mt-3"></div>
							<Segmented
								label="Load basis"
								bind:value={spec.basis}
								options={[
									{ value: 'average', label: 'Average' },
									{ value: 'p90', label: 'P90' },
									{ value: 'peak', label: 'Peak' }
								]}
								info="Is the number above a typical moment or the busiest moment? The cluster must survive the busiest moment, so the app scales your number up to it: 'Peak' is already the worst case (×1), 'P90' is close (×1.2), and 'Average' is far below the peak so it gets the biggest boost (×2). A calmer everyday number hides bigger spikes, which is why it needs the largest bump."
							/>
							<p class="mt-1 text-[10px] text-slate-500">
								Sizing always targets the peak. Your number is treated as the {spec.basis}, so the
								implied peak to provision for is
								<span class="text-slate-300">{chosen.provisionedConcurrency} concurrent</span>
								(average → ×2, p90 → ×1.2, peak → ×1: a weaker statistic implies a higher peak).
							</p>
						</div>

						<Slider
							label="Avg input tokens"
							bind:value={spec.inputTokens}
							min={64}
							max={1048576}
							log
							display="{fmtSeq(spec.inputTokens)} tok"
							info="How long the typical question/prompt is, in tokens (word-pieces). Longer inputs take more work to read and use more memory, so they make the first word take longer."
						/>
						<Slider
							label="Avg output tokens"
							bind:value={spec.outputTokens}
							min={64}
							max={65536}
							log
							display="{fmtSeq(spec.outputTokens)} tok"
							info="How long the typical answer is, in tokens. The model makes one token at a time, so longer answers take proportionally longer and hold their memory for longer. Reasoning models emit long chains of thought — 32k–64k output is normal."
						/>
						{#if isVlm}
							<Slider
								label="Images per request"
								bind:value={spec.imagesPerRequest}
								min={0}
								max={8}
								step={1}
								info="Images each request carries. Each adds ~{modelOf?.vlm
									?.tokensPerImage} extra input tokens (visual patches), inflating prefill compute and KV memory."
							/>
						{/if}

						<div class="border-t border-slate-700 pt-4">
							<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
								Per-user SLO
							</div>
							<Slider
								label="Max TTFT"
								bind:value={spec.ttftTargetMs}
								min={10}
								max={30000}
								step={10}
								display="{spec.ttftTargetMs} ms"
								info="Your promise for the longest a user should wait before the first word appears (time to first token), in milliseconds. Lower is snappier but harder to hit. The app rejects any cluster slower than this. Long-context prompts (100k+ tokens) can push prefill into the multi-second range, so loosen this for those."
							/>
							<div class="mt-3"></div>
							<Slider
								label="Min throughput"
								bind:value={spec.throughputTargetTps}
								min={5}
								max={500}
								step={5}
								display="{spec.throughputTargetTps} tok/s"
								info="Your promise for how fast the answer streams to each user, in tokens per second. About 10 tok/s reads like a person talking; higher feels instant. Real single-user decode tops out in the low hundreds even with speculative decoding. The app rejects any cluster slower than this per user."
							/>
						</div>
					{/if}

					<Segmented
						label="Weight format"
						bind:value={spec.weightFormatId}
						options={wfOpts}
						info="How precisely the model's numbers are stored. Fewer bits (FP8, NVFP4) shrink the model so it needs less memory and fewer GPUs, at a small risk to quality. BF16 is full quality but twice the size of FP8. Formats the selected GPU can't run natively are greyed out."
					/>
					{#if !isDiffusion}
						<Segmented
							label="KV precision"
							bind:value={spec.kvBits}
							options={[
								{ value: 16, label: 'FP16' },
								{ value: 8, label: 'FP8' }
							]}
							info="How precisely the model's short-term memory of the conversation (the KV cache) is stored. FP8 halves that memory versus FP16, letting each GPU handle more users at once."
						/>
					{/if}

					{#if !isDiffusion && !isJepa}
						<div class="rounded-lg border border-slate-700 bg-slate-800/40 p-3">
							<Toggle
								label="Disaggregated prefill / decode"
								bind:checked={disaggEnabled}
								hint="separate pools, KV over fabric"
								info="Modern serving splits the work into two independent pools: prefill (reads the prompt — compute-bound, sets TTFT) and decode (writes the answer — memory-bound, sets streaming speed). Each pool gets its own GPU count, TP/PP and batch, joined by a KV-cache hand-off over the network. Helps most when prefill and decode contend for the same GPUs and TTFT is tight."
							/>
						</div>
					{/if}

					{#if !isDiffusion && !isJepa && !isEncoder && !isVla && !isAsr}
						<Slider
							label="Cached prefix"
							bind:value={spec.cachedPrefixFrac}
							min={0}
							max={0.99}
							step={0.01}
							display="{Math.round(spec.cachedPrefixFrac * 100)}%"
							info="Share of each prompt that's an already-cached shared prefix (system prompt, RAG boilerplate, chat history). Its KV is reused, so prefill only processes the uncached rest — lowering TTFT, which can let a tighter TTFT SLO be met with fewer GPUs. Chunked prefill is the complementary scheduler trick that interleaves prefill chunks with decode so generations don't stall."
						/>
					{/if}

					{#if !isDiffusion && !isJepa && !isEncoder && !isVla && !isAsr}
						<div class="rounded-lg border border-slate-700 bg-slate-800/40 p-3">
							<Toggle
								label="Speculative decoding"
								bind:checked={spec.specDecode}
								hint="draft + verify, faster per-user"
								info="A small draft model proposes several tokens each step; the target model verifies them in a single pass. Accepted tokens come almost free, so per-user decode speeds up — meaning fewer replicas meet the same throughput SLO. The gain depends on how often drafts are accepted."
							/>
							{#if spec.specDecode}
								<div class="mt-3">
									<Slider
										label="Draft accept rate"
										bind:value={spec.draftAcceptRate}
										min={0.3}
										max={0.95}
										step={0.05}
										display="{(spec.draftAcceptRate * 100).toFixed(0)}%"
										info="How often a drafted token is accepted by the target model. Higher acceptance means more tokens land per verify pass, so a bigger speed-up. Typical EAGLE/Medusa-style drafts land 60–80%."
									/>
									<div class="mt-3"></div>
									<Slider
										label="Draft tokens (γ)"
										bind:value={spec.specTokens}
										min={1}
										max={8}
										step={1}
										info="How many tokens the draft proposes per step. More can mean a bigger win when acceptance is high, but wasted work when it's low."
									/>
								</div>
							{/if}
						</div>
					{/if}

					<div class="border-t border-slate-700 pt-4">
						<div class="mb-1 flex items-center justify-center gap-1.5 text-[10px] text-slate-500">
							<span>Not in the list?</span>
							<InfoTip
								text="Add your own model by uploading its HuggingFace config.json. The app reads its size and shape and adds it to the model picker so you can size hardware for it."
							/>
						</div>
						<label
							class="block cursor-pointer rounded-lg border border-dashed border-slate-600 px-3 py-2 text-center text-xs text-slate-400 hover:bg-slate-800"
						>
							+ Add model from JSON
							<input
								type="file"
								accept=".json,application/json,text/plain"
								class="hidden"
								onchange={onModelFile}
							/>
						</label>
						{#if uploadError}<p class="mt-1 text-[10px] text-red-400">{uploadError}</p>{/if}
					</div>
				</div>
			</aside>

			<!-- result -->
			<main class="flex flex-col gap-6">
				{#if !chosen.feasible}
					<div class="rounded-xl border border-red-500/50 bg-red-500/10 p-4 text-sm text-red-300">
						<span class="font-medium">{gpu.name} can't meet these SLOs</span> — {chosen.reason}. See
						feasible alternatives below.
					</div>
				{/if}

				{#if chosen.feasible && p && isDiffusion}
					<div class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
						<StatCard
							label="Cluster"
							value={String(chosen.numGpus)}
							unit="GPU"
							accent={COLORS.memPipe}
							sub="{chosen.numNodes} node × {spec.gpusPerNode} · {gpu.name}"
							info="The total number of these GPUs you need to buy to hit the target images/sec and keep each image under the latency SLO. This is the headline answer — the smallest count that works."
						/>
						<StatCard
							label="Parallelism"
							value="TP{chosen.tp} · PP{chosen.pp}"
							sub="{chosen.dp} replicas · {chosen.replicaGpus} GPU/replica"
							info="How one copy of the model is split across GPUs: TP splits each layer's math across GPUs (kept within a node), PP puts different layers on different GPUs. That group is then duplicated into DP identical copies to produce more images at once."
						/>
						<StatCard
							label="Batch / replica"
							value={String(chosen.batchPerReplica)}
							sub="{isVideoDiff ? 'clips' : 'images'} per pass"
							info="How many {isVideoDiff
								? 'clips'
								: 'images'} one copy of the model denoises together. Bigger batches use each GPU more fully but take longer per pass — the solver picks the largest batch still under your latency SLO."
						/>
						<StatCard
							label={isVideoDiff ? 'Clips / sec' : 'Images / sec'}
							value={(chosen.imagesPerSec ?? 0).toFixed((chosen.imagesPerSec ?? 0) < 1 ? 2 : 1)}
							unit={isVideoDiff ? 'clip/s' : 'img/s'}
							accent="#3ff0b8"
							sub="target ≥ {spec.targetImagesPerSec} {isVideoDiff ? 'clip' : 'img'}/s"
							info="How many {isVideoDiff
								? 'clips'
								: 'images'} the whole cluster finishes per second. It must meet your target — green means it does. Raised by bigger batches and more replicas."
						/>
						<StatCard
							label={isVideoDiff ? 'Sec / clip' : 'Sec / image'}
							value={(chosen.secPerImage ?? 0).toFixed(2)}
							unit="s"
							accent="#3ff0b8"
							sub="target ≤ {spec.maxSecPerImage} s"
							info="Wall-clock time for one {isVideoDiff
								? 'clip'
								: 'image'} (or one batch pass). It must stay under your latency promise — green means it does."
						/>
						<StatCard
							label="Fits / GPU"
							value={p.perGpu.fits ? 'Yes' : 'No'}
							accent={p.perGpu.fits ? '#3ff0b8' : '#ff3b52'}
							sub="{p.bottleneck}-bound"
							info="Whether the denoiser, its text encoders, and working space all fit inside one GPU's memory for this layout. 'bottleneck' names what limits speed: memory bandwidth, compute, or the network."
						/>
					</div>
				{:else if chosen.feasible && p && isAsr}
					<div class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
						<StatCard
							label="Cluster"
							value={String(chosen.numGpus)}
							unit="GPU"
							accent={COLORS.memPipe}
							sub="{chosen.numNodes} node × {spec.gpusPerNode} · {gpu.name}"
							info="Smallest cluster of this GPU that transcribes your stream count in real time under the RTF SLO."
						/>
						<StatCard
							label="Parallelism"
							value="TP{chosen.tp} · PP{chosen.pp}"
							sub="{chosen.dp} replicas · {chosen.replicaGpus} GPU/replica"
							info="Whisper is small enough that TP1/PP1 with many DP replicas is typical."
						/>
						<StatCard
							label="Batch / replica"
							value={String(chosen.batchPerReplica)}
							sub="parallel streams"
							info="Streams processed together per inference cycle."
						/>
						<StatCard
							label="Streams served"
							value={(chosen.streamsServed ?? 0).toFixed(0)}
							accent="#3ff0b8"
							sub="target ≥ {spec.targetStreams}"
							info="Concurrent real-time streams the cluster can sustain."
						/>
						<StatCard
							label="RTF / replica"
							value={(chosen.rtf ?? 0).toFixed(1)}
							unit="×"
							accent="#3ff0b8"
							sub="target ≥ {spec.minRtf}×"
							info="Real-time factor a single replica achieves for the sized batch."
						/>
						<StatCard
							label="Fits / GPU"
							value={p.perGpu.fits ? 'Yes' : 'No'}
							accent={p.perGpu.fits ? '#3ff0b8' : '#ff3b52'}
							sub="{p.bottleneck}-bound"
							info="Whether encoder + decoder + KV fit per-GPU."
						/>
					</div>
				{:else if chosen.feasible && p && isVla}
					<div class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
						<StatCard
							label="Cluster"
							value={String(chosen.numGpus)}
							unit="GPU"
							accent={COLORS.memPipe}
							sub="{chosen.numNodes} node × {spec.gpusPerNode} · {gpu.name}"
							info="Smallest cluster of this GPU that drives your robot fleet at the model's control frequency and stays under the per-chunk latency SLO."
						/>
						<StatCard
							label="Parallelism"
							value="TP{chosen.tp} · PP{chosen.pp}"
							sub="{chosen.dp} replicas · {chosen.replicaGpus} GPU/replica"
							info="VLAs are usually small enough that TP1/PP1 with many DP replicas is the win — each replica drives one or a few robots."
						/>
						<StatCard
							label="Batch / replica"
							value={String(chosen.batchPerReplica)}
							sub="robots per replica"
							info="Robot observations processed together per inference cycle."
						/>
						<StatCard
							label="Robots driven"
							value={(chosen.robotsDriven ?? 0).toFixed(0)}
							accent="#3ff0b8"
							sub="target ≥ {spec.targetRobots}"
							info="Robots the cluster can serve at {modelOf?.vla?.controlHz ??
								50} Hz. Green if it meets your target."
						/>
						<StatCard
							label="Sec / chunk"
							value={((chosen.secPerControl ?? 0) * 1000).toFixed(0)}
							unit="ms"
							accent="#3ff0b8"
							sub="target ≤ {(spec.maxSecPerControl * 1000).toFixed(0)} ms"
							info="Per-chunk inference latency (one observation → chunkSize actions). Green if under your SLO."
						/>
						<StatCard
							label="Fits / GPU"
							value={p.perGpu.fits ? 'Yes' : 'No'}
							accent={p.perGpu.fits ? '#3ff0b8' : '#ff3b52'}
							sub="{p.bottleneck}-bound"
							info="Whether the backbone + vision encoder + action expert fits per-GPU."
						/>
					</div>
				{:else if chosen.feasible && p && isEncoder}
					<div class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
						<StatCard
							label="Cluster"
							value={String(chosen.numGpus)}
							unit="GPU"
							accent={COLORS.memPipe}
							sub="{chosen.numNodes} node × {spec.gpusPerNode} · {gpu.name}"
							info="Smallest cluster of this GPU that hits your target docs/sec and stays under the latency SLO."
						/>
						<StatCard
							label="Parallelism"
							value="TP{chosen.tp} · PP{chosen.pp}"
							sub="{chosen.dp} replicas · {chosen.replicaGpus} GPU/replica"
							info="How one copy of the encoder is split, and how many replicas run in parallel to hit the target rate. Encoders are usually small enough that TP1/PP1 with lots of DP replicas is the win."
						/>
						<StatCard
							label="Batch / replica"
							value={String(chosen.batchPerReplica)}
							sub="{modelOf?.encoder?.task === 'reranker' ? 'pairs' : 'docs'} per pass"
							info="How many documents one replica encodes per forward. Encoders love big batches — the solver picks the largest that still meets your latency SLO."
						/>
						<StatCard
							label={modelOf?.encoder?.task === 'reranker' ? 'Scores / sec' : 'Docs / sec'}
							value={(chosen.docsPerSec ?? 0).toFixed((chosen.docsPerSec ?? 0) < 10 ? 1 : 0)}
							unit={modelOf?.encoder?.task === 'reranker' ? 'score/s' : 'doc/s'}
							accent="#3ff0b8"
							sub="target ≥ {spec.targetDocsPerSec} /s"
							info="How many documents (or pairs) the cluster encodes per second. Must meet your target — green means it does."
						/>
						<StatCard
							label="Sec / batch"
							value={((chosen.secPerDoc ?? 0) * 1000).toFixed(1)}
							unit="ms"
							accent="#3ff0b8"
							sub="target ≤ {(spec.maxSecPerDoc * 1000).toFixed(0)} ms"
							info="Wall-clock time for one batch pass. Must stay under your Max sec/doc promise — green means it does."
						/>
						<StatCard
							label="Fits / GPU"
							value={p.perGpu.fits ? 'Yes' : 'No'}
							accent={p.perGpu.fits ? '#3ff0b8' : '#ff3b52'}
							sub="{p.bottleneck}-bound"
							info="Whether the encoder plus working space fits in each GPU's memory. Encoders are small, so this is usually a big yes."
						/>
					</div>
				{:else if chosen.feasible && p && isJepa}
					<div class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
						<StatCard
							label="Cluster"
							value={String(chosen.numGpus)}
							unit="GPU"
							accent={COLORS.memPipe}
							sub="{chosen.numNodes} node × {spec.gpusPerNode} · {gpu.name}"
							info="The total number of these GPUs you need to buy to hit the target clips/sec and keep each clip under the latency SLO. This is the headline answer — the smallest count that works."
						/>
						<StatCard
							label="Parallelism"
							value="TP{chosen.tp} · PP{chosen.pp}"
							sub="{chosen.dp} replicas · {chosen.replicaGpus} GPU/replica"
							info="How one copy of the model is split across GPUs: TP splits each layer's math across GPUs (kept within a node), PP puts different layers on different GPUs. That group is then duplicated into DP identical copies to encode more clips at once."
						/>
						<StatCard
							label="Batch / replica"
							value={String(chosen.batchPerReplica)}
							sub="clips per pass"
							info="How many clips one copy of the model encodes together. Bigger batches use each GPU more fully (more clips/sec) but take longer per pass — the solver picks the largest batch still under your latency SLO."
						/>
						<StatCard
							label="Clips / sec"
							value={(chosen.clipsPerSec ?? 0).toFixed((chosen.clipsPerSec ?? 0) < 1 ? 2 : 1)}
							unit="clip/s"
							accent="#3ff0b8"
							sub="target ≥ {spec.targetClipsPerSec} clip/s"
							info="How many clips the whole cluster encodes per second. It must meet your target — green means it does. Raised by bigger batches and more replicas."
						/>
						<StatCard
							label="Sec / clip"
							value={(chosen.secPerClip ?? 0).toFixed(3)}
							unit="s"
							accent="#3ff0b8"
							sub="target ≤ {spec.maxSecPerClip} s"
							info="Wall-clock time for one clip (or one batch pass). It must stay under your Max sec/clip promise — green means it does."
						/>
						<StatCard
							label="Fits / GPU"
							value={p.perGpu.fits ? 'Yes' : 'No'}
							accent={p.perGpu.fits ? '#3ff0b8' : '#ff3b52'}
							sub="{p.bottleneck}-bound"
							info="Whether the encoder, its predictor, and working space all fit inside one GPU's memory for this layout. 'bottleneck' names what limits speed: memory bandwidth, compute, or the network."
						/>
					</div>
				{:else if chosen.feasible && p}
					<div class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
						<StatCard
							label="Cluster"
							value={String(chosen.numGpus)}
							unit="GPU"
							accent={COLORS.memPipe}
							sub="{chosen.numNodes} node × {spec.gpusPerNode} · {gpu.name}"
							info="The total number of these GPUs you need to buy to serve this workload and keep every promise. This is the headline answer — the smallest count that works."
						/>
						<StatCard
							label="Parallelism"
							value="TP{chosen.tp} · PP{chosen.pp}{chosen.ep > 1 ? ` · EP${chosen.ep}` : ''}"
							sub="{chosen.dp} replicas · {chosen.replicaGpus} GPU/replica"
							info="How one copy of the model is split across GPUs: TP splits each layer's math across GPUs (kept within a node), PP puts different layers on different GPUs (can span nodes), and EP spreads a mixture-of-experts model's experts across GPUs (can span nodes — this is what lets huge MoE models fit). That group is then duplicated into DP identical copies to serve more users."
						/>
						<StatCard
							label="Batch / replica"
							value={String(chosen.batchPerReplica)}
							sub="serves {chosen.batchPerReplica * chosen.dp} concurrent"
							info="How many users one copy of the model handles at the same time. Bigger batches use each GPU more fully, but too big and per-user speed drops. The solver picks the batch that still keeps your promises."
						/>
						<StatCard
							label="Cluster throughput"
							value={fmtTps(chosen.clusterTps)}
							unit="tok/s"
							sub="aggregate generation"
							info="Total words-per-second the whole cluster produces across all users combined. A measure of how much work the cluster gets through overall."
						/>
						<StatCard
							label="Per-user TTFT"
							value={fmtMs(chosen.ttftMs)}
							accent="#3ff0b8"
							sub="target ≤ {spec.ttftTargetMs} ms"
							info="How long one user waits for the first word on this cluster. It must stay under your Max TTFT promise — green means it does. Measured on a single request on its own."
						/>
						<StatCard
							label="Per-user throughput"
							value={fmtTps(chosen.perUserTps)}
							unit="tok/s"
							accent="#3ff0b8"
							sub="target ≥ {spec.throughputTargetTps} tok/s"
							info="How fast the answer streams to one user, in tokens per second. It must stay above your Min throughput promise — green means it does."
						/>
						<StatCard
							label="Full request"
							value={fmtMs(chosen.fullReqMs)}
							sub="TTFT + {spec.outputTokens} tok"
							info="Total time for one whole answer: the wait for the first word plus streaming all the remaining words. What a user experiences end to end."
						/>
						<StatCard
							label="Fits / GPU"
							value={p.perGpu.fits ? 'Yes' : 'No'}
							accent={p.perGpu.fits ? '#3ff0b8' : '#ff3b52'}
							sub="{p.bottleneck}-bound decode"
							info="Whether the model, its short-term memory, and working space all fit inside one GPU's memory for this layout. 'bottleneck' names what limits speed: memory bandwidth, compute, or the network."
						/>
					</div>
				{/if}
				{#if disagg}
					<section class="rounded-xl border border-teal-700/40 bg-teal-500/[0.03] p-5">
						<div class="mb-3 flex items-center gap-1.5 text-sm font-medium text-slate-200">
							Disaggregated prefill / decode pools
							<InfoTip
								text="Two independent pools sized side-by-side. Prefill takes prompts and produces the first token + KV cache; decode takes over and streams the answer. TTFT = prefill time + KV transfer over the fabric."
							/>
						</div>
						{#if !disagg.feasible}
							<p class="text-sm text-red-300">
								{disagg.reason}. Falls back to the aggregated sizing above.
							</p>
						{:else}
							<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
								<div class="rounded-lg border border-slate-700 bg-slate-900/60 p-4">
									<div class="mb-2 text-xs tracking-wide text-slate-500 uppercase">
										Prefill pool <span class="text-slate-600">— compute-bound</span>
									</div>
									<div class="flex items-baseline justify-between">
										<span class="text-slate-300">GPUs</span><span
											class="font-mono text-lg text-slate-100"
											>{disagg.prefill.gpus}
											<span class="text-xs text-slate-500">/ {disagg.prefill.nodes}n</span></span
										>
									</div>
									<div class="flex items-baseline justify-between text-sm">
										<span class="text-slate-400">Parallelism</span><span
											class="font-mono text-slate-300"
											>TP{disagg.prefill.tp} · PP{disagg.prefill.pp}{disagg.prefill.ep > 1
												? ` · EP${disagg.prefill.ep}`
												: ''}</span
										>
									</div>
									<div class="flex items-baseline justify-between text-sm">
										<span class="text-slate-400">Replicas</span><span
											class="font-mono text-slate-300"
											>{disagg.prefill.replicas} × {disagg.prefill.replicaGpus} GPU</span
										>
									</div>
									<div class="flex items-baseline justify-between text-sm">
										<span class="text-slate-400">Per prompt</span><span
											class="font-mono text-slate-300"
											>{disagg.prefill.msPerPrompt.toFixed(0)} ms · {disagg.prefill.promptsPerSecPerReplica.toFixed(
												1
											)} req/s/replica</span
										>
									</div>
								</div>
								<div class="rounded-lg border border-slate-700 bg-slate-900/60 p-4">
									<div class="mb-2 text-xs tracking-wide text-slate-500 uppercase">
										Decode pool <span class="text-slate-600">— memory-bound</span>
									</div>
									<div class="flex items-baseline justify-between">
										<span class="text-slate-300">GPUs</span><span
											class="font-mono text-lg text-slate-100"
											>{disagg.decode.gpus}
											<span class="text-xs text-slate-500">/ {disagg.decode.nodes}n</span></span
										>
									</div>
									<div class="flex items-baseline justify-between text-sm">
										<span class="text-slate-400">Parallelism</span><span
											class="font-mono text-slate-300"
											>TP{disagg.decode.tp} · PP{disagg.decode.pp}{disagg.decode.ep > 1
												? ` · EP${disagg.decode.ep}`
												: ''}</span
										>
									</div>
									<div class="flex items-baseline justify-between text-sm">
										<span class="text-slate-400">Replicas</span><span
											class="font-mono text-slate-300"
											>{disagg.decode.replicas} × {disagg.decode.replicaGpus} GPU · batch {disagg
												.decode.batchPerReplica}</span
										>
									</div>
									<div class="flex items-baseline justify-between text-sm">
										<span class="text-slate-400">Per user</span><span
											class="font-mono text-slate-300"
											>{disagg.decode.perUserTps.toFixed(1)} tok/s · {(
												disagg.decode.clusterTps / 1000
											).toFixed(1)}k tok/s cluster</span
										>
									</div>
								</div>
							</div>
							<div class="mt-3 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
								<div class="rounded border border-slate-800 bg-slate-900/40 p-2">
									<div class="text-slate-500">Total GPUs</div>
									<div class="font-mono text-slate-100">{disagg.totalGpus}</div>
								</div>
								<div class="rounded border border-slate-800 bg-slate-900/40 p-2">
									<div class="text-slate-500">TTFT (prefill + KV)</div>
									<div
										class="font-mono {disagg.ttftMs <= spec.ttftTargetMs
											? 'text-emerald-300'
											: 'text-red-300'}"
									>
										{disagg.ttftMs.toFixed(0)} ms
									</div>
								</div>
								<div class="rounded border border-slate-800 bg-slate-900/40 p-2">
									<div class="text-slate-500">KV transfer</div>
									<div class="font-mono text-slate-300">{disagg.kvTransferMs.toFixed(1)} ms</div>
								</div>
								<div class="rounded border border-slate-800 bg-slate-900/40 p-2">
									<div class="text-slate-500">Steady-state</div>
									<div class="font-mono text-slate-300">
										{disagg.promptsPerSec.toFixed(1)} req/s
									</div>
								</div>
							</div>
							<p class="mt-3 text-[10px] text-slate-500">
								First-order model: prefill and decode are sized independently for their bottlenecks.
								Real deployments also gain from prefix caching, chunked prefill, and higher batch on
								decode — not modelled here.
							</p>
						{/if}
					</section>
				{/if}
				{#if chosen.feasible && p}
					<section><Topology {p} hasNvlink={gpu.hasNvlink} /></section>

					<section class="rounded-xl border border-slate-700 bg-slate-900/60 p-5">
						<div class="grid grid-cols-1 items-stretch gap-4 md:grid-cols-[1fr_1.4fr]">
							<div class="min-h-[22rem]">
								<MemoryStack
									segments={p.segments}
									capacity={p.perGpu.capacity}
									used={p.perGpu.used}
									fits={p.perGpu.fits}
								/>
							</div>
							<div><ComputeDie units={dieUnits} /></div>
						</div>
					</section>

					<CostPanel
						gpuId={spec.gpuId}
						numGpus={chosen.numGpus}
						metric={costMetric}
						bind:useEstimated={pricing.useEstimatedPricing}
						bind:manualPerGpuHour={pricing.manualPerGpuHour}
						bind:purchasing={pricing.purchasing}
					/>

					<EnergyPanel
						gpuId={spec.gpuId}
						numGpus={chosen.numGpus}
						util={p ? Math.max(p.memBwFrac, p.computeFrac) : 0.8}
						metric={costMetric}
						bind:region
						bind:method={emissionsMethod}
					/>

					<LoadPanel
						gpuId={spec.gpuId}
						numGpus={chosen.numGpus}
						weightBytes={((modelOf?.params ?? 0) *
							(WEIGHT_FORMATS_BY_ID.get(spec.weightFormatId)?.bitsPerWeight ?? 16)) /
							8}
					/>
				{:else}
					<!-- infeasible: show the best-attempt layout, memory pressure + the binding bottleneck flagged -->
					<section class="rounded-xl border border-slate-700 bg-slate-900/60 p-5">
						<div class="mb-3 flex flex-wrap items-center gap-2 text-sm font-medium text-slate-200">
							<span class="flex items-center gap-1.5">
								Best-attempt layout on one {gpu.name}
								<InfoTip
									text="The workload can't be met on this GPU, but here's what a single GPU looks like running this model as hard as it can — the largest batch that fits at the best parallel layout the solver found. The solver shards the model to make it fit, so the memory box usually stays under the line (it overflows only when the model can't fit at any layout). The red-flagged die block is the resource that's maxed out — the reason the SLO can't be met. It isn't a hidden cluster cap."
								/>
							</span>
							<span
								class="rounded-md border border-red-500/50 bg-red-500/10 px-2 py-0.5 text-xs text-red-300"
							>
								Bottleneck: {bottleneckLabel}
							</span>
						</div>
						<div class="grid grid-cols-1 items-stretch gap-4 md:grid-cols-[1fr_1.4fr]">
							<div class="min-h-[22rem]">
								<MemoryStack
									segments={viz.segments}
									capacity={viz.perGpu.capacity}
									used={viz.perGpu.used}
									fits={viz.perGpu.fits}
									warnNearFull
								/>
							</div>
							<div><ComputeDie units={dieUnits} /></div>
						</div>
						<p class="mt-3 text-[10px] text-slate-500">
							Per-GPU view at TP{chosen.config.tp} · PP{chosen.config.pp}{chosen.config.ep > 1
								? ` · EP${chosen.config.ep}`
								: ''}, batch {chosen.config.batchSize} (largest that fits, to load the die).
							{#if !viz.perGpu.fits}The model overflows one GPU's memory even at batch 1 at this
								layout — it can't fit at any layout.{:else}The model fits when sharded, so memory
								isn't the limit here — the flagged block is.{/if}
						</p>
					</section>
				{/if}

				<!-- GPU comparison -->
				<section class="rounded-xl border border-slate-700 bg-slate-900/60 p-5">
					<div class="mb-3 flex flex-wrap items-center justify-between gap-3">
						<div class="flex items-center gap-1.5 text-sm font-medium text-slate-200">
							GPU comparison — cluster to meet this workload
							<InfoTip
								text="The same workload sized on every GPU. Each row is the smallest cluster of that chip that keeps your promises, or the reason it can't. Click a row to load that GPU into the tool (it carries across tabs). Cost uses the on-demand market median for each GPU. Pick what to optimise for and the top feasible row is recommended (★)."
							/>
							<span class="text-[10px] font-normal text-slate-500"
								>· click a row to use that GPU</span
							>
						</div>
						<div class="w-full sm:w-72">
							<Segmented
								label="Optimise for"
								bind:value={objective}
								options={[
									{ value: 'gpus', label: 'Fewest GPUs' },
									{ value: 'hourly', label: '$/hour' },
									{ value: 'unit', label: unitCostLabel }
								]}
								info="How to rank the feasible GPUs. Fewest GPUs = simplest cluster. $/hour = cheapest to run per hour. Per-unit = cheapest per 1M tokens (or per 1k images/clips) — the best efficiency for the money."
							/>
						</div>
					</div>
					<div class="overflow-x-auto">
						<table class="w-full text-sm">
							<thead class="text-left text-xs text-slate-500">
								<tr
									><th class="py-1 pr-4">GPU</th><th class="py-1 pr-4">GPUs</th><th
										class="py-1 pr-4">Nodes</th
									><th class="py-1 pr-4">TP·PP·DP</th><th class="py-1 pr-4">$/hour</th><th
										class="py-1 pr-4">{unitCostLabel}</th
									><th class="py-1">Result</th></tr
								>
							</thead>
							<tbody>
								{#each ranked.rows as { r, hr, unit } (r.gpuId)}
									<tr
										role="button"
										tabindex="0"
										title="Load {r.gpuName} into the tool"
										onclick={() => {
											spec.gpuId = r.gpuId;
											spec.fabricId = defaultFabricFor(r.gpuId);
										}}
										onkeydown={(e) => {
											if (e.key === 'Enter' || e.key === ' ') {
												e.preventDefault();
												spec.gpuId = r.gpuId;
												spec.fabricId = defaultFabricFor(r.gpuId);
											}
										}}
										class="cursor-pointer border-t border-slate-800 hover:bg-slate-800/60 {r.gpuId ===
										spec.gpuId
											? 'bg-slate-800/40'
											: ''}"
									>
										<td
											class="py-1.5 pr-4 {r.gpuId === spec.gpuId
												? 'font-medium text-teal-300'
												: 'text-slate-200'}"
										>
											{#if r.gpuId === ranked.bestGpuId}<span
													class="mr-1 text-amber-400"
													title="Recommended for the chosen objective">★</span
												>{/if}{r.gpuName}
										</td>
										{#if r.feasible}
											<td class="py-1.5 pr-4 font-mono text-slate-100">{r.numGpus}</td>
											<td class="py-1.5 pr-4 font-mono text-slate-400">{r.numNodes}</td>
											<td class="py-1.5 pr-4 font-mono text-slate-400"
												>{r.tp}·{r.pp}{r.ep > 1 ? `·E${r.ep}` : ''}·{r.dp}</td
											>
											<td class="py-1.5 pr-4 font-mono text-slate-300"
												>{hr == null ? '—' : fmtUsd(hr)}</td
											>
											<td class="py-1.5 pr-4 font-mono text-slate-300"
												>{unit == null ? '—' : fmtUsdSmall(unit)}</td
											>
											<td class="py-1.5 text-emerald-400">meets SLOs</td>
										{:else}
											<td class="py-1.5 pr-4 text-slate-600" colspan="5">—</td>
											<td class="py-1.5 text-red-400">{r.reason}</td>
										{/if}
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				</section>

				<p class="text-xs text-slate-600">
					First-order sizing: TTFT is a single request's prefill; per-user throughput is the decode
					rate at the sized batch. Cluster = TP·PP model-parallel groups replicated (DP) until the
					provisioned concurrency is served. Estimates, not a benchmark.
				</p>
			</main>
		</div>
	</div>
</div>
