<script lang="ts">
	import CostPanel from '$lib/components/CostPanel.svelte';
	import EnergyPanel from '$lib/components/EnergyPanel.svelte';
	import LoadPanel from '$lib/components/LoadPanel.svelte';
	import HowTo from '$lib/components/HowTo.svelte';
	import InfoTip from '$lib/components/InfoTip.svelte';
	import MemoryStack from '$lib/components/MemoryStack.svelte';
	import Segmented from '$lib/components/Segmented.svelte';
	import Slider from '$lib/components/Slider.svelte';
	import StatCard from '$lib/components/StatCard.svelte';
	import Toggle from '$lib/components/Toggle.svelte';
	import { fmtBytes, fmtMs, fmtTps } from '$lib/profiler/calc';
	import {
		COLORS,
		defaultFabricFor,
		FABRICS,
		FABRICS_BY_ID,
		GPUS,
		GPUS_BY_ID,
		gpuSupportsFormat,
		MODELS,
		weightFormatOptions,
		WEIGHT_FORMATS_BY_ID
	} from '$lib/profiler/data';
	import { computeTraining } from '$lib/training/calc';
	import { configWarnings } from '$lib/profiler/sanity';
	import SanityNotes from '$lib/components/SanityNotes.svelte';
	import type { TrainConfig } from '$lib/training/types';
	import ShareButton from '$lib/components/ShareButton.svelte';
	import HandoffMenu from '$lib/components/HandoffMenu.svelte';
	import ExportButton from '$lib/components/ExportButton.svelte';
	import type { SizingReport } from '$lib/report/markdown';
	import { defaultPurchasing, estimateCost, PRICE_AS_OF } from '$lib/cost/pricing';
	import { DEFAULT_GRID, type EmissionsMethod } from '$lib/energy/estimate';
	import { applyShared, readSharedState } from '$lib/share/url';
	import { loadShared, saveShared } from '$lib/state/shared';
	import { base } from '$app/paths';
	import { onMount } from 'svelte';

	let cfg = $state<TrainConfig>({
		modelId: 'llama31-70b',
		gpuId: 'h100-sxm',
		numGpus: 8,
		gpusPerNode: 8,
		fabricId: 'mp-400',
		method: 'lora',
		optimizer: 'adamw',
		weightFormatId: 'bf16',
		tp: 8,
		pp: 1,
		ppEnabled: false,
		zeroStage: 1,
		activationCheckpointing: true,
		cpuOffload: false,
		microBatchSize: 1,
		gradAccum: 16,
		seqLen: 4096,
		loraRank: 16,
		datasetTokens: 1e9,
		epochs: 3
	});

	// transformers only for v1 (LLMs + visual-AR); diffusion/JEPA training is backlog
	const trainable = MODELS.filter((m) => (m.kind ?? 'transformer') === 'transformer');
	const groups = [
		{
			label: 'Language models',
			items: trainable.filter((m) => !m.visualAR && !m.reasoning && !m.tts)
		},
		{ label: 'Reasoning models', items: trainable.filter((m) => m.reasoning) },
		{ label: 'Speech synthesis (TTS)', items: trainable.filter((m) => m.tts) },
		{ label: 'Autoregressive image', items: trainable.filter((m) => m.visualAR) }
	].filter((gp) => gp.items.length);

	const GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
	const powersUpTo = (n: number) =>
		[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048].filter((p) => p <= n);

	const isLora = $derived(cfg.method === 'lora' || cfg.method === 'qlora');
	const p = $derived(computeTraining(cfg));
	const gpu = $derived(GPUS_BY_ID.get(cfg.gpuId)!);
	const wfOpts = $derived(weightFormatOptions(gpu, ['bf16', 'fp8']));
	$effect(() => {
		const f = WEIGHT_FORMATS_BY_ID.get(cfg.weightFormatId);
		if (gpu && f && !gpuSupportsFormat(gpu, f)) cfg.weightFormatId = 'bf16';
	});
	const sanity = $derived(
		configWarnings({
			tp: cfg.tp,
			gpusPerNode: cfg.gpusPerNode,
			fabricLabel: FABRICS_BY_ID.get(cfg.fabricId)?.label ?? cfg.fabricId,
			gpu,
			formatTier: WEIGHT_FORMATS_BY_ID.get(cfg.weightFormatId)?.tier ?? 'fp16'
		})
	);

	// shared region: one selector (in EnergyPanel) drives both energy CO2 and cost context
	let region = $state(DEFAULT_GRID);
	let emissionsMethod = $state<EmissionsMethod>('location');

	let syncLoaded = $state(false);
	onMount(() => {
		applyShared(cfg, loadShared());
		applyShared(cfg, readSharedState(location.search));
		syncLoaded = true;
	});
	$effect(() => {
		if (syncLoaded) saveShared(cfg as unknown as Record<string, unknown>);
	});

	// Cluster described as instances (nodes) × GPUs-per-node; total = product. Two-way
	// sync: an external numGpus change (default or shared ?c= link) re-derives nodes; a UI
	// change to nodes/gpusPerNode writes numGpus back.
	let nodes = $state(Math.max(1, Math.round(cfg.numGpus / cfg.gpusPerNode)));
	let lastNumGpus = cfg.numGpus;
	$effect(() => {
		if (cfg.numGpus !== lastNumGpus) {
			nodes = Math.max(1, Math.round(cfg.numGpus / cfg.gpusPerNode));
			lastNumGpus = cfg.numGpus;
		}
	});
	$effect(() => {
		const want = nodes * cfg.gpusPerNode;
		if (cfg.numGpus !== want) {
			cfg.numGpus = want;
			lastNumGpus = want;
		}
	});

	// keep parallelism within the GPU budget
	$effect(() => {
		if (cfg.tp > cfg.numGpus) cfg.tp = cfg.numGpus;
		const maxPp = Math.max(1, Math.floor(cfg.numGpus / cfg.tp));
		if (cfg.pp > maxPp) cfg.pp = maxPp;
	});

	const fmtSeq = (n: number) =>
		n >= 1048576
			? `${(n / 1048576).toFixed(n % 1048576 ? 1 : 0)}M`
			: n >= 1024
				? `${(n / 1024).toFixed(0)}k`
				: String(n);
	const fmtTokens = (n: number) =>
		n >= 1e12
			? `${(n / 1e12).toFixed(1)}T`
			: n >= 1e9
				? `${(n / 1e9).toFixed(0)}B`
				: `${(n / 1e6).toFixed(0)}M`;
	const fmtHours = (h: number) =>
		h < 1
			? `${(h * 60).toFixed(0)} min`
			: h < 48
				? `${h.toFixed(1)} h`
				: `${(h / 24).toFixed(1)} days`;
	const fmtParams = (n: number) =>
		n >= 1e9 ? `${(n / 1e9).toFixed(n < 1e10 ? 1 : 0)}B` : `${(n / 1e6).toFixed(0)}M`;

	const bottleneckColor = { compute: COLORS.tensor, network: COLORS.network };

	const fmtUsd = (n: number) =>
		n >= 100 ? `$${n.toLocaleString('en-US', { maximumFractionDigits: 0 })}` : `$${n.toFixed(2)}`;

	// downloadable Markdown summary of the training run
	function buildReport(): SizingReport {
		const model = MODELS.find((m) => m.id === cfg.modelId);
		const wf = WEIGHT_FORMATS_BY_ID.get(cfg.weightFormatId)?.label ?? cfg.weightFormatId;
		const sections: SizingReport['sections'] = [
			{
				heading: 'Configuration',
				rows: [
					['Model', `${model?.name ?? cfg.modelId} (${fmtParams(p.baseParams)} params)`],
					['Method', cfg.method.toUpperCase() + (isLora ? ` (rank ${cfg.loraRank})` : '')],
					['Optimizer', cfg.optimizer],
					['GPUs', `${cfg.numGpus}× ${gpu.name}`],
					['Parallelism', `TP${cfg.tp} · PP${cfg.pp} · ZeRO-${cfg.zeroStage}`],
					['Activation checkpointing', cfg.activationCheckpointing ? 'on' : 'off'],
					['CPU offload', cfg.cpuOffload ? 'on' : 'off'],
					['Weight format', wf],
					['Sequence length', fmtSeq(cfg.seqLen)],
					['Batch', `micro ${cfg.microBatchSize} × grad-accum ${cfg.gradAccum}`],
					[
						'Dataset',
						`${fmtTokens(cfg.datasetTokens)} tok × ${cfg.epochs} epoch${cfg.epochs > 1 ? 's' : ''}`
					]
				]
			},
			{
				heading: 'Results (first-order)',
				rows: [
					['Fits per GPU', p.perGpu.fits ? 'Yes' : 'No — does not fit'],
					['Bottleneck', `${p.bottleneck}-bound`],
					['Trainable params', fmtParams(p.trainableParams)],
					['Throughput', `${fmtTps(p.tokensPerSec)} tok/s`],
					['Time to train', fmtHours(p.timeToTrainHours)],
					['Memory', `${fmtBytes(p.perGpu.used)} / ${fmtBytes(p.perGpu.capacity)} per GPU`]
				]
			}
		];
		const cph = estimateCost(cfg.gpuId, cfg.numGpus, defaultPurchasing(cfg.gpuId)).clusterPerHour;
		if (cph != null)
			sections.push({
				heading: 'Cost (market estimate)',
				rows: [
					['Cluster / hour', fmtUsd(cph)],
					[
						'Total for the run',
						`${fmtUsd(cph * p.timeToTrainHours)} (${fmtHours(p.timeToTrainHours)})`
					]
				],
				note: `GPU cost: median on-demand market rate (${PRICE_AS_OF}).`
			});
		return {
			title: `Training sizing — ${model?.name ?? cfg.modelId}`,
			subtitle: `${cfg.method.toUpperCase()} · ${cfg.numGpus}× ${gpu.name}`,
			generatedAt: new Date().toISOString().slice(0, 10),
			sections,
			disclaimer:
				'First-order training estimates, not a benchmark. GPU prices are a point-in-time market median. Get a real quote before you decide.'
		};
	}

	const instanceOpts = GPU_COUNTS.map((n) => ({ value: n, label: String(n) }));
	const nodeSizeOpts = [1, 2, 4, 8].map((n) => ({ value: n, label: String(n) }));
	const parallelOpts = $derived(
		powersUpTo(cfg.numGpus).map((n) => ({ value: n, label: String(n) }))
	);
</script>

<svelte:head><title>GenAI Calculator — Training</title></svelte:head>

<div class="min-h-screen bg-slate-950 text-slate-100">
	<div class="mx-auto max-w-7xl px-6 py-8">
		<header class="mb-6 flex items-start justify-between gap-4">
			<div>
				<h1 class="text-2xl font-semibold">Training / Fine-tuning Calculator</h1>
				<p class="mt-1 text-sm text-slate-400">
					Will it fit to train, and how long will it take? First-order estimates for full
					fine-tuning, LoRA and QLoRA.
				</p>
			</div>
			<div class="flex items-center gap-2">
				<ExportButton report={buildReport} filename="gpu-sizing-training" />
				<ShareButton payload={cfg} />
				<HandoffMenu
					payload={cfg}
					targets={[
						{ path: 'modelling', label: 'Modelling' },
						{ path: '', label: 'Workload' },
						{ path: 'economics', label: 'Self-host vs API' }
					]}
				/>
			</div>
		</header>

		<HowTo>
			<p>
				This tab sizes <strong>training and fine-tuning</strong>, not serving. The big difference
				from the other tabs is memory: to <em>train</em> a model you must hold not just its weights,
				but also its <strong>gradients</strong>, the <strong>optimizer's</strong> running state, and
				the <strong>activations</strong> from the forward pass. With normal AdamW that is about
				<strong>16 bytes for every parameter</strong>
				— so a model that <em>runs</em> on one GPU often needs six to eight times the memory to
				<em>train</em>.
			</p>

			<p class="mt-3 font-medium text-slate-300">Full vs LoRA vs QLoRA</p>
			<ul class="mt-1 ml-4 list-disc space-y-1">
				<li>
					<strong>Full fine-tune</strong> — trains every weight. Most accurate, most memory: gradients
					and optimizer state for the whole model.
				</li>
				<li>
					<strong>LoRA</strong> — freezes the model and trains tiny added "adapter" weights. Only the
					adapters need gradients and optimizer state, so the memory drops enormously.
				</li>
				<li>
					<strong>QLoRA</strong> — LoRA on top of a model squeezed to 4-bit. This is how a 70B model fine-tunes
					on a single 80 GB GPU.
				</li>
			</ul>

			<p class="mt-3 font-medium text-slate-300">The controls</p>
			<ul class="mt-1 ml-4 list-disc space-y-1">
				<li>
					<strong>ZeRO / sharding</strong> — split the model states across your data-parallel GPUs so
					each holds a slice. Higher stages shard more (optimizer → +gradients → +weights).
				</li>
				<li>
					<strong>Activation checkpointing</strong> — throw away activations and recompute them in the
					backward pass. Saves a lot of memory for ~30% more compute.
				</li>
				<li>
					<strong>Micro-batch × grad-accum</strong> — the effective batch is micro-batch × grad-accum
					× replicas. Grad-accum lets a small micro-batch reach a large effective batch.
				</li>
			</ul>

			<p class="mt-3">
				Numbers are first-order estimates, not a benchmark. v1 covers transformer language and
				visual-AR models; diffusion and JEPA training will follow. To size <em>serving</em> instead,
				use the <a href="{base}/modelling">Modelling</a> or <a href="{base}/">Workload</a> tabs.
			</p>
		</HowTo>

		<div class="grid grid-cols-1 gap-6 lg:grid-cols-[340px_1fr]">
			<!-- controls -->
			<aside
				class="lg:sticky lg:top-6 lg:max-h-[calc(100vh-3rem)] lg:self-start lg:overflow-y-auto lg:pr-1"
			>
				<div class="flex flex-col gap-5 rounded-xl border border-slate-700 bg-slate-900/60 p-5">
					<div>
						<div class="mb-1 flex items-center gap-1.5 text-sm text-slate-300">
							Model
							<InfoTip
								text="The model you want to train or fine-tune. Bigger models need far more memory to train than to run."
							/>
						</div>
						<select
							bind:value={cfg.modelId}
							class="w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100"
						>
							{#each groups as grp (grp.label)}
								<optgroup label={grp.label}>
									{#each grp.items as m (m.id)}<option value={m.id}
											>{m.name}{m.projected ? ' (projected)' : ''}</option
										>{/each}
								</optgroup>
							{/each}
						</select>
					</div>

					<Segmented
						label="Method"
						bind:value={cfg.method}
						options={[
							{ value: 'full', label: 'Full' },
							{ value: 'lora', label: 'LoRA' },
							{ value: 'qlora', label: 'QLoRA' }
						]}
						info="Full trains every weight (most memory). LoRA freezes the model and trains small adapters. QLoRA does LoRA on a 4-bit base — the least memory."
					/>

					{#if isLora}
						<Slider
							label="LoRA rank"
							bind:value={cfg.loraRank}
							min={4}
							max={256}
							step={4}
							info="How big the adapter matrices are. Higher rank can learn more but adds trainable parameters (and a little memory). 8–64 is typical."
						/>
					{/if}

					<div>
						<div class="mb-1 flex items-center gap-1.5 text-sm text-slate-300">
							GPU
							<InfoTip
								text="The chip you'd train on. Its memory decides whether the model fits; its math speed decides how fast each step runs."
							/>
						</div>
						<select
							bind:value={cfg.gpuId}
							onchange={() => (cfg.fabricId = defaultFabricFor(cfg.gpuId))}
							class="w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100"
						>
							{#each GPUS as g (g.id)}<option value={g.id}>{g.name}</option>{/each}
						</select>
					</div>

					<Segmented
						label="GPUs per node"
						bind:value={cfg.gpusPerNode}
						options={nodeSizeOpts}
						info="GPUs inside one server (instance). Same-server GPUs share over fast NVLink; across servers it's the slower network — which matters for gradient sync. Most GPU instances are 8× per node."
					/>
					<Segmented
						label="Instances (nodes)"
						bind:value={nodes}
						options={instanceOpts}
						info="How many servers (instances). Total GPUs = instances × GPUs-per-node. More GPUs train faster (data-parallel) and, with sharding, let bigger models fit."
					/>
					<p class="-mt-2 text-[10px] text-slate-500">
						= <span class="font-mono text-slate-400"
							>{cfg.numGpus} GPU{cfg.numGpus === 1 ? '' : 's'}</span
						>
						total ({nodes} × {cfg.gpusPerNode})
					</p>
					<div>
						<div class="mb-1 flex items-center gap-1.5 text-sm text-slate-300">
							Network fabric
							<InfoTip
								text="The network between servers. Gradient sync runs over it every step when training spans more than one server, so a faster fabric means less waiting."
							/>
						</div>
						<select
							bind:value={cfg.fabricId}
							class="w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100"
						>
							{#each FABRICS as f (f.id)}<option value={f.id}>{f.label}</option>{/each}
						</select>
					</div>

					<div class="border-t border-slate-700 pt-4">
						<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
							Memory strategy
						</div>
						<Segmented
							label="ZeRO / sharding stage"
							bind:value={cfg.zeroStage}
							options={[
								{ value: 0, label: 'Off' },
								{ value: 1, label: '1' },
								{ value: 2, label: '2' },
								{ value: 3, label: '3' }
							]}
							info="Splits the model states across your data-parallel GPUs. Stage 1 shards the optimizer, stage 2 also the gradients, stage 3 also the weights (like FSDP). Higher = less memory per GPU, a little more communication."
						/>
						<div class="mt-3"></div>
						<Toggle
							label="Activation checkpointing"
							bind:checked={cfg.activationCheckpointing}
							hint="recompute in backward"
							info="Throw activations away after the forward pass and recompute them during backward. Saves a lot of memory for about 30% more compute. Usually worth it."
						/>
						<div class="mt-3"></div>
						<Toggle
							label="CPU offload (ZeRO-Offload)"
							bind:checked={cfg.cpuOffload}
							hint="optimizer state in host RAM"
							info="Puts optimizer state (and updates) in host RAM instead of HBM. Slashes GPU memory — how big models fit on few GPUs — but adds a PCIe shuttle per step. This is where Gen5 PCIe (2× Gen4) pays off during training."
						/>
					</div>

					<div class="border-t border-slate-700 pt-4">
						<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
							Parallelism
						</div>
						<Segmented
							label="Tensor parallel (TP)"
							bind:value={cfg.tp}
							options={parallelOpts}
							info="Split each layer's math across GPUs inside a node. Helps a model that's too big for one GPU's memory, at the cost of constant chatter between them."
						/>
						<div class="mt-3"></div>
						<div class="flex flex-col gap-2 rounded-lg border border-slate-700 bg-slate-800/40 p-3">
							<Toggle
								label="Pipeline parallel (PP)"
								bind:checked={cfg.ppEnabled}
								hint="split layers into stages"
								info="Put different layers on different GPUs, like an assembly line. Helps fit very large models across nodes."
							/>
							{#if cfg.ppEnabled}
								<Segmented
									label="PP degree"
									bind:value={cfg.pp}
									options={parallelOpts}
									info="How many assembly-line stages. More stages fit a bigger model but add hand-offs."
								/>
							{/if}
						</div>
					</div>

					<div class="border-t border-slate-700 pt-4">
						<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
							Optimizer & batch
						</div>
						<Segmented
							label="Optimizer"
							bind:value={cfg.optimizer}
							options={[
								{ value: 'adamw', label: 'AdamW' },
								{ value: 'adamw8bit', label: '8-bit Adam' },
								{ value: 'sgd', label: 'SGD' }
							]}
							info="AdamW keeps two running averages per weight in fp32 (12 bytes/param with the master copy). 8-bit Adam stores them in 8-bit (6 bytes). SGD with momentum is 8 bytes."
						/>
						{#if cfg.method !== 'qlora'}
							<div class="mt-3"></div>
							<Segmented
								label="Base weight format"
								bind:value={cfg.weightFormatId}
								options={wfOpts}
								info="How the base weights are stored. QLoRA forces 4-bit; here BF16 is standard for training and FP8 halves the weight memory (greyed out on GPUs without native FP8)."
							/>
						{/if}
						<div class="mt-3"></div>
						<Slider
							label="Micro-batch size"
							bind:value={cfg.microBatchSize}
							min={1}
							max={64}
							step={1}
							info="Sequences processed together per step, per GPU replica. Bigger uses the GPU more fully but needs more activation memory."
						/>
						<div class="mt-3"></div>
						<Slider
							label="Grad-accum steps"
							bind:value={cfg.gradAccum}
							min={1}
							max={64}
							step={1}
							info="Micro-batches summed before an optimizer step. Lets a small micro-batch reach a big effective batch without more memory."
						/>
						<div class="mt-3"></div>
						<Slider
							label="Sequence length"
							bind:value={cfg.seqLen}
							min={512}
							max={1048576}
							log
							display="{fmtSeq(cfg.seqLen)} tok"
							info="Training context length. Activation memory and attention compute grow with it (attention with its square). Long-context fine-tuning reaches 128k–1M, where activation memory dominates."
						/>
					</div>

					<div class="border-t border-slate-700 pt-4">
						<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
							Dataset (time-to-train)
						</div>
						<Slider
							label="Dataset tokens"
							bind:value={cfg.datasetTokens}
							min={1e7}
							max={1e13}
							log
							display={fmtTokens(cfg.datasetTokens)}
							info="How many tokens you'll train over in one pass. Fine-tuning is often 10M–10B; pretraining is trillions."
						/>
						<div class="mt-3"></div>
						<Slider
							label="Epochs"
							bind:value={cfg.epochs}
							min={1}
							max={10}
							step={1}
							info="How many passes over the dataset. Time-to-train scales with dataset × epochs."
						/>
					</div>
				</div>
			</aside>

			<!-- results -->
			<main class="flex flex-col gap-6">
				<SanityNotes notes={sanity} />
				<div class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
					<StatCard
						label="Fits to train"
						value={p.perGpu.fits ? 'Yes' : 'No'}
						accent={p.perGpu.fits ? '#34d399' : '#f87171'}
						sub={p.perGpu.fits
							? `${fmtBytes(p.perGpu.headroom)} free / GPU`
							: `${fmtBytes(-p.perGpu.headroom)} over / GPU`}
						info="Do the weights, gradients, optimizer state, and activations all fit in each GPU's memory? If No, raise TP/PP, use a higher ZeRO stage, turn on checkpointing, or switch to LoRA/QLoRA."
					/>
					<StatCard
						label="Trainable params"
						value={fmtParams(p.trainableParams)}
						accent={COLORS.gradients}
						sub="{((p.trainableParams / p.baseParams) * 100).toFixed(
							p.trainableParams / p.baseParams < 0.01 ? 2 : 0
						)}% of {fmtParams(p.baseParams)}"
						info="How many parameters actually get trained. Full fine-tuning trains all of them; LoRA/QLoRA train only the small adapters, which is why they need so much less memory."
					/>
					<StatCard
						label="Training throughput"
						value={fmtTps(p.tokensPerSec)}
						unit="tok/s"
						accent={COLORS.memPipe}
						sub="{p.dp} replicas · {p.replicaGpus} GPU each"
						info="Tokens per second the whole cluster trains on, across all GPUs. Higher finishes sooner. More data-parallel replicas raise it until gradient sync gets in the way."
					/>
					<StatCard
						label="Time to train"
						value={fmtHours(p.timeToTrainHours)}
						accent={bottleneckColor[p.bottleneck]}
						sub="{fmtTokens(cfg.datasetTokens)} × {cfg.epochs} ep · {p.bottleneck}-bound"
						info="Wall-clock to make the chosen number of passes over the dataset at this throughput. A rough planning figure, not a guarantee."
					/>
					<StatCard
						label="Step time"
						value={fmtMs(p.stepTimeMs)}
						sub="{cfg.gradAccum} micro-steps + sync"
						info="Time for one optimizer step: all the grad-accumulation micro-batches plus the gradient sync across data-parallel GPUs."
					/>
					<StatCard
						label="MFU"
						value={(p.mfu * 100).toFixed(0)}
						unit="%"
						accent={p.mfu > 0.35 ? '#34d399' : '#fbbf24'}
						sub="{p.computeAchievedTflops.toFixed(0)} TFLOP/s / GPU"
						info="Model-flop utilization: how much of the GPU's math peak the training actually uses. Real large-scale training lands around 35–50%; lower usually means it's waiting on the network."
					/>
				</div>

				<section class="rounded-xl border border-slate-700 bg-slate-900/60 p-5">
					<div class="mb-3 flex items-center gap-1.5 text-sm font-medium text-slate-200">
						Per-GPU memory to train
						<InfoTip
							text="Everything one GPU must hold to train: the (possibly frozen) weights, the gradients, the optimizer's running state, the activations from the forward pass, and framework overhead. Sharding (ZeRO) and checkpointing shrink these."
						/>
					</div>
					<div class="grid grid-cols-1 items-stretch gap-4 md:grid-cols-[1fr_1fr]">
						<div class="min-h-[24rem]">
							<MemoryStack
								segments={p.segments}
								capacity={p.perGpu.capacity}
								used={p.perGpu.used}
								fits={p.perGpu.fits}
							/>
						</div>
						<div class="flex flex-col justify-center gap-2 text-sm">
							<div class="flex items-center justify-between border-b border-slate-800 py-1.5">
								<span class="flex items-center gap-1.5 text-slate-400"
									><span class="inline-block h-3 w-3 rounded-sm" style:background={COLORS.weights}
									></span>Weights{isLora ? ' (frozen)' : ''}</span
								>
								<span class="font-mono text-slate-200">{fmtBytes(p.perGpu.weights)}</span>
							</div>
							<div class="flex items-center justify-between border-b border-slate-800 py-1.5">
								<span class="flex items-center gap-1.5 text-slate-400"
									><span class="inline-block h-3 w-3 rounded-sm" style:background={COLORS.gradients}
									></span>Gradients</span
								>
								<span class="font-mono text-slate-200">{fmtBytes(p.perGpu.gradients)}</span>
							</div>
							<div class="flex items-center justify-between border-b border-slate-800 py-1.5">
								<span class="flex items-center gap-1.5 text-slate-400"
									><span class="inline-block h-3 w-3 rounded-sm" style:background={COLORS.optimizer}
									></span>Optimizer states</span
								>
								<span class="font-mono text-slate-200">{fmtBytes(p.perGpu.optimizer)}</span>
							</div>
							<div class="flex items-center justify-between border-b border-slate-800 py-1.5">
								<span class="flex items-center gap-1.5 text-slate-400"
									><span
										class="inline-block h-3 w-3 rounded-sm"
										style:background={COLORS.activations}
									></span>Activations</span
								>
								<span class="font-mono text-slate-200">{fmtBytes(p.perGpu.activations)}</span>
							</div>
							<div class="flex items-center justify-between py-1.5">
								<span class="flex items-center gap-1.5 text-slate-400"
									><span class="inline-block h-3 w-3 rounded-sm" style:background={COLORS.overhead}
									></span>Overhead</span
								>
								<span class="font-mono text-slate-200">{fmtBytes(p.perGpu.overhead)}</span>
							</div>
							<div
								class="mt-1 flex items-center justify-between border-t border-slate-700 pt-2 font-medium"
							>
								<span class="text-slate-300">Total / GPU</span>
								<span class="font-mono {p.perGpu.fits ? 'text-slate-100' : 'text-red-400'}"
									>{fmtBytes(p.perGpu.used)} / {fmtBytes(p.perGpu.capacity)}</span
								>
							</div>
							<p class="mt-1 text-xs text-slate-500">
								Global batch <span class="font-mono text-slate-400"
									>{fmtTokens(p.globalBatchTokens)} tok</span
								>
								= {cfg.microBatchSize} × {cfg.gradAccum} accum × {p.dp} replicas × {fmtSeq(
									cfg.seqLen
								)}.
							</p>
						</div>
					</div>
				</section>

				<CostPanel
					gpuId={cfg.gpuId}
					numGpus={cfg.numGpus}
					metric={{ kind: 'run', hours: p.timeToTrainHours }}
				/>

				<EnergyPanel
					gpuId={cfg.gpuId}
					numGpus={cfg.numGpus}
					util={0.85}
					metric={{ kind: 'run', hours: p.timeToTrainHours }}
					bind:region
					bind:method={emissionsMethod}
				/>

				<LoadPanel
					gpuId={cfg.gpuId}
					numGpus={cfg.numGpus}
					weightBytes={(p.baseParams *
						(WEIGHT_FORMATS_BY_ID.get(cfg.weightFormatId)?.bitsPerWeight ?? 16)) /
						8}
				/>

				{#if p.notes.length}
					<ul class="rounded-xl border border-slate-800 bg-slate-900/40 p-4 text-xs text-slate-400">
						{#each p.notes as n (n)}<li class="flex gap-2">
								<span class="text-slate-600">•</span>{n}
							</li>{/each}
					</ul>
				{/if}

				<p class="text-xs text-slate-600">
					First-order training roofline (MFU {(0.45).toFixed(2)}) with {fmtBytes(2 * 1024 ** 3)} overhead/GPU.
					Model states use the mixed-precision AdamW convention (weight 2 + grad 2 + optimizer 12 B/param).
					Activation memory follows the Korthikanti et al. per-layer estimate. Pipeline bubble and exact
					comm overlap are approximate; estimates, not a benchmark.
				</p>
			</main>
		</div>
	</div>
</div>
