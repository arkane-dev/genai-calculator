<script lang="ts">
	// Math tab: the calculation chains behind Modelling/Workload (inference) and Training,
	// rendered as LaTeX and tied to the shared cross-tab config. Finals come from the same
	// calc the other tabs use (via chains.ts), so what you see here is exactly what they compute.
	import { onMount } from 'svelte';
	import Controls from '$lib/components/Controls.svelte';
	import Segmented from '$lib/components/Segmented.svelte';
	import HowTo from '$lib/components/HowTo.svelte';
	import Latex from '$lib/components/Latex.svelte';
	import Slider from '$lib/components/Slider.svelte';
	import Toggle from '$lib/components/Toggle.svelte';
	import { MODELS, MODELS_BY_ID } from '$lib/profiler/data';
	import type { Config } from '$lib/profiler/types';
	import type { TrainConfig, TrainMethod, Optimizer } from '$lib/training/types';
	import { applyShared, readSharedState } from '$lib/share/url';
	import { loadShared, saveShared } from '$lib/state/shared';
	import { inferenceChain, trainingChain } from '$lib/math/chains';

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
		imagesPerRequest: 0,
		steps: 50,
		resolution: 1024,
		guidance: true,
		frames: 1
	});

	// Training-only knobs (not in the shared inference config). Model/GPU/parallelism come
	// from the shared config so it tracks the other tabs; these tune the training recipe.
	let method = $state<TrainMethod>('lora');
	let optimizer = $state<Optimizer>('adamw');
	let loraRank = $state(16);
	let zeroStage = $state<0 | 1 | 2 | 3>(1);
	let activationCheckpointing = $state(true);
	let seqLen = $state(4096);
	let gradAccum = $state(16);
	let datasetTokens = $state(1e9);
	let epochs = $state(3);

	let view = $state<'inference' | 'training'>('inference');
	let syncLoaded = $state(false);
	onMount(() => {
		applyShared(config, loadShared());
		applyShared(config, readSharedState(location.search));
		syncLoaded = true;
	});
	$effect(() => {
		if (syncLoaded) saveShared(config as unknown as Record<string, unknown>);
	});

	const isMoe = $derived(!!MODELS_BY_ID.get(config.modelId)?.moe);

	const trainCfg = $derived<TrainConfig>({
		modelId: config.modelId,
		gpuId: config.gpuId,
		numGpus: config.numGpus,
		gpusPerNode: config.gpusPerNode,
		fabricId: config.fabricId,
		method,
		optimizer,
		weightFormatId: config.weightFormatId,
		tp: config.tp,
		pp: config.pp,
		ppEnabled: config.ppEnabled,
		zeroStage,
		activationCheckpointing,
		cpuOffload: false,
		microBatchSize: 1,
		gradAccum,
		seqLen,
		loraRank,
		datasetTokens,
		epochs
	});

	const sections = $derived(
		view === 'inference' ? inferenceChain(config) : trainingChain(trainCfg)
	);
</script>

<svelte:head><title>GenAI Calculator — Math</title></svelte:head>

<div class="mx-auto max-w-6xl px-6 py-8">
	<header class="mb-4">
		<h1 class="text-2xl font-semibold text-slate-100">Show the math</h1>
		<p class="mt-1 text-sm text-slate-400">
			The calculation chain behind the current config — every formula symbolically, then with your
			numbers, then the result. Tied to the shared config, so it matches Modelling / Workload /
			Training exactly.
		</p>
	</header>

	<HowTo>
		<p>
			Pick the view. <strong>Inference</strong> is the Modelling / Workload roofline (memory fit,
			step time, throughput, TTFT). <strong>Training</strong> is the fine-tune memory + step-time story.
			Change the config below (or on any other tab) and the math updates live. Results are sourced from
			the same functions the tool computes with — this tab can't drift from them.
		</p>
	</HowTo>

	<div class="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-[320px_1fr]">
		<aside
			class="lg:sticky lg:top-6 lg:max-h-[calc(100vh-3rem)] lg:self-start lg:overflow-y-auto lg:pr-1"
		>
			<div class="flex flex-col gap-4 rounded-xl border border-slate-700 bg-slate-900/60 p-5">
				<Segmented
					label="View"
					bind:value={view}
					options={[
						{ value: 'inference', label: 'Inference (Modelling / Workload)' },
						{ value: 'training', label: 'Training' }
					]}
				/>
				<Controls bind:config {isMoe} models={MODELS} />
				{#if view === 'training'}
					<div class="border-t border-slate-700 pt-4">
						<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
							Training recipe
						</div>
						<div class="flex flex-col gap-3">
							<Segmented
								label="Method"
								bind:value={method}
								options={[
									{ value: 'full', label: 'Full FT' },
									{ value: 'lora', label: 'LoRA' },
									{ value: 'qlora', label: 'QLoRA' }
								]}
							/>
							<Segmented
								label="Optimizer"
								bind:value={optimizer}
								options={[
									{ value: 'adamw', label: 'AdamW' },
									{ value: 'adamw8bit', label: '8-bit Adam' },
									{ value: 'sgd', label: 'SGD' }
								]}
							/>
							<Segmented
								label="ZeRO stage"
								bind:value={zeroStage}
								options={[
									{ value: 0, label: '0' },
									{ value: 1, label: '1' },
									{ value: 2, label: '2' },
									{ value: 3, label: '3' }
								]}
							/>
							{#if method !== 'full'}
								<Slider
									label="LoRA rank"
									bind:value={loraRank}
									min={4}
									max={256}
									step={4}
									display={String(loraRank)}
								/>
							{/if}
							<Slider
								label="Sequence length"
								bind:value={seqLen}
								min={512}
								max={131072}
								step={512}
								log
								display="{seqLen} tok"
							/>
							<Slider
								label="Grad accumulation"
								bind:value={gradAccum}
								min={1}
								max={64}
								step={1}
								display={String(gradAccum)}
							/>
							<Toggle label="Activation checkpointing" bind:checked={activationCheckpointing} />
						</div>
					</div>
				{/if}
			</div>
		</aside>

		<div class="flex flex-col gap-6">
			{#each sections as section (section.title)}
				<section class="rounded-xl border border-slate-700 bg-slate-900/60 p-5">
					<h2 class="text-sm font-semibold tracking-wide text-teal-300 uppercase">
						{section.title}
					</h2>
					{#if section.intro}<p class="mt-1 text-xs text-slate-400">{section.intro}</p>{/if}
					<div class="mt-3 flex flex-col divide-y divide-slate-800">
						{#each section.steps as step (step.label)}
							<div
								class="grid grid-cols-1 gap-1 py-3 sm:grid-cols-[1fr_auto] sm:items-center sm:gap-4"
							>
								<div class="min-w-0">
									<div class="text-xs text-slate-400">{step.label}</div>
									<div class="mt-1 overflow-x-auto text-slate-100">
										<Latex display math={`${step.formula} \\;=\\; ${step.substituted}`} />
									</div>
									{#if step.note}<div class="mt-1 text-[11px] text-slate-500">{step.note}</div>{/if}
								</div>
								<div class="shrink-0 font-mono text-sm text-emerald-300 sm:text-right">
									{step.result}
								</div>
							</div>
						{/each}
					</div>
				</section>
			{/each}
		</div>
	</div>
</div>
