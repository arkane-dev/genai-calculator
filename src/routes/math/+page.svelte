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

<div class="page narrow">
	<header class="page-head">
		<div>
			<h1 class="page-title">Show the math</h1>
		<p class="lede">
			The calculation chain behind the current config — every formula symbolically, then with your
			numbers, then the result. Tied to the shared config, so it matches Modelling / Workload /
			Training exactly.
		</p>
		</div>
	</header>

	<HowTo>
		<p>
			Pick the view. <strong>Inference</strong> is the Modelling / Workload roofline (memory fit,
			step time, throughput, TTFT). <strong>Training</strong> is the fine-tune memory + step-time story.
			Change the config below (or on any other tab) and the math updates live. Results are sourced from
			the same functions the tool computes with — this tab can't drift from them.
		</p>
	</HowTo>

	<div class="layout">
		<aside
			class="sidebar"
		>
			<div class="panel controls tight">
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
					<div class="group">
						<div class="group-title">
							Training recipe
						</div>
						<div class="stack-m">
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

		<div class="results">
			{#each sections as section (section.title)}
				<section class="panel">
					<h2 class="section-title">
						{section.title}
					</h2>
					{#if section.intro}<p class="section-intro">{section.intro}</p>{/if}
					<div class="steps">
						{#each section.steps as step (step.label)}
							<div class="step">
								<div class="step-main">
									<div class="small dim">{step.label}</div>
									<!-- Scrollable, so it must take keyboard focus (WCAG 2.1.1). -->
									<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
									<div class="formula" tabindex="0" role="region" aria-label="{step.label}">
										<Latex display math={`${step.formula} \\;=\\; ${step.substituted}`} />
									</div>
									{#if step.note}<div class="step-note">{step.note}</div>{/if}
								</div>
								<div class="step-result num">
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

<style>
	.section-title {
		margin: 0;
		color: var(--nd-accent);
		font-size: var(--nd-text-sm);
		letter-spacing: 0.05em;
	}
	.section-intro { margin: var(--nd-space-1) 0 0; color: var(--nd-text-dim); font-size: var(--nd-text-xs); }
	.steps { display: flex; flex-direction: column; margin-top: var(--nd-space-3); }
	.step {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--nd-space-1);
		padding: var(--nd-space-3) 0;
	}
	.step + .step { border-top: 1px solid var(--nd-surface-2); }
	@media (min-width: 640px) {
		.step { grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: var(--nd-space-4); }
		.step-result { text-align: right; }
	}
	.step-main { min-width: 0; }
	.formula { margin-top: var(--nd-space-1); overflow-x: auto; color: var(--nd-text); }
	.formula:focus-visible { outline: 2px solid var(--nd-focus); outline-offset: 2px; }
	.step-note { margin-top: var(--nd-space-1); color: var(--nd-text-mute); font-size: 0.6875rem; }
	.step-result { flex: none; color: var(--nd-jade); font-size: var(--nd-text-sm); }
</style>
