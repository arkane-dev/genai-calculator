<script lang="ts">
	// Self-host vs API, built on the SAME Modelling setup: you configure the exact cluster
	// with the full Modelling control panel, the forward model (computeProfile) gives its
	// throughput, we cost that cluster and compare to a per-token API. Because it uses the
	// Modelling `Config`, it syncs cleanly with Modelling/Workload via the handoff.
	import BreakevenChart from '$lib/components/BreakevenChart.svelte';
	import Controls from '$lib/components/Controls.svelte';
	import HowTo from '$lib/components/HowTo.svelte';
	import InfoTip from '$lib/components/InfoTip.svelte';
	import Segmented from '$lib/components/Segmented.svelte';
	import Slider from '$lib/components/Slider.svelte';
	import StatCard from '$lib/components/StatCard.svelte';
	import ShareButton from '$lib/components/ShareButton.svelte';
	import HandoffMenu from '$lib/components/HandoffMenu.svelte';
	import SanityNotes from '$lib/components/SanityNotes.svelte';
	import { computeProfile } from '$lib/profiler/calc';
	import {
		FABRICS_BY_ID,
		GPUS_BY_ID,
		MODELS,
		MODELS_BY_ID,
		WEIGHT_FORMATS_BY_ID
	} from '$lib/profiler/data';
	import type { Config } from '$lib/profiler/types';
	import { configWarnings } from '$lib/profiler/sanity';
	import { defaultPurchasing, estimateCost } from '$lib/cost/pricing';
	import { breakeven } from '$lib/economics/breakeven';
	import {
		apiPriceFor,
		priceSourceFor,
		apiTierPrice,
		API_TIERS,
		DEFAULT_API_PERF,
		PRICE_SNAPSHOT,
		type ApiTier
	} from '$lib/economics/apiPricing';
	import { applyShared, readSharedState } from '$lib/share/url';
	import { loadShared, saveShared, readPricing, type PricingState } from '$lib/state/shared';
	import Toggle from '$lib/components/Toggle.svelte';
	import { onMount, untrack } from 'svelte';

	// Only token-priced text models make sense against a per-token API.
	const textModels = MODELS.filter(
		(m) => (m.kind ?? 'transformer') === 'transformer' && !m.visualAR
	);

	// Full Modelling config (same shape → syncs via the handoff). Realistic mid-size default.
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

	// Break-even knobs (kept out of Config so the shared link / handoff stays clean).
	// Pricing (purchasing model + a custom $/GPU-hour) is shared with Modelling/Workload's
	// cost panel, so a manual rate set there flows here and vice-versa.
	let pricing = $state<PricingState>({
		purchasing: defaultPurchasing('h100-sxm'),
		useEstimatedPricing: true,
		manualPerGpuHour: 0
	});
	let manualSeeded = $state(false);
	let dutyPct = $state(40);
	let apiInPerM = $state(untrack(() => apiPriceFor(config.modelId).inPerM));
	let apiOutPerM = $state(untrack(() => apiPriceFor(config.modelId).outPerM));
	// API latency has no reliable per-model public figure; start from a typical value.
	let apiPerUserTps = $state(DEFAULT_API_PERF.tps);
	let apiTtftMs = $state(DEFAULT_API_PERF.ttftMs);
	// Per-user SLO to sanity-check the API against (Config has no SLO fields).
	let slaTps = $state(30);
	let slaTtftMs = $state(1500);

	let syncLoaded = $state(false);
	onMount(() => {
		applyShared(config, loadShared());
		applyShared(config, readSharedState(location.search));
		readPricing(loadShared(), pricing);
		readPricing(readSharedState(location.search), pricing);
		// This tab is token-priced only; if a non-text model synced in, fall back.
		if (!textModels.some((m) => m.id === config.modelId)) config.modelId = 'llama31-70b';
		syncLoaded = true;
	});
	$effect(() => {
		if (syncLoaded) saveShared({ ...(config as unknown as Record<string, unknown>), ...pricing });
	});

	const model = $derived(MODELS_BY_ID.get(config.modelId));
	const isMoe = $derived(!!model?.moe);
	const gpu = $derived(GPUS_BY_ID.get(config.gpuId)!);
	const p = $derived(computeProfile(config));

	// API pricing tier (standard / batch) + prompt caching. Any API model can use both.
	// The tier reseeds the API price; caching discounts the cached-prefix input.
	let apiTier = $state<ApiTier>('standard');
	let cacheOn = $state(false);

	function seedApiPrice() {
		const pr = apiTierPrice(config.modelId, apiTier, {
			cachedFrac: cacheOn ? config.cachedPrefixFrac : 0
		});
		apiInPerM = pr.inPerM;
		apiOutPerM = pr.outPerM;
	}

	// Reseed the API price when the model or tier/caching changes, and purchasing when the
	// GPU changes — only on an actual change, so manual edits survive re-renders.
	let lastModel = untrack(() => config.modelId);
	let lastGpu = untrack(() => config.gpuId);
	let lastTierKey = untrack(() => `${apiTier}|${cacheOn}|${config.cachedPrefixFrac}`);
	$effect(() => {
		const key = `${apiTier}|${cacheOn}|${cacheOn ? config.cachedPrefixFrac : 0}`;
		if (config.modelId !== lastModel || key !== lastTierKey) {
			seedApiPrice();
			lastModel = config.modelId;
			lastTierKey = key;
		}
		if (config.gpuId !== lastGpu) {
			pricing.purchasing = defaultPurchasing(config.gpuId);
			lastGpu = config.gpuId;
		}
	});

	const cost = $derived(estimateCost(config.gpuId, config.numGpus, pricing.purchasing));
	// per-GPU estimated rate, used to seed the manual input the first time it's shown
	const estPerGpu = $derived(
		cost.clusterPerHour != null && config.numGpus > 0 ? cost.clusterPerHour / config.numGpus : null
	);
	$effect(() => {
		if (!pricing.useEstimatedPricing && !manualSeeded && !(pricing.manualPerGpuHour > 0)) {
			pricing.manualPerGpuHour = estPerGpu != null ? Math.round(estPerGpu * 100) / 100 : 4;
			manualSeeded = true;
		}
	});
	// cluster $/hr: the market estimate, or the user's per-GPU rate × GPU count (shared w/ other tabs)
	const clusterPerHour = $derived(
		pricing.useEstimatedPricing
			? cost.clusterPerHour
			: pricing.manualPerGpuHour > 0
				? pricing.manualPerGpuHour * config.numGpus
				: null
	);

	const be = $derived(
		p.perGpu.fits && clusterPerHour != null
			? breakeven({
					clusterPerHourUsd: clusterPerHour,
					clusterTps: p.throughputTps,
					inputTokens: config.inputTokens,
					outputTokens: config.outputTokens,
					apiInPerM,
					apiOutPerM,
					dutyCycle: dutyPct / 100
				})
			: null
	);

	// API-can't-serve check (against the per-user SLOs above).
	const apiSloReasons = $derived([
		...(apiPerUserTps < slaTps
			? [`${apiPerUserTps} tok/s below the ${slaTps} tok/s min-throughput`]
			: []),
		...(apiTtftMs > slaTtftMs ? [`${apiTtftMs} ms TTFT above the ${slaTtftMs} ms max`] : [])
	]);
	const apiViolatesSlo = $derived(apiSloReasons.length > 0);
	const apiSloNote = $derived(
		apiViolatesSlo ? `API misses your SLO — ${apiSloReasons.join(' and ')}` : ''
	);

	const sanity = $derived([
		...configWarnings({
			tp: config.tp,
			gpusPerNode: config.gpusPerNode,
			fabricLabel: FABRICS_BY_ID.get(config.fabricId)?.label ?? config.fabricId,
			gpu,
			formatTier: WEIGHT_FORMATS_BY_ID.get(config.weightFormatId)?.tier ?? 'fp16'
		}),
		...(dutyPct === 100
			? [
					'Duty cycle 100% assumes the cluster runs flat-out every hour — the best case for self-hosting. Real deployments rarely exceed ~40–60%; lower it to see the honest break-even against the API.'
				]
			: [])
	]);

	const fmtUsd = (n: number) =>
		n >= 1e6
			? `$${(n / 1e6).toFixed(2)}M`
			: n >= 1e3
				? `$${(n / 1e3).toFixed(1)}k`
				: `$${n.toFixed(0)}`;
	const fmtReq = (n: number) =>
		n >= 1e9
			? `${(n / 1e9).toFixed(1)}B`
			: n >= 1e6
				? `${(n / 1e6).toFixed(1)}M`
				: n >= 1e3
					? `${(n / 1e3).toFixed(0)}k`
					: n.toFixed(0);
	const fmtTok = (n: number) =>
		n >= 1e12
			? `${(n / 1e12).toFixed(1)}T`
			: n >= 1e9
				? `${(n / 1e9).toFixed(1)}B`
				: n >= 1e6
					? `${(n / 1e6).toFixed(1)}M`
					: `${(n / 1e3).toFixed(0)}k`;

	const beReqPerDay = $derived(
		be?.breakevenReqPerMonth != null ? be.breakevenReqPerMonth / 30.4 : null
	);
	const verdict = $derived.by(() => {
		if (!be) return null;
		if (be.cheaper === 'self-host')
			return {
				title: 'Self-hosting wins',
				accent: '#22f2f7',
				save: be.apiMonthly - be.selfHostMonthly
			};
		if (be.cheaper === 'api')
			return { title: 'The API wins', accent: '#3ff0b8', save: be.selfHostMonthly - be.apiMonthly };
		return { title: 'Line-ball', accent: '#ecebff', save: 0 };
	});
</script>

<svelte:head><title>GenAI Calculator — Self-host vs API</title></svelte:head>

<div class="min-h-screen bg-slate-950 text-slate-100">
	<div class="mx-auto max-w-7xl px-6 py-8">
		<header class="mb-6 flex items-start justify-between gap-4">
			<div>
				<h1 class="text-2xl font-semibold">Self-host vs API</h1>
				<p class="mt-1 text-sm text-slate-400">
					When does owning GPUs beat paying per token? Configure the exact cluster (same controls as
					Modelling), set your API price, find the crossover.
				</p>
			</div>
			<div class="flex items-center gap-2">
				<ShareButton payload={config} />
				<HandoffMenu
					payload={config}
					targets={[
						{ path: 'modelling', label: 'Modelling' },
						{ path: '', label: 'Workload' },
						{ path: 'training', label: 'Training' }
					]}
				/>
			</div>
		</header>

		<HowTo>
			<p>
				This tab answers "<strong>should I run this model myself, or just call an API?</strong>".
				Self-hosting means renting GPUs and running the model yourself — you pay for the whole
				cluster around the clock, busy or not. An API charges <strong>per token</strong> — only for what
				you use, but each token costs a bit more.
			</p>
			<ol class="mt-2 ml-4 list-decimal space-y-1">
				<li>
					Configure the <strong>cluster</strong> with the full Modelling controls — model, GPU, GPU
					count, parallelism, quantization, and the request shape (input/output tokens). The forward
					model computes the cluster's throughput (the same engine as the
					<a href="/modelling">Modelling</a> tab).
				</li>
				<li>
					Set the <strong>duty cycle</strong>: what fraction of the month that cluster is actually
					busy. Flat-out all day is very different from spiking at lunch and idle overnight.
				</li>
				<li>
					Type the <strong>API price</strong> for the same model (input and output $/1M). Seeded from
					the model's real market price — replace with your quote.
				</li>
			</ol>
			<p class="mt-2">
				The app shows which is cheaper today, the <strong>break-even duty cycle</strong>, and a
				chart of both. The API is a straight line up from zero; self-hosting is a
				<strong>staircase</strong> — one cluster serves only so much, so cost jumps a step as you add
				servers.
			</p>
			<div
				class="mt-3 rounded-lg border border-amber-600/40 bg-amber-500/[0.06] p-3 text-xs text-amber-200"
			>
				<strong>These are first-order estimates.</strong> Self-host cost here is GPU rental only. It leaves
				out engineering/on-call, storage, data transfer, failover headroom, and cold-start. A managed
				API bundles that in. Treat the crossover as a starting point, then check it against real quotes
				from your providers.
			</div>
		</HowTo>

		<div class="grid grid-cols-1 gap-6 lg:grid-cols-[320px_1fr]">
			<!-- inputs: the Modelling control panel + break-even knobs -->
			<aside
				class="lg:sticky lg:top-6 lg:max-h-[calc(100vh-3rem)] lg:self-start lg:overflow-y-auto lg:pr-1"
			>
				<div class="flex flex-col gap-5 rounded-xl border border-slate-700 bg-slate-900/60 p-5">
					<Controls bind:config {isMoe} models={textModels} />

					<div class="border-t border-slate-700 pt-4">
						<div class="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
							Self-host cost
						</div>
						<Toggle
							label="Use estimated pricing"
							bind:checked={pricing.useEstimatedPricing}
							info="On: the median on-demand market rate for this GPU, with a typical discount for committed or spot capacity. Off: enter your own $/GPU-hour, such as a quote or a negotiated rate. Shared with the Modelling and Workload cost panels, so a rate set on any tab carries here."
						/>
						<div class="mt-3"></div>
						{#if pricing.useEstimatedPricing}
							<Segmented
								label="Purchasing"
								bind:value={pricing.purchasing}
								options={[
									{ value: 'on-demand', label: 'On-demand' },
									{ value: 'committed', label: 'Committed' },
									{ value: 'spot', label: 'Spot' }
								]}
								info="How you'd buy the GPUs. On-demand is the median market rate. Committed (a 1-3 year reservation) is about 35% less. Spot (interruptible) is about 60% less, but can be taken away."
							/>
						{:else}
							<label class="block">
								<span class="flex items-center gap-1.5 text-xs text-slate-300">
									$ / GPU-hour
									<InfoTip
										text="Your own price for one GPU for one hour. Whole-cluster cost is this × the {config.numGpus} GPU(s) sized here. Shared with the Modelling/Workload cost panels."
									/>
								</span>
								<input
									type="number"
									bind:value={pricing.manualPerGpuHour}
									min="0"
									step="0.01"
									class="mt-1 w-full max-w-[12rem] rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100"
								/>
							</label>
							<p class="mt-1.5 text-[11px] text-slate-500">
								Cluster / hour = {fmtUsd(pricing.manualPerGpuHour || 0)} × {config.numGpus} GPU = {clusterPerHour ==
								null
									? '—'
									: fmtUsd(clusterPerHour)}/hr. Your figure, not a market estimate.
							</p>
						{/if}
						<div class="mt-3"></div>
						<Slider
							label="Duty cycle"
							bind:value={dutyPct}
							min={1}
							max={100}
							step={1}
							display="{dutyPct}%"
							info="What fraction of the month the cluster runs at peak load. The crux: a self-hosted cluster costs the same whether flat-out or idle, so low utilisation is where the API wins."
						/>
					</div>

					<div class="border-t border-slate-700 pt-4">
						<div
							class="mb-2 flex items-center gap-1.5 text-xs font-medium tracking-wide text-slate-400 uppercase"
						>
							API price <InfoTip
								text="Per-token price for the same model through an API, in $/1M tokens. Seeded from the OpenRouter market price, else a typical market price, else the median of listed models. Editable."
							/>
						</div>
						<p class="mb-2 text-[10px] text-slate-500">
							{priceSourceFor(config.modelId) === 'openrouter'
								? `Default = OpenRouter market price, ${PRICE_SNAPSHOT}.`
								: priceSourceFor(config.modelId) === 'market'
									? 'Default = typical market price for this model.'
									: 'Not individually listed. Default is the median of listed models. Enter a real quote.'}
						</p>
						<div class="mb-2">
							<Segmented
								label="Pricing tier"
								bind:value={apiTier}
								options={API_TIERS.map((t) => ({
									value: t,
									label: t === 'batch' ? 'Batch (~−50%)' : 'Standard'
								}))}
								info="Most APIs sell the same tokens at two rates. Standard is the normal synchronous rate; Batch is asynchronous bulk jobs at about half price. Picking a tier reseeds the API price below (still editable)."
							/>
							<div class="mt-2">
								<Toggle
									label="Prompt caching"
									bind:checked={cacheOn}
									info="Model prompt caching: the cached-prefix fraction (set on Modelling/Workload) is billed at the cache read rate, not the full input price. Uses the listed read rate where known, else 10% of input."
								/>
							</div>
						</div>
						<label class="mb-2 block">
							<span class="text-xs text-slate-400">$ / 1M input tokens</span>
							<input
								type="number"
								bind:value={apiInPerM}
								min="0"
								step="0.01"
								class="mt-1 w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100"
							/>
						</label>
						<label class="block">
							<span class="text-xs text-slate-400">$ / 1M output tokens</span>
							<input
								type="number"
								bind:value={apiOutPerM}
								min="0"
								step="0.01"
								class="mt-1 w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100"
							/>
						</label>
						<div class="mt-4 mb-1 text-xs font-medium tracking-wide text-slate-400 uppercase">
							What the API delivers
						</div>
						<p class="mb-2 text-[10px] text-slate-500">
							Typical values. Enter your provider's real numbers.
						</p>
						<div class="grid grid-cols-2 gap-2">
							<label class="block">
								<span class="flex items-center gap-1.5 text-xs text-slate-400"
									>API output (tok/s)
									<InfoTip
										text="How fast the API streams ONE user's answer (tok/s). Typical hosted ~30–100; fast-silicon several hundred. Checked against the min-throughput SLO below; doesn't change the cost math."
									/></span
								>
								<input
									type="number"
									bind:value={apiPerUserTps}
									min="1"
									step="1"
									class="mt-1 w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100"
								/>
							</label>
							<label class="block">
								<span class="flex items-center gap-1.5 text-xs text-slate-400"
									>API TTFT (ms)
									<InfoTip
										text="API time-to-first-token for one request (ms), typically ~200–600. Checked against the Max TTFT SLO below; doesn't change the cost math."
									/></span
								>
								<input
									type="number"
									bind:value={apiTtftMs}
									min="1"
									step="10"
									class="mt-1 w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100"
								/>
							</label>
						</div>
						<div class="mt-4 mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
							Your requirement (per-user SLO)
						</div>
						<div class="grid grid-cols-2 gap-2">
							<label class="block">
								<span class="flex items-center gap-1.5 text-xs text-slate-400"
									>Min throughput (tok/s)
									<InfoTip
										text="Per-user speed you require. The API line goes red if its output rate is below this — the price comparison is moot if it can't meet the SLO."
									/></span
								>
								<input
									type="number"
									bind:value={slaTps}
									min="1"
									step="5"
									class="mt-1 w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100"
								/>
							</label>
							<label class="block">
								<span class="flex items-center gap-1.5 text-xs text-slate-400"
									>Max TTFT (ms)
									<InfoTip
										text="Longest acceptable time to first token. The API line goes red if its TTFT exceeds this."
									/></span
								>
								<input
									type="number"
									bind:value={slaTtftMs}
									min="10"
									step="10"
									class="mt-1 w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100"
								/>
							</label>
						</div>
					</div>
				</div>
			</aside>

			<!-- result -->
			<main class="flex flex-col gap-6">
				<SanityNotes notes={sanity} />
				{#if !p.perGpu.fits}
					<div class="rounded-xl border border-red-500/50 bg-red-500/10 p-4 text-sm text-red-300">
						<span class="font-medium"
							>{model?.name ?? config.modelId} doesn't fit on {config.numGpus}× {gpu.name}</span
						>
						— it overflows GPU memory at this config. Add GPUs, raise TP/PP, or use a smaller weight format
						(see the <a href="/modelling" class="underline">Modelling</a> tab), then the cost can be compared.
					</div>
				{:else if clusterPerHour == null}
					<div
						class="rounded-xl border border-amber-500/50 bg-amber-500/10 p-4 text-sm text-amber-200"
					>
						{#if !pricing.useEstimatedPricing}
							Enter a $/GPU-hour above to price the self-hosted cluster.
						{:else}
							No price for {gpu.name} under this purchasing model{cost.warning
								? ` — ${cost.warning}`
								: ''}. Switch the purchasing model, or turn off estimated pricing and enter your own
							rate.
						{/if}
					</div>
				{:else if be}
					<!-- verdict -->
					<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
						<div
							class="col-span-2 rounded-xl border border-slate-700 bg-slate-800/50 p-4 sm:col-span-2"
						>
							<div class="text-xs tracking-wide text-slate-400 uppercase">
								At {dutyPct}% duty cycle
							</div>
							<div class="mt-1 text-3xl font-semibold" style:color={verdict?.accent}>
								{verdict?.title}
							</div>
							<div class="mt-1 text-sm text-slate-400">
								{#if be.cheaper === 'equal'}Both cost about {fmtUsd(
										be.selfHostMonthly
									)}/mo.{:else}Saves ~<span class="text-slate-200"
										>{fmtUsd(verdict?.save ?? 0)}/mo</span
									> vs the other option.{/if}
							</div>
						</div>
						<StatCard
							label="Self-host"
							value={fmtUsd(be.selfHostMonthly)}
							sub="/mo · {fmtUsd(clusterPerHour ?? 0)}/hr · {config.numGpus}× {gpu.name}"
							accent="#22f2f7"
							info="Cost to rent the cluster for a month, billed around the clock. Fixed — independent of usage."
						/>
						<StatCard
							label="API"
							value={fmtUsd(be.apiMonthly)}
							sub="/mo at this volume"
							accent="#3ff0b8"
							info="Per-token API cost for the same traffic this month: requests × (input × $in + output × $out)."
						/>
					</div>

					<!-- break-even -->
					<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
						<StatCard
							label="Break-even"
							value={be.regime === 'crossover' && beReqPerDay != null
								? `${fmtReq(beReqPerDay)}/day`
								: be.regime === 'api-always'
									? 'never'
									: 'always'}
							accent="#f6bd6a"
							sub={be.regime === 'crossover'
								? `≈ ${fmtTok(be.breakevenOutTokensPerMonth ?? 0)} out tok/mo`
								: be.regime === 'api-always'
									? 'API cheaper at any utilisation'
									: 'self-host cheaper at any volume'}
							info="The request volume where the two cost the same. Below it the API is cheaper; above it self-hosting is."
						/>
						<StatCard
							label="Break-even duty cycle"
							value={be.breakevenDutyCycle != null && be.regime === 'crossover'
								? `${(be.breakevenDutyCycle * 100).toFixed(0)}%`
								: '—'}
							sub={be.regime === 'crossover' ? `you're at ${dutyPct}%` : be.regime}
							accent={be.breakevenDutyCycle != null && dutyPct / 100 >= be.breakevenDutyCycle
								? '#22f2f7'
								: '#3ff0b8'}
							info="How busy the cluster must be for self-hosting to pay off. Above this, self-host wins; below, the API."
						/>
						<StatCard
							label="Self-host $/1M"
							value={be.selfHostPerMTokens != null ? `$${be.selfHostPerMTokens.toFixed(2)}` : '—'}
							sub="blended, at {dutyPct}%"
							accent="#22f2f7"
							info="Effective self-host $/1M tokens at your utilisation. Falls as the cluster gets busier."
						/>
						<StatCard
							label="API $/1M"
							value="${be.apiPerMTokens.toFixed(2)}"
							sub="blended in+out"
							accent="#3ff0b8"
							info="The API's effective $/1M for your input/output mix. Constant at any volume."
						/>
					</div>

					<!-- per-user speed -->
					<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
						<div class="col-span-2 rounded-xl border border-slate-700 bg-slate-800/50 p-4">
							<div class="flex items-center gap-1.5 text-xs tracking-wide text-slate-400 uppercase">
								Per-user speed
								<InfoTip
									text="How fast one user's answer streams (tok/s). An API has a bounded per-request rate (your input). Self-host streams at the sized batch's decode rate. Experience axis; doesn't change the cost verdict."
								/>
							</div>
							<div
								class="mt-1 text-lg font-semibold"
								style:color={p.perUserTps >= apiPerUserTps ? '#22f2f7' : '#3ff0b8'}
							>
								{p.perUserTps >= apiPerUserTps
									? 'Self-host streams faster'
									: 'The API streams faster'}
							</div>
							<div class="mt-1 text-xs text-slate-400">
								Your min-throughput SLO is {slaTps} tok/s/user —
								{#if apiPerUserTps >= slaTps}both meet it.{:else}<span class="text-amber-300"
										>the API ({apiPerUserTps} tok/s) is below it</span
									>; self-host ({p.perUserTps.toFixed(0)} tok/s) meets it.{/if}
							</div>
						</div>
						<StatCard
							label="Self-host / user"
							value={p.perUserTps.toFixed(0)}
							unit="tok/s"
							accent="#22f2f7"
							sub="at batch {config.batchSize}"
							info="Decode speed one user feels on the cluster at the configured batch."
						/>
						<StatCard
							label="API / user"
							value={String(apiPerUserTps)}
							unit="tok/s"
							accent={apiPerUserTps < slaTps ? '#ff3b52' : '#3ff0b8'}
							sub={apiPerUserTps < slaTps ? `below your ${slaTps} tok/s SLO` : 'your entered limit'}
							info="The API per-user streaming rate, from your input — not infinite."
						/>
					</div>

					<BreakevenChart
						unitCostMonthly={be.selfHostMonthly}
						capReqPerMonth={be.peakReqPerSec * 730 * 3600}
						apiCostPerRequest={be.apiCostPerRequest}
						breakevenReqPerMonth={be.breakevenReqPerMonth}
						currentReqPerMonth={be.reqPerMonth}
						regime={be.regime}
						{apiViolatesSlo}
						{apiSloNote}
						{fmtUsd}
						{fmtReq}
					/>

					<!-- traffic + cluster detail -->
					<section class="rounded-xl border border-slate-700 bg-slate-900/60 p-5">
						<div class="mb-3 text-sm font-medium text-slate-200">
							This month, at {dutyPct}% duty cycle
						</div>
						<div class="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
							<div class="flex justify-between">
								<span class="text-slate-400">Requests</span><span class="font-mono text-slate-200"
									>{fmtReq(be.reqPerMonth)}</span
								>
							</div>
							<div class="flex justify-between">
								<span class="text-slate-400">Input tokens</span><span
									class="font-mono text-slate-200">{fmtTok(be.inTokensPerMonth)}</span
								>
							</div>
							<div class="flex justify-between">
								<span class="text-slate-400">Output tokens</span><span
									class="font-mono text-slate-200">{fmtTok(be.outTokensPerMonth)}</span
								>
							</div>
							<div class="flex justify-between">
								<span class="text-slate-400">Peak rate</span><span class="font-mono text-slate-200"
									>{be.peakReqPerSec.toFixed(1)} req/s</span
								>
							</div>
							<div class="flex justify-between">
								<span class="text-slate-400">Cluster throughput</span><span
									class="font-mono text-slate-200"
									>{(p.throughputTps / 1000).toFixed(1)}k tok/s</span
								>
							</div>
							<div class="flex justify-between">
								<span class="text-slate-400">Cluster</span><span class="font-mono text-slate-200"
									>{config.numGpus}× {gpu.name} · TP{config.tp}·PP{config.pp}{config.ep > 1
										? `·EP${config.ep}`
										: ''}</span
								>
							</div>
						</div>
						{#if cost.warning}<p class="mt-3 text-[11px] text-amber-300">{cost.warning}</p>{/if}
						<p class="mt-3 text-[10px] text-slate-500">
							Self-host is billed 24/7 for the cluster you configured. API volume is derived from
							the cluster's peak throughput scaled by the duty cycle, so both sides serve the same
							traffic.
						</p>
					</section>
				{/if}
			</main>
		</div>
	</div>
</div>
