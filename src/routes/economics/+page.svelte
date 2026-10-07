<script lang="ts">
	import { base } from '$app/paths';
	// Self-host vs API, built on the SAME Modelling setup: you configure the exact cluster
	// with the full Modelling control panel, the forward model (computeProfile) gives its
	// throughput, we cost that cluster and compare to a per-token API. Because it uses the
	// Modelling `Config`, it shares every field with Modelling/Workload through the shared scenario.
	import BreakevenChart from '$lib/components/BreakevenChart.svelte';
	import Controls from '$lib/components/Controls.svelte';
	import HowTo from '$lib/components/HowTo.svelte';
	import InfoTip from '$lib/components/InfoTip.svelte';
	import Segmented from '$lib/components/Segmented.svelte';
	import Slider from '$lib/components/Slider.svelte';
	import StatCard from '$lib/components/StatCard.svelte';
	import ShareButton from '$lib/components/ShareButton.svelte';
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
	import {
		loadShared,
		saveShared,
		readPricing,
		LocalModelFallback,
		type PricingState
	} from '$lib/state/shared.svelte';
	import Toggle from '$lib/components/Toggle.svelte';
	import { onMount, untrack } from 'svelte';

	// Only token-priced text models make sense against a per-token API.
	const textModels = MODELS.filter(
		(m) => (m.kind ?? 'transformer') === 'transformer' && !m.visualAR
	);

	// Full Modelling config (same shape → shares every field). Realistic mid-size default.
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

	// Break-even knobs (kept out of Config so the share link stays clean).
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
		// This tab is token-priced only; a non-text model falls back to Llama, locally.
		modelFallback.apply(config, (id) => textModels.some((m) => m.id === id));
		syncLoaded = true;
	});
	const modelFallback = new LocalModelFallback('llama31-70b');
	$effect(() => modelFallback.track(config.modelId));
	$effect(() => {
		if (syncLoaded)
			saveShared(
				{ ...(config as unknown as Record<string, unknown>), ...pricing },
				{ except: modelFallback.except }
			);
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

<div>
	<div class="page">
		<header class="page-head">
			<div>
				<h1 class="page-title">Self-host vs API</h1>
				<p class="lede">
					When does owning GPUs beat paying per token? Configure the exact cluster (same controls as
					Modelling), set your API price, find the crossover.
				</p>
			</div>
			<div class="row">
				<ShareButton payload={config} />
			</div>
		</header>

		<HowTo>
			<p>
				This tab answers "<strong>should I run this model myself, or just call an API?</strong>".
				Self-hosting means renting GPUs and running the model yourself — you pay for the whole
				cluster around the clock, busy or not. An API charges <strong>per token</strong> — only for what
				you use, but each token costs a bit more.
			</p>
			<ol class="bullets numbered space-top-s">
				<li>
					Configure the <strong>cluster</strong> with the full Modelling controls — model, GPU, GPU
					count, parallelism, quantization, and the request shape (input/output tokens). The forward
					model computes the cluster's throughput (the same engine as the
					<a href="{base}/modelling">Modelling</a> tab).
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
			<p class="space-top-s">
				The app shows which is cheaper today, the <strong>break-even duty cycle</strong>, and a
				chart of both. The API is a straight line up from zero; self-hosting is a
				<strong>staircase</strong> — one cluster serves only so much, so cost jumps a step as you add
				servers.
			</p>
			<div class="notice space-top">
				<strong>These are first-order estimates.</strong> Self-host cost here is GPU rental only. It leaves
				out engineering/on-call, storage, data transfer, failover headroom, and cold-start. A managed
				API bundles that in. Treat the crossover as a starting point, then check it against real quotes
				from your providers.
			</div>
		</HowTo>

		<div class="layout">
			<!-- inputs: the Modelling control panel + break-even knobs -->
			<aside
				class="sidebar"
			>
				<div class="panel controls">
					<Controls bind:config {isMoe} models={textModels} />

					<div class="group">
						<div class="group-title">
							Self-host cost
						</div>
						<Toggle
							label="Use estimated pricing"
							bind:checked={pricing.useEstimatedPricing}
							info="On: the median on-demand market rate for this GPU, with a typical discount for committed or spot capacity. Off: enter your own $/GPU-hour, such as a quote or a negotiated rate. Shared with the Modelling and Workload cost panels, so a rate set on any tab carries here."
						/>
						<div class="space-top"></div>
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
							<label class="field">
								<span class="field-label small">
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
									class="input narrow"
								/>
							</label>
							<p class="hint">
								Cluster / hour = {fmtUsd(pricing.manualPerGpuHour || 0)} × {config.numGpus} GPU = {clusterPerHour ==
								null
									? '—'
									: fmtUsd(clusterPerHour)}/hr. Your figure, not a market estimate.
							</p>
						{/if}
						<div class="space-top"></div>
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

					<div class="group">
						<div class="group-title row">
							API price <InfoTip
								text="Per-token price for the same model through an API, in $/1M tokens. Seeded from the OpenRouter market price, else a typical market price, else the median of listed models. Editable."
							/>
						</div>
						<p class="hint space-bottom-s">
							{priceSourceFor(config.modelId) === 'openrouter'
								? `Default = OpenRouter market price, ${PRICE_SNAPSHOT}.`
								: priceSourceFor(config.modelId) === 'market'
									? 'Default = typical market price for this model.'
									: 'Not individually listed. Default is the median of listed models. Enter a real quote.'}
						</p>
						<div class="space-bottom-s">
							<Segmented
								label="Pricing tier"
								bind:value={apiTier}
								options={API_TIERS.map((t) => ({
									value: t,
									label: t === 'batch' ? 'Batch (~−50%)' : 'Standard'
								}))}
								info="Most APIs sell the same tokens at two rates. Standard is the normal synchronous rate; Batch is asynchronous bulk jobs at about half price. Picking a tier reseeds the API price below (still editable)."
							/>
							<div class="space-top-s">
								<Toggle
									label="Prompt caching"
									bind:checked={cacheOn}
									info="Model prompt caching: the cached-prefix fraction (set on Modelling/Workload) is billed at the cache read rate, not the full input price. Uses the listed read rate where known, else 10% of input."
								/>
							</div>
						</div>
						<label class="field space-bottom-s">
							<span class="small dim">$ / 1M input tokens</span>
							<input
								type="number"
								bind:value={apiInPerM}
								min="0"
								step="0.01"
								class="input"
							/>
						</label>
						<label class="field">
							<span class="small dim">$ / 1M output tokens</span>
							<input
								type="number"
								bind:value={apiOutPerM}
								min="0"
								step="0.01"
								class="input"
							/>
						</label>
						<div class="group-title spaced-above">
							What the API delivers
						</div>
						<p class="hint space-bottom-s">
							Typical values. Enter your provider's real numbers.
						</p>
						<div class="pair">
							<label class="field">
								<span class="field-label small"
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
									class="input"
								/>
							</label>
							<label class="field">
								<span class="field-label small"
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
									class="input"
								/>
							</label>
						</div>
						<div class="group-title spaced-above">
							Your requirement (per-user SLO)
						</div>
						<div class="pair">
							<label class="field">
								<span class="field-label small"
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
									class="input"
								/>
							</label>
							<label class="field">
								<span class="field-label small"
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
									class="input"
								/>
							</label>
						</div>
					</div>
				</div>
			</aside>

			<!-- result -->
			<main class="results">
				<SanityNotes notes={sanity} />
				{#if !p.perGpu.fits}
					<div class="notice danger big">
						<span class="notice-title"
							>{model?.name ?? config.modelId} doesn't fit on {config.numGpus}× {gpu.name}</span
						>
						— it overflows GPU memory at this config. Add GPUs, raise TP/PP, or use a smaller weight format
						(see the <a href="{base}/modelling">Modelling</a> tab), then the cost can be compared.
					</div>
				{:else if clusterPerHour == null}
					<div class="notice big">
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
					<div class="tiles">
						<div class="verdict">
							<div class="verdict-label">
								At {dutyPct}% duty cycle
							</div>
							<div class="verdict-title" style:color={verdict?.accent}>
								{verdict?.title}
							</div>
							<div class="lede">
								{#if be.cheaper === 'equal'}Both cost about {fmtUsd(
										be.selfHostMonthly
									)}/mo.{:else}Saves ~<span class="bright"
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
					<div class="tiles">
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
					<div class="tiles">
						<div class="verdict">
							<div class="verdict-label">
								Per-user speed
								<InfoTip
									text="How fast one user's answer streams (tok/s). An API has a bounded per-request rate (your input). Self-host streams at the sized batch's decode rate. Experience axis; doesn't change the cost verdict."
								/>
							</div>
							<div
								class="verdict-title small"
								style:color={p.perUserTps >= apiPerUserTps ? '#22f2f7' : '#3ff0b8'}
							>
								{p.perUserTps >= apiPerUserTps
									? 'Self-host streams faster'
									: 'The API streams faster'}
							</div>
							<div class="tile-sub">
								Your min-throughput SLO is {slaTps} tok/s/user —
								{#if apiPerUserTps >= slaTps}both meet it.{:else}<span class="warn"
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
					<section class="panel">
						<h3 class="panel-title spaced">
							This month, at {dutyPct}% duty cycle
						</h3>
						<div class="facts">
							<div class="kv">
								<span class="dim">Requests</span><span class="num bright"
									>{fmtReq(be.reqPerMonth)}</span
								>
							</div>
							<div class="kv">
								<span class="dim">Input tokens</span><span
									class="num bright">{fmtTok(be.inTokensPerMonth)}</span
								>
							</div>
							<div class="kv">
								<span class="dim">Output tokens</span><span
									class="num bright">{fmtTok(be.outTokensPerMonth)}</span
								>
							</div>
							<div class="kv">
								<span class="dim">Peak rate</span><span class="num bright"
									>{be.peakReqPerSec.toFixed(1)} req/s</span
								>
							</div>
							<div class="kv">
								<span class="dim">Cluster throughput</span><span
									class="num bright"
									>{(p.throughputTps / 1000).toFixed(1)}k tok/s</span
								>
							</div>
							<div class="kv">
								<span class="dim">Cluster</span><span class="num bright"
									>{config.numGpus}× {gpu.name} · TP{config.tp}·PP{config.pp}{config.ep > 1
										? `·EP${config.ep}`
										: ''}</span
								>
							</div>
						</div>
						{#if cost.warning}<p class="hint warn space-top">{cost.warning}</p>{/if}
						<p class="note space-top">
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
