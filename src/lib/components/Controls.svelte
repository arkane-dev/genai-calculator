<script lang="ts">
	import {
		defaultFabricFor,
		FABRICS,
		GPUS,
		GPUS_BY_ID,
		gpuSupportsFormat,
		GPUS_PER_NODE,
		MODELS,
		WEIGHT_FORMATS,
		WEIGHT_FORMATS_BY_ID
	} from '$lib/profiler/data';
	import type { Config, ModelSpec } from '$lib/profiler/types';
	import InfoTip from './InfoTip.svelte';
	import Segmented from './Segmented.svelte';
	import Slider from './Slider.svelte';
	import Toggle from './Toggle.svelte';

	interface Props {
		config: Config;
		isMoe: boolean;
		models?: ModelSpec[];
	}
	let { config = $bindable(), isMoe, models = MODELS }: Props = $props();

	const modelOf = $derived(models.find((mm) => mm.id === config.modelId));
	const gpuOf = $derived(GPUS_BY_ID.get(config.gpuId));
	// If the selected format isn't supported by the current GPU (e.g. FP4 after
	// switching to a non-Blackwell card, or a shared link), fall back to BF16.
	$effect(() => {
		const f = WEIGHT_FORMATS_BY_ID.get(config.weightFormatId);
		if (gpuOf && f && !gpuSupportsFormat(gpuOf, f)) config.weightFormatId = 'bf16';
	});
	const isDiffusion = $derived(modelOf?.kind === 'diffusion');
	const isVideoDiff = $derived(isDiffusion && (modelOf?.diffusion?.frames ?? 1) > 1);
	const isJepa = $derived(modelOf?.kind === 'jepa');
	const isEncoder = $derived(modelOf?.kind === 'encoder');
	const isVla = $derived(modelOf?.kind === 'vla');
	const isVlm = $derived(!isVla && !!modelOf?.vlm);
	const isAsr = $derived(modelOf?.kind === 'asr');
	const VID_FRAMES = [5, 9, 13, 17, 21];
	// group the dropdown by model kind
	const groups = $derived(
		[
			{
				label: 'Language models',
				items: models.filter(
					(mm) =>
						(mm.kind ?? 'transformer') === 'transformer' &&
						!mm.visualAR &&
						!mm.vlm &&
						!mm.reasoning &&
						!mm.tts
				)
			},
			{ label: 'Reasoning models', items: models.filter((mm) => mm.reasoning) },
			{ label: 'Speech synthesis (TTS)', items: models.filter((mm) => mm.tts) },
			{
				label: 'Vision-language (VLM)',
				items: models.filter((mm) => (mm.kind ?? 'transformer') === 'transformer' && mm.vlm)
			},
			{ label: 'Vision-language-action (VLA)', items: models.filter((mm) => mm.kind === 'vla') },
			{ label: 'Autoregressive image', items: models.filter((mm) => mm.visualAR) },
			{ label: 'Diffusion models', items: models.filter((mm) => mm.kind === 'diffusion') },
			{ label: 'Embeddings & rerankers', items: models.filter((mm) => mm.kind === 'encoder') },
			{ label: 'Speech (ASR)', items: models.filter((mm) => mm.kind === 'asr') },
			{ label: 'World models (JEPA)', items: models.filter((mm) => mm.kind === 'jepa') }
		].filter((gp) => gp.items.length)
	);
	// reseed the runtime knobs to the model's defaults when a diffusion or JEPA
	// model is selected (and when switching between them)
	let seededFor = $state('');
	$effect(() => {
		const dsm = modelOf?.diffusion;
		const jm = modelOf?.jepa;
		if (dsm && config.modelId !== seededFor) {
			config.steps = dsm.defaultSteps;
			config.resolution = dsm.defaultResolution;
			config.guidance = dsm.cfg;
			config.frames = dsm.frames ?? 1;
			if (seededFor === '' || config.batchSize > 16) config.batchSize = 1; // single image by default
			seededFor = config.modelId;
		} else if (jm && config.modelId !== seededFor) {
			config.resolution = jm.defaultResolution;
			config.frames = jm.defaultFrames;
			if (seededFor === '' || config.batchSize > 16) config.batchSize = 1; // single clip by default
			seededFor = config.modelId;
		} else if (modelOf?.encoder && config.modelId !== seededFor) {
			config.inputTokens = modelOf.encoder.defaultSeqLen;
			config.outputTokens = 64; // unused for encoders; keep the slider valid
			config.phase = 'prefill';
			if (seededFor === '' || config.batchSize > 32) config.batchSize = 8; // batching docs is the norm
			seededFor = config.modelId;
		} else if (modelOf?.kind === 'vla' && config.modelId !== seededFor) {
			config.batchSize = 1; // one observation per inference cycle
			seededFor = config.modelId;
		} else if (modelOf?.kind === 'asr' && config.modelId !== seededFor) {
			config.batchSize = 1;
			config.phase = 'decode';
			seededFor = config.modelId;
		} else if (modelOf?.vlm && config.modelId !== seededFor) {
			config.imagesPerRequest = 1;
			seededFor = config.modelId;
		} else if (!modelOf?.vlm && !modelOf?.vla && config.modelId !== seededFor) {
			// switching away from a VLM/VLA back to a plain LLM: reset the image count
			if (config.imagesPerRequest > 0) config.imagesPerRequest = 0;
			seededFor = config.modelId;
		}
	});

	const RES = [256, 512, 768, 1024, 1536, 2048];
	const JEPA_RES = [224, 256, 384, 512];
	const FRAMES = [8, 16, 32, 64];

	const INSTANCE_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128];
	const powersUpTo = (n: number) =>
		[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].filter((p) => p <= n);

	// The cluster is described as instances (nodes) × GPUs-per-node; total GPUs is the
	// product. numGpus stays the source of truth for the calc. The two effects keep them
	// in sync both ways: an EXTERNAL numGpus change (default or a shared ?c= link) re-derives
	// the instance count; a UI change to nodes/gpusPerNode writes numGpus back.
	let nodes = $state(Math.max(1, Math.round(config.numGpus / config.gpusPerNode)));
	let lastNumGpus = config.numGpus;
	$effect(() => {
		if (config.numGpus !== lastNumGpus) {
			nodes = Math.max(1, Math.round(config.numGpus / config.gpusPerNode));
			lastNumGpus = config.numGpus;
		}
	});
	$effect(() => {
		const want = nodes * config.gpusPerNode;
		if (config.numGpus !== want) {
			config.numGpus = want;
			lastNumGpus = want;
		}
	});

	// Always include the current node count so a synced/solver value that isn't a power of
	// two (e.g. 6 nodes = 48 GPUs) still shows as a selected chip rather than nothing.
	const instanceOptions = $derived(
		[...new Set([...INSTANCE_COUNTS, nodes])]
			.sort((a, b) => a - b)
			.map((n) => ({ value: n, label: String(n) }))
	);
	const nodeSizeOptions = $derived(
		[...new Set<number>([...GPUS_PER_NODE, config.gpusPerNode])]
			.sort((a, b) => a - b)
			.map((n) => ({ value: n, label: String(n) }))
	);
	const parallelOptions = $derived(
		powersUpTo(config.numGpus).map((n) => ({ value: n, label: String(n) }))
	);

	const multipathFabrics = FABRICS.filter((f) => f.kind === 'multipath');
	const otherFabrics = FABRICS.filter((f) => f.kind !== 'multipath');

	// Keep every degree within the GPU budget: tp*pp must leave room for >=1 replica.
	$effect(() => {
		if (config.tp > config.numGpus) config.tp = config.numGpus;
		const maxPp = Math.max(1, Math.floor(config.numGpus / config.tp));
		if (config.pp > maxPp) config.pp = maxPp;
		if (config.ep > config.numGpus) config.ep = config.numGpus;
	});

	const fmtSeq = (n: number) =>
		n >= 1048576
			? `${(n / 1048576).toFixed(n % 1048576 ? 1 : 0)}M`
			: n >= 1024
				? `${(n / 1024).toFixed(0)}k`
				: String(n);
</script>

<div class="controls">
	<div>
		<div class="field-label">
			Model
			<InfoTip
				text="The AI model you want to run. Bigger models are smarter but need more memory and run slower. 'MoE' models are large but only use a slice of themselves for each word, so they run faster than their size suggests. You can also upload your own."
			/>
		</div>
		<select
			aria-label="Model"
			bind:value={config.modelId}
			class="input"
		>
			{#each groups as grp (grp.label)}
				<optgroup label={grp.label}>
					{#each grp.items as m (m.id)}
						<option value={m.id}>{m.name}{m.projected ? ' (projected)' : ''}</option>
					{/each}
				</optgroup>
			{/each}
		</select>
	</div>

	<div>
		<div class="field-label">
			GPU
			<InfoTip
				text="The chip that runs the model. Each GPU has a fixed amount of memory and a top math speed. The app uses the real specs of the one you pick to work out whether the model fits and how fast it goes."
			/>
		</div>
		<select
			aria-label="GPU"
			bind:value={config.gpuId}
			onchange={() => (config.fabricId = defaultFabricFor(config.gpuId))}
			class="input"
		>
			{#each GPUS as g (g.id)}
				<option value={g.id}>{g.name}</option>
			{/each}
		</select>
	</div>

	<Segmented
		label="GPUs per node"
		bind:value={config.gpusPerNode}
		options={nodeSizeOptions}
		info="How many GPUs sit inside one server (instance). GPUs in the same server talk over very fast NVLink cables; GPUs in different servers use the slower network. Most GPU instances are 8× per node."
	/>
	<Segmented
		label="Instances (nodes)"
		bind:value={nodes}
		options={instanceOptions}
		info="How many servers (instances) you're using. Total GPUs = instances × GPUs-per-node. More GPUs give more memory and serve more users; a single model only runs faster if you also split it across them (see the parallelism controls below)."
	/>
	<p class="hint tight">
		= <span class="num dim"
			>{config.numGpus} GPU{config.numGpus === 1 ? '' : 's'}</span
		>
		total ({nodes} × {config.gpusPerNode})
	</p>

	<div>
		<div class="field-label">
			Network fabric (per GPU)
			<InfoTip
				text="The network that connects separate servers, such as InfiniBand or RDMA Ethernet. It only matters when a model is split across more than one server — then this is how the servers pass results to each other."
			/>
		</div>
		<select
			aria-label="Network fabric (per GPU)"
			bind:value={config.fabricId}
			class="input"
		>
			<optgroup label="InfiniBand / Ethernet">
				{#each otherFabrics as f (f.id)}
					<option value={f.id}>{f.label}</option>
				{/each}
			</optgroup>
			<optgroup label="Multipath RDMA (adaptive routing)">
				{#each multipathFabrics as f (f.id)}
					<option value={f.id}>{f.label}</option>
				{/each}
			</optgroup>
		</select>
	</div>

	<Segmented
		label="Tensor parallel (TP)"
		bind:value={config.tp}
		options={parallelOptions}
		info="Splits each layer's math across several GPUs so they share one big calculation. Helps a model that's too big or too slow for one GPU, but the GPUs must talk constantly, so very high TP can waste time on chatter."
	/>

	<div class="tile stack-s">
		<Toggle
			label="Pipeline parallel (PP)"
			bind:checked={config.ppEnabled}
			hint="split layers into stages"
			info="Puts different layers of the model on different GPUs, like stations on an assembly line. Turn on when a model is too big for tensor-parallel alone. It adds a little delay as work passes down the line."
		/>
		{#if config.ppEnabled}
			<Segmented
				label="PP degree"
				bind:value={config.pp}
				options={parallelOptions}
				info="How many assembly-line stages to use. More stages fit a bigger model but add more hand-offs between GPUs."
			/>
		{/if}
	</div>

	{#if isMoe && !isDiffusion && !isJepa && !isEncoder}
		<div class="tile stack-s">
			<Toggle
				label="Expert parallel (EP)"
				bind:checked={config.epEnabled}
				hint="shard experts, all-to-all routing"
				info="Spreads a MoE model's many 'experts' across GPUs so each stores only some of them. This is how very large MoE models fit in memory. It needs an all-to-all exchange between GPUs every step."
			/>
			{#if config.epEnabled}
				<Segmented
					label="EP degree"
					bind:value={config.ep}
					options={parallelOptions}
					info="How many groups to spread the experts across. Higher lets a bigger MoE fit, at the cost of more all-to-all network traffic."
				/>
			{/if}
		</div>
	{/if}

	{#if isDiffusion}
		<Segmented
			label="Resolution"
			bind:value={config.resolution}
			options={RES.map((v) => ({ value: v, label: `${v}²` }))}
			info="Image size in pixels per side. Larger images have far more latent patches to denoise, so time and memory rise with the area — doubling the side is roughly 4× the work."
		/>
		<Slider
			label="Denoising steps"
			bind:value={config.steps}
			min={1}
			max={100}
			step={1}
			info="How many denoising passes to run. Each step is one full pass over the {isVideoDiff
				? 'video'
				: 'image'}; total time is steps × step time. Fewer steps are faster but can look rougher."
		/>
		{#if isVideoDiff}
			<Segmented
				label="Video length (latent frames)"
				bind:value={config.frames}
				options={VID_FRAMES.map((v) => ({ value: v, label: String(v) }))}
				info="How long the clip is, measured in the model's compressed 'latent' frames — the 3D VAE squeezes several real frames into each one. More frames means far more space-time patches, and because attention is quadratic, the compute climbs fast."
			/>
		{/if}
		<Slider
			label={isVideoDiff ? 'Batch (clips)' : 'Batch (images)'}
			bind:value={config.batchSize}
			min={1}
			max={64}
			step={1}
			info="How many {isVideoDiff
				? 'clips'
				: 'images'} to generate at once. Bigger batches use the GPU more fully (higher throughput) but need more memory — and don't make any single one finish sooner."
		/>
		<Toggle
			label="Classifier-free guidance"
			bind:checked={config.guidance}
			hint="cond + uncond pass"
			info="Runs two forward passes per step (with and without the prompt) to sharpen prompt-following — doubling the work. Some models (e.g. FLUX.1-dev) are 'distilled' and skip this, so leave it off for them."
		/>
	{:else if isJepa}
		<Segmented
			label="Resolution"
			bind:value={config.resolution}
			options={JEPA_RES.map((v) => ({ value: v, label: `${v}²` }))}
			info="Frame size in pixels per side. Each frame is cut into patches; larger frames mean more patches to encode, so time and memory rise with the area."
		/>
		<Segmented
			label="Frames per clip"
			bind:value={config.frames}
			options={FRAMES.map((v) => ({ value: v, label: String(v) }))}
			info="How many video frames the model reads at once. Frames are grouped in twos (tubelets), so more frames mean proportionally more patches — and, because attention is quadratic, more than proportionally more compute."
		/>
		<Slider
			label="Batch (clips)"
			bind:value={config.batchSize}
			min={1}
			max={64}
			step={1}
			info="How many video clips to encode at once. Bigger batches use the GPU more fully (higher clips/sec) but need more memory — and don't make any single clip finish sooner."
		/>
	{:else if isEncoder}
		<Slider
			label="Sequence length"
			bind:value={config.inputTokens}
			min={64}
			max={modelOf?.encoder?.maxSeqLen ?? 8192}
			log
			display="{fmtSeq(config.inputTokens)} tok"
			info="How many tokens the encoder reads per document (or per query+doc for a reranker). Longer sequences quadruple attention cost, so short passages are much cheaper than long ones."
		/>
		<Slider
			label={modelOf?.encoder?.task === 'reranker' ? 'Batch (pairs)' : 'Batch (docs)'}
			bind:value={config.batchSize}
			min={1}
			max={512}
			step={1}
			info="How many documents (or query-doc pairs) to encode at once. Encoders love big batches: activations are tiny, so throughput climbs steeply with batch until memory or compute saturates."
		/>
	{:else if isAsr}
		<Slider
			label="Concurrent streams (batch)"
			bind:value={config.batchSize}
			min={1}
			max={64}
			step={1}
			info="How many audio streams the GPU transcribes in parallel. Each contributes one {modelOf
				?.asr?.audioWindowSec ?? 30}-second window of audio per inference cycle."
		/>
		<p class="hint">
			Each cycle: encoder over {modelOf?.asr?.audioTokens ?? 1500} audio tokens ({modelOf?.asr
				?.audioWindowSec ?? 30}s of audio) + autoregressive decode of ~{modelOf?.asr
				?.avgTextTokens ?? 128} text tokens.
		</p>
	{:else if isVla}
		<Slider
			label="Concurrent robots (batch)"
			bind:value={config.batchSize}
			min={1}
			max={16}
			step={1}
			info="How many robot observations the GPU serves in parallel. Each contributes one observation → one action chunk per inference cycle. On a single robot this stays at 1."
		/>
		<p class="hint">
			Each cycle reads {modelOf?.vla?.camerasPerObs ?? 3} cameras and produces a chunk of
			{modelOf?.vla?.chunkSize ?? 50} actions in
			{modelOf?.vla?.flowSteps ?? 10} flow-matching steps. Target control rate:
			{modelOf?.vla?.controlHz ?? 50} Hz.
		</p>
	{:else}
		<Segmented
			label="Phase"
			bind:value={config.phase}
			options={[
				{ value: 'decode', label: 'Decode' },
				{ value: 'prefill', label: 'Prefill' }
			]}
			info="Which part of the work to measure. 'Prefill' is reading the question — math-heavy, and it sets the wait for the first word. 'Decode' is writing the answer one word at a time — memory-heavy, and it sets the ongoing speed."
		/>

		<Slider
			label="Batch size"
			bind:value={config.batchSize}
			min={1}
			max={512}
			step={1}
			info="How many requests the GPU works on at once. Bigger batches use the GPU more fully and raise total speed, but each user waits a little longer and it needs more memory."
		/>
		<Slider
			label="Input tokens"
			bind:value={config.inputTokens}
			min={64}
			max={1048576}
			log
			display="{fmtSeq(config.inputTokens)} tok"
			info="How long the question (prompt) is, in tokens. Longer inputs take more time to read and use more KV-cache memory. For images or video, count their patches here too — a token is a token to the model."
		/>
		<Slider
			label="Output tokens"
			bind:value={config.outputTokens}
			min={64}
			max={65536}
			log
			display="{fmtSeq(config.outputTokens)} tok"
			info="How long the answer is, in tokens. The model writes output one word at a time, so more output means a longer total wait for the user. Reasoning models (R1, QwQ, o-series) emit long chains of thought — 32k–64k output is normal."
		/>
		<p class="hint tight">
			Total context = input + output = <span class="num dim"
				>{fmtSeq(config.inputTokens + config.outputTokens)} tok</span
			>{#if isVlm}
				+ {config.imagesPerRequest} image{config.imagesPerRequest === 1 ? '' : 's'} × {modelOf?.vlm
					?.tokensPerImage} visual patches{/if}. For multimodal models, count image/video patch
			tokens in <em>input</em>: a token is a token to the transformer.
		</p>

		{#if isVlm}
			<Slider
				label="Images per request"
				bind:value={config.imagesPerRequest}
				min={0}
				max={8}
				step={1}
				info="How many images each request carries. Each image adds {modelOf?.vlm
					?.tokensPerImage} visual patch tokens to the prompt at the model's default resolution — so 1 image at 448² costs ~256 extra input tokens, driving up prefill FLOPs and KV memory."
			/>
		{/if}

		<Slider
			label="Cached prefix"
			bind:value={config.cachedPrefixFrac}
			min={0}
			max={0.99}
			step={0.01}
			display="{Math.round((config.cachedPrefixFrac ?? 0) * 100)}%"
			info="Share of the prompt that's an already-cached shared prefix — a system prompt, RAG boilerplate, or chat history seen before. Its KV is reused, so prefill only processes the uncached rest, cutting the wait for the first word (TTFT). Chunked prefill is the complementary scheduler trick: it splits a long prefill into chunks interleaved with decode so other users' generations don't stall. Memory is unchanged — the prefix KV stays resident."
		/>

		{#if config.phase === 'decode'}
			<div class="tile stack-s">
				<Toggle
					label="Speculative decoding"
					bind:checked={config.specDecode}
					hint="draft proposes, target verifies"
					info="A small 'draft' model guesses several next tokens, and the big model checks them all in one pass. Accepted guesses come almost free, so decode gets faster with no quality change. Only helps decode, not the first word."
				/>
				{#if config.specDecode}
					<Slider
						label="Acceptance rate"
						bind:value={config.draftAcceptRate}
						min={0.3}
						max={0.95}
						step={0.05}
						display="{Math.round((config.draftAcceptRate ?? 0.7) * 100)}%"
						info="How often the draft's guesses are correct. Higher means more free tokens per check. Well-matched draft models on easy text reach 70–90%."
					/>
					<Slider
						label="Draft tokens"
						bind:value={config.specTokens}
						min={1}
						max={8}
						step={1}
						info="How many tokens the draft proposes each round. More can help when acceptance is high, but wasted guesses cost draft time when it's low."
					/>
				{/if}
			</div>
		{/if}
	{/if}

	<div>
		<div class="field-label">
			Weight format
			<InfoTip
				text="How precisely each of the model's numbers is stored. Fewer bits (like FP8 or FP4) shrink the model so it fits in less memory and can run faster, with a small drop in accuracy. More bits (FP16/BF16) are the most accurate but largest."
			/>
		</div>
		<select
			aria-label="Weight format"
			bind:value={config.weightFormatId}
			class="input"
		>
			{#each WEIGHT_FORMATS as f (f.id)}
				<option value={f.id} disabled={!!gpuOf && !gpuSupportsFormat(gpuOf, f)}
					>{f.label} · {f.bitsPerWeight} bit{gpuOf && !gpuSupportsFormat(gpuOf, f)
						? ' — not on this GPU'
						: ''}</option
				>
			{/each}
		</select>
	</div>
	{#if !isDiffusion && !isJepa && !isEncoder}
		<Segmented
			label="KV cache precision"
			bind:value={config.kvBits}
			options={[
				{ value: 16, label: 'FP16' },
				{ value: 8, label: 'FP8' }
			]}
			info="How precisely the conversation notes (KV cache) are stored. FP8 uses half the memory of FP16, so you can fit longer chats or more users at once, with a tiny quality cost."
		/>
		<Segmented
			label="KV allocation"
			bind:value={config.kvAllocation}
			options={[
				{ value: 'paged', label: 'Paged' },
				{ value: 'contiguous', label: 'Contiguous' }
			]}
			info="How KV-cache memory is handed out. 'Paged' (like vLLM) gives it out in small blocks with little waste, so more fits. 'Contiguous' reserves one big chunk per request and wastes more."
		/>
	{/if}
</div>
