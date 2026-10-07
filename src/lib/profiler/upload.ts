import type { ModelSpec } from './types';

/* eslint-disable @typescript-eslint/no-explicit-any */

function slug(s: string): string {
	return (
		s
			.toLowerCase()
			.replace(/[^a-z0-9]+/g, '-')
			.replace(/^-|-$/g, '')
			.slice(0, 40) || 'uploaded'
	);
}

/** Rough parameter count from architecture, when the file doesn't state it. */
function estimateParams(m: Omit<ModelSpec, 'params' | 'id' | 'name'>): number {
	const embedAndHead = 2 * m.vocabSize * m.hiddenSize; // input + output embeddings
	const attn = 2 * m.hiddenSize * m.hiddenSize + 2 * m.hiddenSize * m.numKvHeads * m.headDim;
	const ffn = m.moe
		? m.moe.numExperts * 3 * m.hiddenSize * m.intermediateSize
		: 3 * m.hiddenSize * m.intermediateSize;
	return embedAndHead + m.numLayers * (attn + ffn);
}

/** Accept either our ModelSpec JSON or a raw HuggingFace config.json. Throws a
 * helpful message if neither. Params are estimated when not provided. */
export function parseModelUpload(obj: any): ModelSpec {
	// our ModelSpec shape (camelCase)
	if (typeof obj?.hiddenSize === 'number' && typeof obj?.numLayers === 'number') {
		for (const k of ['numHeads', 'numKvHeads', 'headDim', 'intermediateSize', 'vocabSize']) {
			if (typeof obj[k] !== 'number') throw new Error(`missing numeric field: ${k}`);
		}
		const id = obj.id || slug(obj.name || 'uploaded');
		const spec: ModelSpec = {
			id,
			name: obj.name || id,
			params: obj.params,
			hiddenSize: obj.hiddenSize,
			numLayers: obj.numLayers,
			numHeads: obj.numHeads,
			numKvHeads: obj.numKvHeads,
			headDim: obj.headDim,
			intermediateSize: obj.intermediateSize,
			vocabSize: obj.vocabSize,
			moe: obj.moe,
			mla: obj.mla,
			activeParams: obj.activeParams,
			kvLayers: obj.kvLayers
		};
		if (typeof spec.params !== 'number') spec.params = estimateParams(spec);
		return spec;
	}

	// Multimodal HF configs wrap the language model in `text_config`.
	if (obj?.text_config && typeof obj.text_config === 'object') {
		obj = { ...obj.text_config, _name_or_path: obj._name_or_path ?? obj.model_type };
	}

	// HuggingFace config.json (snake_case)
	if (typeof obj?.hidden_size === 'number') {
		const heads = obj.num_attention_heads;
		const kvHeads = obj.num_key_value_heads || heads;
		// Some configs (multimodal, MLA) set head_dim = 0; derive from hidden_size.
		const headDim = obj.head_dim > 0 ? obj.head_dim : Math.round(obj.hidden_size / heads);
		const numExperts = obj.n_routed_experts ?? obj.num_experts;
		const moe = numExperts
			? {
					numExperts,
					expertsPerToken: obj.num_experts_per_tok ?? obj.num_experts_per_token ?? 2
				}
			: undefined;
		// MLA: kv_lora_rank + qk_rope_head_dim (0 in some configs) gives the latent size.
		const mla = obj.kv_lora_rank
			? { kvLatentDim: obj.kv_lora_rank + (obj.qk_rope_head_dim || 0) }
			: undefined;
		const name = obj._name_or_path || obj.model_type || 'uploaded model';
		const base = {
			hiddenSize: obj.hidden_size,
			numLayers: obj.num_hidden_layers,
			numHeads: heads,
			numKvHeads: kvHeads,
			headDim,
			intermediateSize: obj.moe_intermediate_size ?? obj.intermediate_size,
			vocabSize: obj.vocab_size,
			moe,
			mla
		};
		const params = obj.num_parameters ?? estimateParams(base as any);
		return {
			id: slug(name),
			name,
			params,
			activeParams: obj.num_activated_parameters,
			...base
		} as ModelSpec;
	}

	throw new Error(
		'Unrecognized JSON. Provide a ModelSpec (params, hiddenSize, numLayers, numHeads, numKvHeads, headDim, intermediateSize, vocabSize) or a HuggingFace config.json.'
	);
}
