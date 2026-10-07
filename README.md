# GenAI Calculator

Size a GenAI workload and see what it costs. It answers three questions:

- Does this model fit on this GPU?
- What is the best cluster to serve this model at this scale?
- Is it cheaper to self-host, or to pay per token with an API?

Everything runs in your browser. Nothing you enter is sent to a server. It works with any
cloud or your own hardware.

## Tabs

- **Workload:** describe the traffic and the promises you need to keep (latency, throughput).
  It finds the smallest cluster on each GPU that meets them, and recommends one.
- **Modelling:** a first-order roofline calculator. Pick a model, GPU, count, parallelism,
  batch, tokens, weight format and KV settings. It shows whether it fits, how many sequences
  run at once, throughput, time to first token, and the bottleneck. Memory, bandwidth,
  compute and the node topology are drawn live.
- **Self-host vs API:** the break-even point between running your own GPUs and paying an
  API per token.
- **Training:** memory, time and cost for full fine-tuning, LoRA and QLoRA.
- **Math:** every number shown as its calculation, step by step.

## What it models

- Models: dense and mixture-of-experts, latent attention (MLA) and hybrid attention, plus
  diffusion, vision-language, speech, embedding and world models. Upload any Hugging Face
  `config.json` to add a model.
- GPUs: B300, B200, H200, H100, A100, L40S, L4, A10G, T4, RTX 4090 and RTX PRO cards.
- Parallelism: tensor, pipeline, expert and data parallel, with NVLink inside a server and
  InfiniBand, RDMA Ethernet or standard Ethernet between servers.
- Weight formats: BF16/FP16, FP8, MXFP6, NVFP4, MXFP4 and INT4.
- KV cache: GQA and MLA sizing, FP16/FP8, paged or contiguous.
- Cost: $/GPU-hour from the median on-demand market rate, with committed and spot
  discounts, or your own rate.
- Energy and carbon: power at the wall, PUE, and emissions for a grid you pick (or your own
  gCO₂e/kWh), location-based or market-based.

The numbers are first-order estimates, labelled as such in the app. Treat them as sizing
guidance, not a benchmark. Prices are a point-in-time snapshot (October 2026). Every price
in the tool is editable.

## Run it

The look comes from NEONDECK, which installs from a folder beside this repo:

```sh
mkdir cyberpunk_apps && cd cyberpunk_apps
git clone https://github.com/arkane-dev/genai-calculator
git clone https://github.com/arkane-dev/neondeck sharable_assets
(cd sharable_assets/neondeck && npm install && npm run build)
```

Node lives in a project-local environment (uv + nodeenv):

```sh
cd genai-calculator
uv venv .venv && uv pip install --python .venv nodeenv
.venv/bin/nodeenv -p --node=lts
source .venv/bin/activate
npm install
npm run dev          # http://localhost:5173
```

Other scripts:

```sh
npm run build        # static site in build/
npm run preview      # serve the build
npm run check        # svelte-check (types)
npm test             # vitest
npm run lint         # prettier + eslint
```

Set `BASE_PATH` to build for a subpath. The live copy at [andrewrkane.com/tools/genai-calculator](https://andrewrkane.com/tools/genai-calculator/) is built with `BASE_PATH=/tools/genai-calculator` as part of the website's build.
