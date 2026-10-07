<script lang="ts">
	// Render a LaTeX string with KaTeX (offline, bundled — no network). Errors fall back to
	// the raw source rather than throwing, so a bad formula never blanks the page.
	import katex from 'katex';
	import 'katex/dist/katex.min.css';

	interface Props {
		math: string;
		display?: boolean; // block (centered, larger) vs inline
	}
	let { math, display = false }: Props = $props();

	const html = $derived.by(() => {
		try {
			return katex.renderToString(math, {
				displayMode: display,
				throwOnError: false,
				output: 'html'
			});
		} catch {
			return math;
		}
	});
</script>

<!-- eslint-disable-next-line svelte/no-at-html-tags -- KaTeX output is trusted (our own formulas) -->
<span class="katex-host">{@html html}</span>
