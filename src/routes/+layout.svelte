<script lang="ts">
	import './layout.css';
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import { AppShell, Tag } from '@cyberpunk-apps/neondeck';
	import { PRICE_AS_OF } from '$lib/cost/pricing';

	let { children } = $props();
	// path relative to the deploy base (base may be '' or '/tools/genai-calculator')
	const rel = $derived(page.url.pathname.slice(base.length).replace(/\/$/, '') || '/');
	const tabs = [
		{ href: '/', label: 'Workload' },
		{ href: '/modelling', label: 'Modelling' },
		{ href: '/economics', label: 'Self-host vs API' },
		{ href: '/training', label: 'Training' },
		{ href: '/math', label: 'Math' }
	];
	const nav = $derived(tabs.map((t) => ({ label: t.label, href: `${base}${t.href}`, active: rel === t.href })));
</script>

<AppShell brand="GENAI_CALC_" home="{base}/" {nav} railCaption="GENAI CALCULATOR // V0.1">
	{#snippet status()}
		<span><Tag tone="success" dot>runs in your browser</Tag></span>
		<span>prices as of {PRICE_AS_OF}</span>
		<span style="margin-left:auto">first-order estimates // not a benchmark</span>
	{/snippet}
	{@render children()}
</AppShell>
