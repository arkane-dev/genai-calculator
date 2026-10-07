<script lang="ts">
	import './layout.css';
	import { base } from '$app/paths';
	import { page } from '$app/state';

	let { children } = $props();
	// path relative to the deploy base (base may be '' or '/tools/genai-calculator')
	const rel = $derived(page.url.pathname.slice(base.length).replace(/\/$/, '') || '/');
	const linkCls = (active: boolean) =>
		`rounded-md px-3 py-1.5 text-sm ${active ? 'bg-teal-500 font-medium text-slate-900' : 'text-slate-300 hover:bg-slate-800'}`;
	const tabs = [
		{ href: '/', label: 'Workload' },
		{ href: '/modelling', label: 'Modelling' },
		{ href: '/economics', label: 'Self-host vs API' },
		{ href: '/training', label: 'Training' },
		{ href: '/math', label: 'Math' }
	];
</script>

<nav class="flex items-center gap-2 border-b border-slate-800 bg-slate-950 px-6 py-3">
	<span class="mr-2 text-sm font-semibold text-slate-100">GenAI Calculator</span>
	{#each tabs as t (t.href)}
		<a href="{base}{t.href}" class={linkCls(rel === t.href)}>{t.label}</a>
	{/each}
</nav>
{@render children()}
