<script lang="ts">
	import { reportToMarkdown, type SizingReport } from '$lib/report/markdown';

	interface Props {
		// a function so the report is built from the latest derived state at click time
		report: () => SizingReport;
		filename?: string; // without extension
	}
	let { report, filename = 'sizing-report' }: Props = $props();

	let done = $state(false);

	function download() {
		const md = reportToMarkdown(report());
		const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `${filename}.md`;
		document.body.appendChild(a);
		a.click();
		a.remove();
		setTimeout(() => URL.revokeObjectURL(url), 0);
		done = true;
		setTimeout(() => (done = false), 1800);
	}
</script>

<button
	type="button"
	onclick={download}
	class="inline-flex items-center gap-1.5 rounded-lg border border-slate-600 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800"
	title="Download this scenario as a Markdown summary"
>
	{#if done}
		<span class="text-emerald-400">✓</span> Report saved
	{:else}
		⬇ Export report
	{/if}
</button>
