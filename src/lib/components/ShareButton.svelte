<script lang="ts">
	import { shareUrl } from '$lib/share/url';

	interface Props {
		// the tab's config object to encode into the link
		payload: object;
	}
	let { payload }: Props = $props();

	let copied = $state(false);
	let failed = $state(false);

	async function copy() {
		failed = false;
		const url = shareUrl(location.origin, location.pathname, payload);
		try {
			await navigator.clipboard.writeText(url);
			copied = true;
			setTimeout(() => (copied = false), 1800);
		} catch {
			// clipboard blocked (e.g. insecure context) — drop the link into the URL bar instead
			try {
				history.replaceState(null, '', url);
				copied = true;
				setTimeout(() => (copied = false), 1800);
			} catch {
				failed = true;
			}
		}
	}
</script>

<button
	type="button"
	onclick={copy}
	class="inline-flex items-center gap-1.5 rounded-lg border border-slate-600 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800"
	title="Copy a link that reopens this exact configuration"
>
	{#if copied}
		<span class="text-emerald-400">✓</span> Link copied
	{:else if failed}
		<span class="text-red-400">✗</span> Couldn't copy
	{:else}
		Share this setup
	{/if}
</button>
