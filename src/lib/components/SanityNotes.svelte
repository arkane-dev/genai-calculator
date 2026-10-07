<script lang="ts">
	// Soft "heads up" advisories for a valid-but-questionable config. Renders nothing
	// when the list is empty, so pages can drop it in unconditionally.
	let { notes }: { notes: string[] } = $props();
</script>

{#if notes.length}
	<div class="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-200/90">
		<div class="mb-1 font-medium text-amber-300">Heads up</div>
		<ul class="ml-4 list-disc space-y-1">
			{#each notes as n (n)}<li>{n}</li>{/each}
		</ul>
	</div>
{/if}
