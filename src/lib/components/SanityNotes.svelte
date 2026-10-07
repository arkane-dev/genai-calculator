<script lang="ts">
	// Soft "heads up" advisories for a valid-but-questionable config. Renders nothing
	// when the list is empty, so pages can drop it in unconditionally.
	let { notes }: { notes: string[] } = $props();
</script>

{#if notes.length}
	<div class="notice">
		<div class="notice-title">Heads up</div>
		<ul class="bullets">
			{#each notes as n (n)}<li>{n}</li>{/each}
		</ul>
	</div>
{/if}
