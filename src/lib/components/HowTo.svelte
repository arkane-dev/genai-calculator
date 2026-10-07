<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		title?: string;
		open?: boolean;
		children: Snippet;
	}
	let { title = 'How to use this tab', open = false, children }: Props = $props();
</script>

<details class="howto" {open}>
	<summary>
		<span class="title">
			<span class="badge" aria-hidden="true">i</span>
			{title}
		</span>
		<span class="chev" aria-hidden="true">▾</span>
	</summary>
	<div class="body">
		{@render children()}
	</div>
</details>

<style>
	.howto {
		margin-bottom: var(--nd-space-6);
		border: 1px solid var(--nd-line-strong);
		background: color-mix(in srgb, var(--nd-surface-1) 60%, transparent);
	}
	summary {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--nd-space-2);
		padding: var(--nd-space-3) var(--nd-space-5);
		color: var(--nd-text);
		font-size: var(--nd-text-sm);
		font-weight: 500;
		list-style: none;
		cursor: pointer;
		user-select: none;
	}
	summary::-webkit-details-marker { display: none; }
	.title { display: flex; align-items: center; gap: var(--nd-space-2); }
	.badge {
		display: inline-grid;
		place-items: center;
		width: 1rem;
		height: 1rem;
		border: 1px solid currentColor;
		font-family: var(--nd-font-mono);
		font-size: 0.625rem;
	}
	.chev { color: var(--nd-text-mute); transition: transform var(--nd-dur-fast) var(--nd-ease); }
	.howto[open] .chev { transform: rotate(180deg); }
	.body {
		padding: var(--nd-space-4) var(--nd-space-5);
		border-top: 1px solid var(--nd-surface-2);
		color: var(--nd-text-dim);
		font-size: var(--nd-text-sm);
		line-height: 1.6;
	}
	.body :global(p) { max-width: none; }
	.body :global(a) { color: var(--nd-accent); }
	@media (prefers-reduced-motion: reduce) { .chev { transition: none; } }
</style>
