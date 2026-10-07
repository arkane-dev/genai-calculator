<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		title?: string;
		open?: boolean;
		children: Snippet;
	}
	let { title = 'How to use this tab', open = false, children }: Props = $props();
</script>

<details
	class="group mb-6 rounded-xl border border-slate-700 bg-slate-900/60 [&_a]:text-teal-400 [&_a:hover]:underline"
	{open}
>
	<summary
		class="flex cursor-pointer list-none items-center justify-between gap-2 px-5 py-3 text-sm font-medium text-slate-200 select-none [&::-webkit-details-marker]:hidden"
	>
		<span class="flex items-center gap-2">
			<svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
				<circle cx="8" cy="8" r="7" stroke="currentColor" stroke-width="1.4" />
				<path d="M8 7v4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" />
				<circle cx="8" cy="4.6" r="0.9" fill="currentColor" />
			</svg>
			{title}
		</span>
		<svg
			class="h-4 w-4 text-slate-500 transition-transform group-open:rotate-180"
			viewBox="0 0 16 16"
			fill="none"
			aria-hidden="true"
		>
			<path
				d="M4 6l4 4 4-4"
				stroke="currentColor"
				stroke-width="1.6"
				stroke-linecap="round"
				stroke-linejoin="round"
			/>
		</svg>
	</summary>
	<div class="border-t border-slate-800 px-5 py-4 text-sm leading-relaxed text-slate-400">
		{@render children()}
	</div>
</details>
