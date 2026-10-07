<script lang="ts">
	// A small "i" button that opens a plain-language explanation. The popover is
	// position:fixed and measured from the button, so it floats above every panel
	// (no stacking-context clipping) and flips to stay on-screen. Click to toggle;
	// click anywhere else, scroll, or resize closes it.
	interface Props {
		text: string;
		label?: string;
	}
	let { text, label = 'What is this?' }: Props = $props();

	let open = $state(false);
	let btn: HTMLButtonElement;
	let pop = $state<HTMLElement | null>(null);
	let pos = $state({ top: 0, left: 0 });

	function place() {
		if (!btn) return;
		const r = btn.getBoundingClientRect();
		const pw = pop?.offsetWidth ?? 240;
		const ph = pop?.offsetHeight ?? 120;
		let left = r.left;
		if (left + pw > window.innerWidth - 8) left = window.innerWidth - pw - 8;
		if (left < 8) left = 8;
		let top = r.bottom + 6;
		if (top + ph > window.innerHeight - 8) top = Math.max(8, r.top - 6 - ph);
		pos = { top, left };
	}

	// Once the popover is in the DOM, measure it and position within the viewport.
	$effect(() => {
		if (open && pop) place();
	});

	// Close on outside scroll/resize so it never drifts away from its button.
	$effect(() => {
		if (!open) return;
		const close = () => (open = false);
		window.addEventListener('scroll', close, true);
		window.addEventListener('resize', close);
		return () => {
			window.removeEventListener('scroll', close, true);
			window.removeEventListener('resize', close);
		};
	});

	// Move the popover to <body> so no ancestor stacking context (opacity,
	// isolation, transform, …) can trap its z-index or clip it behind a panel.
	function portal(node: HTMLElement) {
		document.body.appendChild(node);
		return { destroy: () => node.remove() };
	}
</script>

<button
	bind:this={btn}
	type="button"
	aria-label={label}
	aria-expanded={open}
	onclick={(e) => {
		e.stopPropagation();
		e.preventDefault();
		open = !open;
	}}
	class="inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-slate-600 align-middle text-[10px] leading-none font-semibold text-slate-400 hover:border-teal-400 hover:text-teal-300"
>
	i
</button>
{#if open}
	<!-- invisible backdrop closes the popover on any outside click -->
	<button
		use:portal
		type="button"
		aria-hidden="true"
		tabindex="-1"
		class="fixed inset-0 z-[59] cursor-default"
		onclick={(e) => {
			e.stopPropagation();
			e.preventDefault();
			open = false;
		}}
	></button>
	<span
		use:portal
		bind:this={pop}
		role="tooltip"
		style:top="{pos.top}px"
		style:left="{pos.left}px"
		class="fixed z-[60] w-60 rounded-lg border border-slate-600 bg-slate-800 p-3 text-xs leading-relaxed font-normal tracking-normal text-slate-200 normal-case shadow-xl"
	>
		{text}
	</span>
{/if}
