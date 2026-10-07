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
	class="tip-btn"
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
		class="tip-scrim"
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
		class="tip-pop"
	>
		{text}
	</span>
{/if}

<style>
	/* 24px hit area (WCAG 2.5.8) around a 16px visible box; negative margins keep the layout. */
	.tip-btn {
		position: relative;
		display: inline-flex;
		flex: none;
		align-items: center;
		justify-content: center;
		width: 1.5rem;
		height: 1.5rem;
		margin: -0.25rem;
		padding: 0;
		border: 0;
		background: transparent;
		color: var(--nd-text-dim);
		font-family: var(--nd-font-mono);
		font-size: 0.625rem;
		font-weight: 600;
		line-height: 1;
		vertical-align: middle;
		text-transform: none;
		cursor: pointer;
	}
	.tip-btn::before {
		content: '';
		position: absolute;
		inset: 0.25rem;
		border: 1px solid var(--nd-text-mute);
	}
	.tip-btn:hover::before { border-color: var(--nd-accent); }
	.tip-btn:focus-visible { outline: 2px solid var(--nd-focus); outline-offset: -2px; }
	.tip-btn:hover { color: var(--nd-accent); }
	/* Portalled to <body>, so these must be global. */
	:global(.tip-scrim) {
		position: fixed;
		inset: 0;
		z-index: 59;
		border: 0;
		background: transparent;
		cursor: default;
	}
	:global(.tip-pop) {
		position: fixed;
		z-index: 60;
		width: 15rem;
		padding: var(--nd-space-3);
		border: 1px solid var(--nd-line-strong);
		background: var(--nd-surface-3);
		color: var(--nd-text);
		font-family: var(--nd-font-body);
		font-size: var(--nd-text-xs);
		font-weight: 400;
		line-height: 1.6;
		letter-spacing: 0;
		text-transform: none;
	}
</style>
