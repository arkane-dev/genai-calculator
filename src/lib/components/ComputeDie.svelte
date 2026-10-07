<script lang="ts">
	interface Unit {
		label: string;
		color: string;
		areaFrac: number; // share of the die; band height is proportional
		util?: number; // 0..1 utilization (undefined => structural)
		isStatic?: boolean; // cache / IO region, never "used"
		// this block is the limiting resource. 'info' = you're bound here (amber, normal);
		// 'alert' = this is why an SLO can't be met (red, failure).
		bottleneck?: 'info' | 'alert';
	}
	interface Props {
		units: Unit[];
		cols?: number;
	}
	let { units, cols = 16 }: Props = $props();

	const litCount = (u: Unit) =>
		u.isStatic ? 0 : Math.round(Math.max(0, Math.min(1, u.util ?? 0)) * cols);
	// marker colour by severity
	const flagColor = (b: Unit['bottleneck']) => (b === 'alert' ? '#ff3b52' : '#f5ec58');
</script>

<div class="die">
	<div class="kv head">
		<span class="title">GPU compute die</span>
		<span class="caption">block size ∝ die area</span>
	</div>

	<div class="blocks">
		{#each units as u (u.label)}
			{@const lit = litCount(u)}
			<div
				class="block"
				style:flex-grow={u.areaFrac}
				style:border-color={u.bottleneck ? flagColor(u.bottleneck) : `${u.color}55`}
				style:border-width={u.bottleneck ? '2px' : '1px'}
				style:background-color={u.bottleneck ? `${flagColor(u.bottleneck)}18` : `${u.color}12`}
			>
				<div class="kv row">
					<span
						class="name"
						class:bad={u.bottleneck === 'alert'}
						class:warn={u.bottleneck === 'info'}
						class:flagged={!!u.bottleneck}
					>
						{u.label}{#if u.bottleneck}<span class="flag">◄ bottleneck</span>{/if}
					</span>
					<span
						class="num"
						class:bad={u.bottleneck === 'alert'}
						class:warn={u.bottleneck === 'info'}
						class:mute={!u.bottleneck && u.isStatic}
						class:bright={!u.bottleneck && !u.isStatic}
					>
						{u.isStatic ? 'structural' : `${Math.round((u.util ?? 0) * 100)}%`}
					</span>
				</div>
				<div class="cells">
					{#each Array(cols) as _, i (i)}
						{@const on = u.isStatic || i < lit}
						<div
							class="cell"
							style:background-color={u.isStatic ? `${u.color}55` : on ? u.color : '#12163a'}
							style:transition-delay="{i * 10}ms"
							style:box-shadow={on && !u.isStatic ? `0 0 4px ${u.color}99` : 'none'}
						></div>
					{/each}
				</div>
			</div>
		{/each}
	</div>
</div>

<style>
	.die {
		display: flex;
		flex-direction: column;
		height: 100%;
		padding: var(--nd-space-3);
		border: 1px solid var(--nd-line-strong);
		background: var(--nd-surface-1);
	}
	.head { margin-bottom: var(--nd-space-2); }
	.title { color: var(--nd-text); font-size: var(--nd-text-sm); font-weight: 500; }
	.caption { color: var(--nd-text-mute); font-size: 0.625rem; }
	.blocks { display: flex; flex: 1; flex-direction: column; gap: 0.375rem; min-height: 22rem; }
	.block {
		display: flex;
		flex-basis: 0;
		flex-direction: column;
		min-height: 2.25rem;
		overflow: hidden;
		border-style: solid;
	}
	.row { padding: var(--nd-space-1) var(--nd-space-2) 0; font-size: 0.6875rem; }
	.name { overflow: hidden; color: var(--nd-text-dim); text-overflow: ellipsis; white-space: nowrap; }
	.name.flagged { font-weight: 500; }
	.flag { margin-left: var(--nd-space-1); }
	.cells { display: flex; flex: 1; gap: 2px; padding: var(--nd-space-1) var(--nd-space-2) 0.375rem; }
	.cell { flex: 1; transition: background-color 300ms var(--nd-ease); }
	@media (prefers-reduced-motion: reduce) { .cell { transition: none; } }
</style>
