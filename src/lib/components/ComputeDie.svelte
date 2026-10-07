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
	const flagColor = (b: Unit['bottleneck']) => (b === 'alert' ? '#f87171' : '#fbbf24');
</script>

<div class="flex h-full flex-col rounded-lg border border-slate-600 bg-slate-900 p-3">
	<div class="mb-2 flex items-baseline justify-between">
		<span class="text-sm font-medium text-slate-200">GPU compute die</span>
		<span class="text-[10px] text-slate-500">block size ∝ die area</span>
	</div>

	<div class="flex min-h-[22rem] flex-1 flex-col gap-1.5">
		{#each units as u (u.label)}
			{@const lit = litCount(u)}
			<div
				class="flex min-h-[2.25rem] flex-col overflow-hidden rounded-md border"
				style:flex-grow={u.areaFrac}
				style:flex-basis="0"
				style:border-color={u.bottleneck ? flagColor(u.bottleneck) : `${u.color}55`}
				style:border-width={u.bottleneck ? '2px' : '1px'}
				style:background-color={u.bottleneck ? `${flagColor(u.bottleneck)}18` : `${u.color}12`}
			>
				<div class="flex items-baseline justify-between px-2 pt-1 text-[11px]">
					<span
						class="truncate {u.bottleneck === 'alert'
							? 'font-medium text-red-300'
							: u.bottleneck === 'info'
								? 'font-medium text-amber-300'
								: 'text-slate-300'}"
					>
						{u.label}{#if u.bottleneck}<span
								class="ml-1 {u.bottleneck === 'alert' ? 'text-red-400' : 'text-amber-400'}"
								>◄ bottleneck</span
							>{/if}
					</span>
					<span
						class="font-mono {u.bottleneck === 'alert'
							? 'text-red-300'
							: u.bottleneck === 'info'
								? 'text-amber-300'
								: u.isStatic
									? 'text-slate-500'
									: 'text-slate-200'}"
					>
						{u.isStatic ? 'structural' : `${Math.round((u.util ?? 0) * 100)}%`}
					</span>
				</div>
				<div class="flex flex-1 gap-[2px] px-2 pt-1 pb-1.5">
					{#each Array(cols) as _, i (i)}
						{@const on = u.isStatic || i < lit}
						<div
							class="flex-1 rounded-[2px] transition-colors duration-300"
							style:background-color={u.isStatic ? `${u.color}55` : on ? u.color : '#1e293b'}
							style:transition-delay="{i * 10}ms"
							style:box-shadow={on && !u.isStatic ? `0 0 4px ${u.color}99` : 'none'}
						></div>
					{/each}
				</div>
			</div>
		{/each}
	</div>
</div>
