// Green → red for a 0..1 load fraction (0 = idle, 1 = saturated).
export function heatColor(x: number): string {
	const c = Math.max(0, Math.min(1, x));
	return `hsl(${Math.round(150 * (1 - c))}, 75%, 45%)`;
}
