import { describe, expect, it } from 'vitest';
import { estimateLoad } from './estimate';

const weights70bBf16 = 140e9; // 70B × 2 bytes = 140 GB

describe('estimateLoad', () => {
	it('is bounded by PCIe when the source is fast (host RAM cache on 1 GPU → PCIe-bound)', () => {
		const r = estimateLoad({
			weightBytes: weights70bBf16,
			numGpus: 1,
			pcieGBs: 63,
			sourceId: 'host-ram',
			sharded: false,
			shardWaste: 1
		});
		expect(r.bottleneck).toBe('pcie');
		expect(r.seconds).toBeGreaterThan(0);
	});

	it('is bounded by the source when it is slow (single-stream object storage)', () => {
		const r = estimateLoad({
			weightBytes: weights70bBf16,
			numGpus: 8,
			pcieGBs: 63,
			sourceId: 'object-single',
			sharded: false,
			shardWaste: 1
		});
		expect(r.bottleneck).toBe('source');
	});

	it('Gen5 loads faster than Gen4 when PCIe-bound', () => {
		const g4 = estimateLoad({
			weightBytes: weights70bBf16,
			numGpus: 1,
			pcieGBs: 32,
			sourceId: 'host-ram',
			sharded: false,
			shardWaste: 1
		});
		const g5 = estimateLoad({
			weightBytes: weights70bBf16,
			numGpus: 1,
			pcieGBs: 63,
			sourceId: 'host-ram',
			sharded: false,
			shardWaste: 1
		});
		expect(g5.seconds).toBeLessThan(g4.seconds);
	});

	it('sharded loads (FSDP) split weights across GPUs', () => {
		const dp = estimateLoad({
			weightBytes: weights70bBf16,
			numGpus: 8,
			pcieGBs: 63,
			sourceId: 'host-ram',
			sharded: false,
			shardWaste: 1
		});
		const fsdp = estimateLoad({
			weightBytes: weights70bBf16,
			numGpus: 8,
			pcieGBs: 63,
			sourceId: 'host-ram',
			sharded: true,
			shardWaste: 1
		});
		expect(fsdp.perGpuBytes).toBeLessThan(dp.perGpuBytes);
		expect(fsdp.seconds).toBeLessThan(dp.seconds);
	});
});
