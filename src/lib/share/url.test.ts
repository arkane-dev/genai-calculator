import { describe, expect, it } from 'vitest';
import { applyShared, decodeState, encodeState, readSharedState, shareUrl } from './url';

describe('share URL encode/decode', () => {
	it('round-trips a config object', () => {
		const cfg = { modelId: 'llama31-8b', numGpus: 16, ratio: 0.74, on: true };
		const decoded = decodeState(encodeState(cfg));
		expect(decoded).toEqual(cfg);
	});

	it('produces a URL-safe param (no +, /, =)', () => {
		const enc = encodeState({ a: 'x'.repeat(50), b: 3 });
		expect(enc).not.toMatch(/[+/=]/);
	});

	it('returns null for absent or malformed params', () => {
		expect(decodeState(null)).toBeNull();
		expect(decodeState('')).toBeNull();
		expect(decodeState('!!!not-base64!!!')).toBeNull();
		expect(decodeState(encodeState([1, 2, 3] as unknown as Record<string, unknown>))).toBeNull(); // arrays rejected
	});

	it('builds a shareable URL and reads it back', () => {
		const url = shareUrl('https://x.dev', '/workload', { gpuId: 'h100-sxm', numGpus: 8 });
		const search = new URL(url).search;
		expect(readSharedState(search)).toEqual({ gpuId: 'h100-sxm', numGpus: 8 });
	});
});

describe('applyShared', () => {
	it('overlays only existing primitive keys, ignoring unknown/typemismatched ones', () => {
		const target = { modelId: 'a', numGpus: 8, tp: 1 };
		applyShared(target, { modelId: 'b', numGpus: 32, bogus: 'x', tp: { nested: true } });
		expect(target).toEqual({ modelId: 'b', numGpus: 32, tp: 1 }); // bogus dropped, tp type-mismatch kept
	});

	it('is a no-op for null shared state', () => {
		const target = { a: 1 };
		applyShared(target, null);
		expect(target).toEqual({ a: 1 });
	});
});
