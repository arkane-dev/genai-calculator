// Shareable scenarios: encode a tab's config into a compact URL param so a sizing
// can be sent to a colleague/customer and reopened exactly. Client-side only
// (static app); the param travels in the querystring under `c`.

const PARAM = 'c';

/** base64url (URL-safe, no padding) so the param survives a querystring cleanly. */
function toB64Url(s: string): string {
	const b64 = btoa(unescape(encodeURIComponent(s)));
	return b64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
function fromB64Url(s: string): string {
	const b64 = s.replace(/-/g, '+').replace(/_/g, '/');
	return decodeURIComponent(escape(atob(b64)));
}

/** Encode a plain config object to a URL-safe string. */
export function encodeState(state: object): string {
	return toB64Url(JSON.stringify(state));
}

/** Decode a URL param back to an object, or null if absent/malformed. */
export function decodeState(param: string | null | undefined): Record<string, unknown> | null {
	if (!param) return null;
	try {
		const obj = JSON.parse(fromB64Url(param));
		return obj && typeof obj === 'object' && !Array.isArray(obj) ? obj : null;
	} catch {
		return null;
	}
}

/** Full shareable URL for the current page + a config object. */
export function shareUrl(origin: string, pathname: string, state: object): string {
	return `${origin}${pathname}?${PARAM}=${encodeState(state)}`;
}

/** Read the shared config from a URL's search params (returns null if none). */
export function readSharedState(search: string | URLSearchParams): Record<string, unknown> | null {
	const params = typeof search === 'string' ? new URLSearchParams(search) : search;
	return decodeState(params.get(PARAM));
}

/** Overlay decoded values onto a config, but ONLY keys that already exist on it
 * (and match primitive type), so a hand-tampered link can't inject junk. */
export function applyShared<T extends object>(
	target: T,
	shared: Record<string, unknown> | null
): void {
	if (!shared) return;
	const t = target as Record<string, unknown>;
	for (const k of Object.keys(t)) {
		if (!(k in shared)) continue;
		const cur = t[k];
		const next = shared[k];
		if (typeof cur === typeof next && typeof cur !== 'object') {
			t[k] = next;
		}
	}
}
