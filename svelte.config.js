import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),

	compilerOptions: {
		// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
		runes: ({ filename }) => (filename.split(/[/\\]/).includes('node_modules') ? undefined : true)
	},

	kit: {
		// Fully static output. `404.html` is the SPA fallback so any route that
		// isn't prerendered still resolves client-side.
		adapter: adapter({ fallback: '404.html' }),
		// Set BASE_PATH to serve under a subpath, e.g. "/tools/genai-calculator" on
		// andrewrkane.com. Empty works for a root domain and for local dev.
		paths: { base: process.env.BASE_PATH ?? '' }
	}
};

export default config;
