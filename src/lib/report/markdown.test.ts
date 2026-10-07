import { describe, expect, it } from 'vitest';
import { reportToMarkdown, type SizingReport } from './markdown';

const base: SizingReport = {
	title: 'GPU sizing — Llama 3.1 70B on H100',
	subtitle: '4× H100 SXM · TP4·PP1',
	generatedAt: '2026-09-07',
	sections: [
		{
			heading: 'Configuration',
			rows: [
				['Model', 'Llama 3.1 70B'],
				['GPUs', '4']
			]
		},
		{ heading: 'Results', rows: [['Fits', 'Yes']], note: 'First-order estimate.' }
	],
	disclaimer: 'Estimates only. Get a real quote.'
};

describe('reportToMarkdown', () => {
	it('renders title, subtitle and generated line', () => {
		const md = reportToMarkdown(base);
		expect(md).toContain('# GPU sizing — Llama 3.1 70B on H100');
		expect(md).toContain('4× H100 SXM · TP4·PP1');
		expect(md).toContain('_Generated 2026-09-07_');
	});

	it('renders each section as a two-column table', () => {
		const md = reportToMarkdown(base);
		expect(md).toContain('## Configuration');
		expect(md).toContain('| Field | Value |');
		expect(md).toContain('| Model | Llama 3.1 70B |');
		expect(md).toContain('| GPUs | 4 |');
	});

	it('renders a section note and the disclaimer as a blockquote', () => {
		const md = reportToMarkdown(base);
		expect(md).toContain('_First-order estimate._');
		expect(md).toContain('> Estimates only. Get a real quote.');
	});

	it('escapes pipes and flattens newlines in cells', () => {
		const md = reportToMarkdown({
			...base,
			sections: [{ heading: 'X', rows: [['a|b', 'line1\nline2']] }]
		});
		expect(md).toContain('| a\\|b | line1 line2 |');
	});
});
