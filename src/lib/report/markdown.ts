// A sizing scenario, rendered to Markdown for download. Pure + testable so the
// component stays a thin download trigger. Keeps the tool useful as a deliverable:
// size a workload, export it, and share the summary with your team or a provider.

export interface ReportSection {
	heading: string;
	rows: [string, string][]; // [field, value]
	note?: string;
}

export interface SizingReport {
	title: string;
	subtitle?: string;
	generatedAt: string; // ISO or human string
	sections: ReportSection[];
	disclaimer?: string;
}

function escapeCell(s: string): string {
	// keep table cells on one line and don't let a literal pipe break the column
	return s.replace(/\|/g, '\\|').replace(/\r?\n/g, ' ');
}

export function reportToMarkdown(r: SizingReport): string {
	const out: string[] = [];
	out.push(`# ${r.title}`);
	if (r.subtitle) out.push(`\n${r.subtitle}`);
	out.push(`\n_Generated ${r.generatedAt}_`);

	for (const s of r.sections) {
		out.push(`\n## ${s.heading}\n`);
		if (s.rows.length) {
			out.push('| Field | Value |');
			out.push('| --- | --- |');
			for (const [k, v] of s.rows) out.push(`| ${escapeCell(k)} | ${escapeCell(v)} |`);
		}
		if (s.note) out.push(`\n_${s.note}_`);
	}

	if (r.disclaimer) out.push(`\n> ${r.disclaimer.replace(/\r?\n/g, ' ')}`);
	out.push('');
	return out.join('\n');
}
