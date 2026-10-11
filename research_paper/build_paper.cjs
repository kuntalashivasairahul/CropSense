// Render the manuscript source with the supplied template's A4/two-column conventions.
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, ExternalHyperlink, AlignmentType, HeadingLevel, WidthType,
  BorderStyle, SectionType, TableLayoutType } = require('docx');

const root = __dirname;
const source = fs.readFileSync(path.join(root, 'CropSense_AI_Paper.md'), 'utf8');
const refs = JSON.parse(fs.readFileSync(path.join(root, 'references.json'), 'utf8'));
assert(refs.length >= 25);
assert((source.match(/^## [IVX]+\./gm) || []).length === 8);
assert((source.match(/^!\[/gm) || []).length === 5);
assert((source.match(/^TABLE [IVX]+\./gm) || []).length === 4);
const bodyText = source.split('## Abstract\n\n')[1];
assert(bodyText.split('\n\nKeywords:')[0].split(/\s+/).length >= 200);
const cited = new Set([...source.split('## References')[0].matchAll(/\[(\d+)\]/g)].map(m => +m[1]));
assert(refs.every(r => cited.has(r.id)));

function runs(text, size = 20) {
  return text.split(/(\[[^\]]+\]\(https?:\/\/[^)]+\))/g).filter(Boolean).map(part => {
    const link = part.match(/^\[([^\]]+)\]\((https?:\/\/[^)]+)\)$/);
    return link ? new ExternalHyperlink({ link: link[2], children: [new TextRun({ text: link[1], size, color: '235785' })] })
      : new TextRun({ text: part, size });
  });
}

const noBorder = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
const borders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder, insideHorizontal: noBorder, insideVertical: noBorder };
const authorLines = source.split('\n').filter(line => line.includes('@anurag.edu.in'));
assert(authorLines.length === 4);
const authorRows = [authorLines.slice(0, 2), authorLines.slice(2)].map(names => new TableRow({
  children: names.map(line => {
    const [name, email] = line.split(' | ');
    return new TableCell({ width: { size: 5060, type: WidthType.DXA }, borders,
      margins: { top: 80, bottom: 100, left: 50, right: 50 },
      children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 30 },
        children: [new TextRun({ text: name, size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
        children: [new TextRun({ text: email, size: 18 })] })] });
  })
}));
const front = [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: source.split('\n')[0].slice(2), size: 40 })] }),
new Table({ width: { size: 10120, type: WidthType.DXA }, columnWidths: [5060, 5060],
  layout: TableLayoutType.FIXED, rows: authorRows, borders }),
new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 220 },
  children: [new TextRun({ text: 'School of Engineering, Anurag University, Hyderabad, India', size: 20, italics: true })] })];

const body = [];
const lines = ('## Abstract\n\n' + bodyText).split('\n');
let inReferences = false;
for (let i = 0; i < lines.length; i++) {
  const line = lines[i].trim();
  if (!line || line.startsWith('<!--')) continue;
  if (line.startsWith('## ')) {
    const text = line.slice(3);
    inReferences = text === 'References';
    body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER,
      keepNext: true, spacing: { before: 150, after: 80 },
      children: [new TextRun({ text: text.toUpperCase(), size: 20, bold: true, color: '000000' })] }));
  } else if (line.startsWith('### ')) {
    body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true,
      spacing: { before: 110, after: 60 }, children: [new TextRun({ text: line.slice(4), size: 20, italics: true, color: '000000' })] }));
  } else if (line.startsWith('![')) {
    const match = line.match(/^!\[(.+)\]\((.+)\)$/);
    assert(match);
    const data = fs.readFileSync(path.join(root, match[2]));
    const width = data.readUInt32BE(16), height = data.readUInt32BE(20);
    body.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true,
      spacing: { before: 100, after: 20 }, children: [new ImageRun({ type: 'png', data,
        transformation: { width: 324, height: Math.round(324 * height / width) },
        altText: { title: match[1].split('.')[0], description: match[1], name: match[2] } })] }));
    body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
      children: runs(match[1], 16) }));
  } else if (line.startsWith('TABLE ')) {
    body.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true,
      spacing: { before: 100, after: 60 }, children: [new TextRun({ text: line, size: 16, bold: true })] }));
  } else if (line.startsWith('|')) {
    const table = [];
    while (i < lines.length && lines[i].trim().startsWith('|')) {
      const cells = lines[i].trim().split('|').slice(1, -1).map(s => s.trim());
      if (!cells.every(c => /^:?-+:?$/.test(c))) table.push(cells);
      i++;
    }
    i--;
    const n = table[0].length;
    const widths = table[0][0] === 'Data path' ? [1200, 1000, 1000, 1680]
      : n === 4 ? [1890, 600, 1150, 1240] : [1330, 1390, 2160];
    assert(widths.length === n && widths.reduce((a, b) => a + b, 0) === 4880);
    body.push(new Table({ width: { size: 4880, type: WidthType.DXA }, columnWidths: widths,
      layout: TableLayoutType.FIXED, rows: table.map((row, r) => new TableRow({
        cantSplit: true, tableHeader: r === 0,
        children: row.map((text, c) => new TableCell({ width: { size: widths[c], type: WidthType.DXA },
          margins: { top: 45, bottom: 45, left: 50, right: 50 },
          children: [new Paragraph({ spacing: { after: 0 },
            children: [new TextRun({ text, size: 16, bold: r === 0 })] })] })) })) }));
    body.push(new Paragraph({ spacing: { after: 70 }, children: [] }));
  } else {
    const equation = /\(\d\)$/.test(line);
    body.push(new Paragraph({ alignment: equation ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
      spacing: { after: inReferences ? 60 : 80 }, widowControl: true,
      children: runs(line.replace(/ {2,}/g, ' '), inReferences || equation ? 16 : 20) }));
  }
}

const page = { size: { width: 11906, height: 16838 },
  margin: { top: 1080, right: 893, bottom: 1440, left: 893, header: 360, footer: 360 } };
const doc = new Document({ creator: 'CropSense AI project team', title: source.split('\n')[0].slice(2),
  description: 'Research manuscript draft for team and guide review.',
  styles: { default: { document: { run: { font: 'Times New Roman', size: 20, color: '000000' },
    paragraph: { spacing: { line: 240, after: 80 }, widowControl: true } } } },
  sections: [{ properties: { page }, children: front },
    { properties: { type: SectionType.CONTINUOUS, page, column: { count: 2, space: 360, equalWidth: true } }, children: body }] });
Packer.toBuffer(doc).then(buffer => {
  const destination = path.join(root, 'CropSense_AI_Paper_Draft.docx');
  fs.writeFileSync(destination, buffer);
  console.log('Created ' + destination);
});
