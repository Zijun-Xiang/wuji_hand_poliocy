// Pre-render KaTeX math and inline the SVG figures: node render.js note.src.html note.html
const fs = require('fs');
const katex = require('katex');
const [,, inPath, outPath] = process.argv;
let text = fs.readFileSync(inPath, 'utf8');
const macros = {
  "\\hl": "\\textcolor{dc2626}{#1}",
  "\\vd": "\\substack{\\cdot\\\\\\cdot\\\\\\cdot}",
  "\\bq": "\\mathbf{q}", "\\bp": "\\mathbf{p}", "\\bff": "\\mathbf{f}", "\\bx": "\\mathbf{x}",
  "\\bz": "\\mathbf{z}", "\\bn": "\\hat{\\mathbf{n}}", "\\br": "\\mathbf{r}", "\\rb": "\\bar{\\mathbf{r}}",
  "\\ba": "\\mathbf{a}", "\\be": "\\mathbf{e}", "\\bo": "\\mathbf{o}", "\\bl": "\\mathbf{l}",
  "\\brho": "\\boldsymbol{\\rho}", "\\bom": "\\boldsymbol{\\omega}", "\\dq": "\\dot{\\mathbf{q}}",
  "\\clip": "\\operatorname{clip}",
};
const decode = s => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
const errors = [];
const render = (tex, display) => {
  try {
    return katex.renderToString(decode(tex), { displayMode: display, throwOnError: true, macros, strict: false, trust: true });
  } catch (e) { errors.push(String(e.message).slice(0, 160) + '  <<' + tex.slice(0, 80)); return '<span style="color:red">MATH ERROR</span>'; }
};
const pres = [];
text = text.replace(/<pre[\s\S]*?<\/pre>/g, m => { pres.push(m); return `@@PRE${pres.length - 1}@@`; });
text = text.replace(/\$\$([\s\S]+?)\$\$/g, (m, tex) => render(tex, true));
text = text.replace(/\$([^\$\n]+?)\$/g, (m, tex) => render(tex, false));
text = text.replace(/@@PRE(\d+)@@/g, (m, i) => pres[+i]);
text = text.replace(/\{\{FIG_([A-Z]+)\}\}/g, (m, name) => fs.readFileSync(`fig_${name.toLowerCase()}.svg`, 'utf8'));
fs.writeFileSync(outPath, text);
if (errors.length) { console.error(errors.join('\n')); process.exit(1); }
console.log('rendered OK');
