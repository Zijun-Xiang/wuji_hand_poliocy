# Source of `docs/penspin_graph_step_by_step.pdf`

* `verify.py` re-checks every formula and number in the PDF against the code and `runs/full3/`.
* `figs.py` draws the two flow figures as SVG (text widths are measured, no overlaps).
* `note.src.html` is the document; `render.js` pre-renders the KaTeX math; `build.sh` prints it with headless Chrome and `footer.py` adds page numbers.

Rebuild: `cd docs/pdf_src && npm install katex@0.16 && ./build.sh`
