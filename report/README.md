# Report skeleton

`main.tex` is an IEEEtran conference skeleton. It holds the five section headings (Introduction, Related Work, Methods, Results, Conclusion) plus the abstract, float shells with empty captions, and hooks for the generated tables (`\gentable`) and figures (`\genfig`), and nothing else. The students write all the text, including the abstract and every caption. The generated `.tex` tables are bare `tabular`s; their only text is a LaTeX comment, which doesn't print.

- Regenerate the tables and figures with `make train`, `make agree` and `make figures`.
- Build locally with `latexmk -pdf main.tex` from this folder. On Overleaf, upload `report/` and `results/` side by side so the `../results/...` paths still work.
- If an input is missing, a boxed note names the command to run, so the document still compiles.
- The bibliography lines are commented out until `refs.bib` has an entry you `\cite`, because BibTeX stops with an error while there is nothing to cite.
