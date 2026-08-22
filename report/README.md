# Build the report on Ubuntu 22.04

Install a LaTeX distribution if `pdflatex` and `bibtex` are unavailable:

```bash
sudo apt update
sudo apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

Then compile:

```bash
cd report
pdflatex neurips_2026.tex
bibtex neurips_2026
pdflatex neurips_2026.tex
pdflatex neurips_2026.tex
```

The expected output is `report/neurips_2026.pdf`. Before submission, both members must verify the contribution statement, submit the separately produced demonstration video through the required course workflow, and ensure the main paper remains within the assignment's eight-page limit excluding references and appendix.
