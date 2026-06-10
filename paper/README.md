# Compiling the Paper

## Requirements

- TeX distribution: TeX Live 2023+ or MiKTeX
- Required packages: IEEEtran, booktabs, subcaption, hyperref, algorithm, algorithmic, multirow

## Figures needed (create and save to paper/figs/)

The paper references these figures — generate them from your results:

| File                    | Source                                       |
| ----------------------- | -------------------------------------------- |
| `figs/architecture.pdf` | Architecture diagram                         |
| `figs/asbc_weights.pdf` | `utils/visualize.py` → `plot_asbc_weights()` |
| `figs/qualitative.pdf`  | Side-by-side inpainting comparisons          |
| `figs/sensitivity.pdf`  | `utils/visualize.py` → `plot_sensitivity()`  |

## Build commands

```
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Page budget (DICTA — 6 pages + 1 references)

Current structure:

- Abstract + intro: ~1.0 page
- Related work: ~0.7 page
- Method: ~1.5 pages
- Experiments: ~2.5 pages
- Discussion + conclusion: ~0.5 page
- References: ~0.5–1.0 page

Total: ~6.7 pages before figures.
Figures (arch, qualitative, asbc, sensitivity) will fill the remaining space.
Trim discussion or collapse related work if over 8 pages.
