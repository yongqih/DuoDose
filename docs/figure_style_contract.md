# Figure style contract

All manuscript-facing and supplementary plots are generated after calling
`duodose.plotting_style.apply_manuscript_style()`.

The shared style requests Arial first, with Liberation Sans and DejaVu Sans as
fallbacks only when Arial is unavailable. It applies to all Matplotlib text,
including titles, labels, ticks, legends, colorbars, annotations, footers, and
rendered table text. PDF and PostScript output use font type 42 so text remains
TrueType-compatible rather than Type 3 glyphs. Saved figures use an opaque
white background and tight bounding boxes.

Formal plotting modules must not set a conflicting local font family. The
generated `results/final_v1/figure_style_contract_audit.csv` records shared
style adoption, conflicting overrides, and Type 3 inspection where PDF output
is available.
