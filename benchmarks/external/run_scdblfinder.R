#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1) args[[1]] else "results/external_input/dataset"
output_csv <- if (length(args) >= 2) args[[2]] else "scdblfinder_scores.csv"
expected_doublet_rate <- if (length(args) >= 3) as.numeric(args[[3]]) else 0.1
random_seed <- if (length(args) >= 4) as.integer(args[[4]]) else 0
set.seed(random_seed)

write_status <- function(cell_ids, scores, status, message) {
  out <- data.frame(
    cell_id = cell_ids,
    method = "scDblFinder",
    score = scores,
    status = status,
    message = message,
    stringsAsFactors = FALSE
  )
  write.csv(out, output_csv, row.names = FALSE)
}

barcodes_path <- file.path(input_dir, "barcodes.tsv")
matrix_path <- file.path(input_dir, "matrix.mtx")
genes_path <- file.path(input_dir, "genes.tsv")

if (!file.exists(barcodes_path)) {
  write_status(character(), numeric(), "skipped", paste("Missing", barcodes_path))
  quit(status = 0)
}
cell_ids <- readLines(barcodes_path)

if (!requireNamespace("Matrix", quietly = TRUE) ||
    !requireNamespace("SingleCellExperiment", quietly = TRUE) ||
    !requireNamespace("scDblFinder", quietly = TRUE)) {
  write_status(cell_ids, rep(NA_real_, length(cell_ids)), "skipped", "Required R package missing: Matrix, SingleCellExperiment, or scDblFinder")
  quit(status = 0)
}

tryCatch({
  counts <- Matrix::readMM(matrix_path)
  genes <- if (file.exists(genes_path)) readLines(genes_path) else paste0("gene_", seq_len(nrow(counts)))
  rownames(counts) <- make.unique(genes)
  colnames(counts) <- cell_ids
  sce <- SingleCellExperiment::SingleCellExperiment(list(counts = counts))
  sce <- scDblFinder::scDblFinder(sce, dbr = expected_doublet_rate)
  score <- SingleCellExperiment::colData(sce)$scDblFinder.score
  write_status(cell_ids, as.numeric(score), "success", "")
}, error = function(e) {
  write_status(cell_ids, rep(NA_real_, length(cell_ids)), "skipped", paste("scDblFinder failed:", conditionMessage(e)))
})
