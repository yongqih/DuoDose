#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1) args[[1]] else "results/external_input/dataset"
output_csv <- if (length(args) >= 2) args[[2]] else "scds_scores.csv"
expected_doublet_rate <- if (length(args) >= 3) as.numeric(args[[3]]) else 0.1
random_seed <- if (length(args) >= 4) as.integer(args[[4]]) else 0
set.seed(random_seed)

write_status <- function(cell_ids, scores, status, message) {
  scores <- as.numeric(scores)
  scores[!is.finite(scores)] <- NA_real_
  out <- data.frame(
    cell_id = cell_ids,
    method = "scds",
    score = scores,
    status = status,
    message = message,
    stringsAsFactors = FALSE
  )
  write.csv(out, output_csv, row.names = FALSE)
}

extract_score <- function(result, score_name) {
  if (is.null(result)) {
    return(NULL)
  }
  if (inherits(result, "SingleCellExperiment")) {
    col_data <- SummarizedExperiment::colData(result)
    if (score_name %in% colnames(col_data)) {
      return(col_data[[score_name]])
    }
  }
  if (is.data.frame(result) || inherits(result, "DataFrame")) {
    if (score_name %in% colnames(result)) {
      return(result[[score_name]])
    }
  }
  if (is.numeric(result) || is.integer(result)) {
    return(result)
  }
  NULL
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
    !requireNamespace("SummarizedExperiment", quietly = TRUE) ||
    !requireNamespace("scds", quietly = TRUE)) {
  write_status(cell_ids, rep(NA_real_, length(cell_ids)), "skipped", "Required R package missing: Matrix, SingleCellExperiment, SummarizedExperiment, or scds")
  quit(status = 0)
}

tryCatch({
  counts_matrix <- Matrix::readMM(matrix_path)
  if (ncol(counts_matrix) != length(cell_ids) && nrow(counts_matrix) == length(cell_ids)) {
    counts_matrix <- Matrix::t(counts_matrix)
  }
  genes <- if (file.exists(genes_path)) readLines(genes_path) else paste0("gene_", seq_len(nrow(counts_matrix)))
  if (length(genes) != nrow(counts_matrix)) {
    genes <- paste0("gene_", seq_len(nrow(counts_matrix)))
  }
  rownames(counts_matrix) <- make.unique(as.character(genes))
  colnames(counts_matrix) <- cell_ids

  gene_keep <- Matrix::rowSums(counts_matrix) > 0
  if (any(gene_keep)) {
    counts_matrix <- counts_matrix[gene_keep, , drop = FALSE]
  }
  if (nrow(counts_matrix) == 0 || ncol(counts_matrix) == 0) {
    write_status(cell_ids, rep(NA_real_, length(cell_ids)), "failed", "No nonzero genes available for scds")
    quit(status = 0)
  }

  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = counts_matrix)
  )

  messages <- character()
  score <- NULL
  sce_hybrid <- tryCatch(
    scds::cxds_bcds_hybrid(sce, verb = FALSE),
    error = function(e) {
      messages <<- c(messages, paste("cxds_bcds_hybrid failed:", conditionMessage(e)))
      NULL
    }
  )
  if (!is.null(sce_hybrid)) {
    score <- extract_score(sce_hybrid, "hybrid_score")
  }

  if (is.null(score)) {
    sce_cxds <- tryCatch(
      scds::cxds(sce, verb = FALSE),
      error = function(e) {
        messages <<- c(messages, paste("cxds failed:", conditionMessage(e)))
        NULL
      }
    )
    if (!is.null(sce_cxds)) {
      score <- extract_score(sce_cxds, "cxds_score")
    }
  }

  if (is.null(score)) {
    sce_bcds <- tryCatch(
      scds::bcds(sce, retRes = TRUE, verb = FALSE),
      error = function(e) {
        messages <<- c(messages, paste("bcds failed:", conditionMessage(e)))
        NULL
      }
    )
    if (!is.null(sce_bcds)) {
      score <- extract_score(sce_bcds, "bcds_score")
    }
  }

  if (is.null(score)) {
    write_status(cell_ids, rep(NA_real_, length(cell_ids)), "failed", paste(messages, collapse = " | "))
    quit(status = 0)
  }
  score <- as.numeric(score)
  if (length(score) != length(cell_ids)) {
    write_status(cell_ids, rep(NA_real_, length(cell_ids)), "failed", paste("scds score length", length(score), "did not match", length(cell_ids), "cells"))
    quit(status = 0)
  }
  score[!is.finite(score)] <- NA_real_
  write_status(cell_ids, score, "success", "")
}, error = function(e) {
  write_status(cell_ids, rep(NA_real_, length(cell_ids)), "failed", paste("scds failed:", conditionMessage(e)))
})
