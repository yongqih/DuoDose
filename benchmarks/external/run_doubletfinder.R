#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1) args[[1]] else "results/external_input/dataset"
output_csv <- if (length(args) >= 2) args[[2]] else "doubletfinder_scores.csv"
expected_doublet_rate <- if (length(args) >= 3) as.numeric(args[[3]]) else 0.1
random_seed <- if (length(args) >= 4) as.integer(args[[4]]) else 0
audit_dir <- if (length(args) >= 5 && nzchar(args[[5]])) args[[5]] else NULL
set.seed(random_seed)

if (!is.null(audit_dir)) {
  dir.create(audit_dir, recursive = TRUE, showWarnings = FALSE)
}

write_status <- function(cell_ids, scores, status, message) {
  out <- data.frame(
    cell_id = cell_ids,
    method = "DoubletFinder",
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
    !requireNamespace("Seurat", quietly = TRUE) ||
    !requireNamespace("DoubletFinder", quietly = TRUE)) {
  write_status(cell_ids, rep(NA_real_, length(cell_ids)), "skipped", "Required R package missing: Matrix, Seurat, or DoubletFinder")
  quit(status = 0)
}

tryCatch({
  counts <- Matrix::readMM(matrix_path)
  genes <- if (file.exists(genes_path)) readLines(genes_path) else paste0("gene_", seq_len(nrow(counts)))
  if (ncol(counts) != length(cell_ids)) {
    stop(sprintf("matrix/barcode mismatch: %d matrix columns versus %d barcodes", ncol(counts), length(cell_ids)))
  }
  if (anyDuplicated(cell_ids)) {
    stop("input barcodes are not unique")
  }
  rownames(counts) <- make.unique(genes)
  colnames(counts) <- cell_ids
  seu <- Seurat::CreateSeuratObject(counts = counts)
  initial_seurat_ids <- colnames(seu)
  if (!identical(initial_seurat_ids, cell_ids)) {
    stop("Seurat changed or reordered input barcodes during CreateSeuratObject")
  }
  seu <- Seurat::NormalizeData(seu, verbose = FALSE)
  seu <- Seurat::FindVariableFeatures(seu, verbose = FALSE)
  seu <- Seurat::ScaleData(seu, verbose = FALSE)
  seu <- Seurat::RunPCA(seu, npcs = 20, verbose = FALSE)
  n_exp <- max(1, round(expected_doublet_rate * length(cell_ids)))
  pN <- 0.25
  pK <- 0.09
  pK_source <- "frozen_wrapper_parameter"
  pcs <- seq_len(min(20, ncol(seu[["pca"]]@cell.embeddings)))
  stale_pann_cols <- grep("^pANN", colnames(seu@meta.data), value = TRUE)

  get_df_fun <- function(name, fallback) {
    ns <- asNamespace("DoubletFinder")
    if (exists(name, where = ns, inherits = FALSE)) {
      return(get(name, envir = ns))
    }
    if (exists(fallback, where = ns, inherits = FALSE)) {
      return(get(fallback, envir = ns))
    }
    return(NULL)
  }

  doublet_finder <- get_df_fun("doubletFinder_v3", "doubletFinder")
  if (is.null(doublet_finder)) {
    stop("No DoubletFinder doubletFinder function found")
  }
  seu <- tryCatch(
    doublet_finder(seu, PCs = pcs, pN = pN, pK = pK, nExp = n_exp, reuse.pANN = FALSE, sct = FALSE),
    error = function(e) {
      tryCatch(
        doublet_finder(seu, PCs = pcs, pN = pN, pK = pK, nExp = n_exp, sct = FALSE),
        error = function(e2) doublet_finder(seu, PCs = pcs, pN = pN, pK = pK, nExp = n_exp)
      )
    }
  )
  final_seurat_ids <- rownames(seu@meta.data)
  if (!identical(final_seurat_ids, cell_ids)) {
    stop("DoubletFinder changed or reordered Seurat barcodes")
  }
  all_pann_cols <- grep("^pANN", colnames(seu@meta.data), value = TRUE)
  new_pann_cols <- setdiff(all_pann_cols, stale_pann_cols)
  if (length(new_pann_cols) != 1) {
    stop(sprintf("expected exactly one newly created pANN column, found %d: %s", length(new_pann_cols), paste(new_pann_cols, collapse = ",")))
  }
  pann_col <- new_pann_cols[[1]]
  score <- as.numeric(seu@meta.data[cell_ids, pann_col, drop = TRUE])
  if (length(score) != length(cell_ids) || any(!is.finite(score))) {
    stop("DoubletFinder did not return one finite pANN score for every input barcode")
  }
  if (!is.null(audit_dir)) {
    alignment <- data.frame(
      input_position = seq_along(cell_ids),
      input_cell_id = cell_ids,
      initial_seurat_cell_id = initial_seurat_ids,
      final_seurat_cell_id = final_seurat_ids,
      output_cell_id = cell_ids,
      initial_order_match = initial_seurat_ids == cell_ids,
      final_order_match = final_seurat_ids == cell_ids,
      score_finite = is.finite(score),
      stringsAsFactors = FALSE
    )
    write.csv(alignment, file.path(audit_dir, "doubletfinder_alignment_audit.csv"), row.names = FALSE)
    parameter_audit <- data.frame(
      score_definition = "pANN: proportion of artificial nearest neighbors; larger values are more doublet-like",
      selected_pann_column = pann_col,
      stale_pann_column_count = length(stale_pann_cols),
      stale_pann_columns = paste(stale_pann_cols, collapse = ";"),
      new_pann_column_count = length(new_pann_cols),
      pN = pN,
      pK = pK,
      pK_source = pK_source,
      n_pcs = length(pcs),
      nExp = n_exp,
      expected_doublet_rate = expected_doublet_rate,
      n_input_cells = length(cell_ids),
      n_input_genes = nrow(counts),
      n_output_cells = length(score),
      finite_score_count = sum(is.finite(score)),
      input_output_barcode_sets_identical = setequal(cell_ids, final_seurat_ids),
      input_output_barcode_order_identical = identical(cell_ids, final_seurat_ids),
      stringsAsFactors = FALSE
    )
    write.csv(parameter_audit, file.path(audit_dir, "doubletfinder_parameter_audit_raw.csv"), row.names = FALSE)
  }
  write_status(cell_ids, score, "success", paste("continuous", pann_col, "score aligned by barcode"))
}, error = function(e) {
  write_status(cell_ids, rep(NA_real_, length(cell_ids)), "skipped", paste("DoubletFinder failed:", conditionMessage(e)))
})
