#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript scripts/convert_xili_rds.R input.rds output_dir", call. = FALSE)
}

input_rds <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

label_candidates <- c(
  "doublet", "Doublet", "label", "labels", "class", "classification",
  "experimental_doublet", "doublet_label", "doublet_labels", "is_doublet",
  "demuxlet_cls", "hashing_label", "hto_classification", "multiplet", "singlet"
)
matrix_candidates <- c("counts", "count", "raw_counts", "matrix", "expr", "exprs", "data", "x")
metadata_candidates <- c("labels", "label", "doublet", "doublet_labels", "is_doublet", "classification", "metadata", "meta", "cell_metadata", "obs")

log_msg <- function(...) cat(paste0(...), "\n", sep = "")

json_escape <- function(x) {
  x <- as.character(x)
  x <- gsub("\\\\", "\\\\\\\\", x)
  x <- gsub('"', '\\"', x)
  x <- gsub("\n", "\\\\n", x)
  x
}

json_value <- function(value) {
  if (length(value) == 0 || all(is.na(value))) return("null")
  if (length(value) > 1) {
    return(paste0("[", paste(vapply(value, json_value, character(1)), collapse = ", "), "]"))
  }
  if (is.numeric(value) || is.integer(value)) return(as.character(value))
  if (is.logical(value)) return(tolower(as.character(value)))
  paste0('"', json_escape(value), '"')
}

write_report <- function(fields) {
  defaults <- list(
    dataset = tools::file_path_sans_ext(basename(input_rds)),
    input_rds = normalizePath(input_rds, winslash = "/", mustWork = FALSE),
    status = "failed",
    message = "",
    object_class = NA_character_,
    rds_structure = NA_character_,
    matrix_class = NA_character_,
    matrix_dim_original = NA_character_,
    matrix_dim_written = NA_character_,
    dim_from_dim = NA_character_,
    dim_from_nrow_ncol = NA_character_,
    dim_from_S4_Dim = NA_character_,
    dim_from_dimnames = NA_character_,
    selected_orientation = NA_character_,
    n_genes = NA_integer_,
    n_cells = NA_integer_,
    n_doublets = NA_integer_,
    n_singlets = NA_integer_,
    doublet_rate = NA_real_,
    label_values = NA_character_,
    label_value_table = NA_character_,
    label_source = NA_character_,
    gene_id_source = NA_character_,
    cell_id_source = NA_character_
  )
  defaults[names(fields)] <- fields
  keys <- names(defaults)
  lines <- c("{")
  for (i in seq_along(keys)) {
    key <- keys[[i]]
    comma <- if (i < length(keys)) "," else ""
    lines <- c(lines, paste0('  "', key, '": ', json_value(defaults[[key]]), comma))
  }
  lines <- c(lines, "}")
  writeLines(lines, file.path(output_dir, "conversion_report.json"))
}

fail <- function(message, fields = list()) {
  log_msg("conversion failed: ", message)
  fields$status <- "failed"
  fields$message <- message
  write_report(fields)
  stop(message, call. = FALSE)
}

if (!requireNamespace("Matrix", quietly = TRUE)) {
  fail("Required R package Matrix is not installed")
}

is_matrix_like <- function(x) {
  inherits(x, "Matrix") || is.matrix(x) || is.data.frame(x)
}

is_vector_like <- function(x) {
  is.character(x) || is.factor(x) || is.logical(x) || is.numeric(x) || is.integer(x)
}

get_matrix_dims <- function(mat) {
  dim_from_dim <- tryCatch(dim(mat), error = function(e) NULL)
  dim_from_nrow_ncol <- tryCatch(c(nrow(mat), ncol(mat)), error = function(e) NULL)
  dim_from_s4 <- tryCatch(if (methods::is(mat, "Matrix") || isS4(mat)) methods::slot(mat, "Dim") else NULL, error = function(e) NULL)
  dim_from_dimnames <- tryCatch({
    rn <- rownames(mat)
    cn <- colnames(mat)
    if (!is.null(rn) && !is.null(cn)) c(length(rn), length(cn)) else NULL
  }, error = function(e) NULL)
  selected <- NULL
  for (candidate in list(dim_from_dim, dim_from_nrow_ncol, dim_from_s4, dim_from_dimnames)) {
    if (!is.null(candidate) && length(candidate) == 2 && all(!is.na(candidate))) {
      selected <- as.integer(candidate)
      break
    }
  }
  list(
    selected = selected,
    dim_from_dim = dim_from_dim,
    dim_from_nrow_ncol = dim_from_nrow_ncol,
    dim_from_S4_Dim = dim_from_s4,
    dim_from_dimnames = dim_from_dimnames
  )
}

dim_label <- function(x) {
  if (is.null(x) || length(x) == 0 || any(is.na(x))) return(NA_character_)
  paste(as.integer(x), collapse = " x ")
}

coerce_matrix <- function(x) {
  if (inherits(x, "Matrix")) return(x)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (is.matrix(x)) return(Matrix::Matrix(x, sparse = TRUE))
  NULL
}

coerce_dgc <- function(mat) {
  if (!inherits(mat, "Matrix")) mat <- Matrix::Matrix(mat, sparse = TRUE)
  if (!inherits(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")
  mat
}

parse_label_vector <- function(raw) {
  values <- trimws(tolower(as.character(raw)))
  values[is.na(values)] <- ""
  positive <- c("doublet", "multiplet", "true", "1", "yes")
  negative <- c("singlet", "single", "false", "0", "no")
  labels <- rep(NA_integer_, length(values))
  labels[values %in% positive] <- 1L
  labels[values %in% negative] <- 0L
  numeric_values <- suppressWarnings(as.numeric(as.character(raw)))
  labels[!is.na(numeric_values) & numeric_values == 1] <- 1L
  labels[!is.na(numeric_values) & numeric_values == 0] <- 0L
  if (any(is.na(labels))) {
    stop(paste("unrecognized label values:", paste(unique(as.character(raw)), collapse = ", ")), call. = FALSE)
  }
  if (length(unique(labels)) < 2) {
    stop(paste("labels contain only one class:", paste(unique(as.character(raw)), collapse = ", ")), call. = FALSE)
  }
  labels
}

label_table_text <- function(raw) {
  tbl <- table(as.character(raw), useNA = "ifany")
  paste(paste(names(tbl), as.integer(tbl), sep = ":"), collapse = ";")
}

extract_from_named_list <- function(obj, candidates) {
  nms <- names(obj)
  if (is.null(nms)) return(NULL)
  lower <- tolower(nms)
  for (candidate in candidates) {
    idx <- which(lower == tolower(candidate))
    for (i in idx) {
      value <- obj[[i]]
      if (is_matrix_like(value)) return(value)
    }
  }
  for (i in seq_along(obj)) {
    value <- obj[[i]]
    if (is_matrix_like(value)) return(value)
  }
  NULL
}

extract_metadata_from_list <- function(obj) {
  nms <- names(obj)
  if (is.null(nms)) return(NULL)
  lower <- tolower(nms)
  for (candidate in metadata_candidates) {
    idx <- which(lower == tolower(candidate))
    for (i in idx) {
      value <- obj[[i]]
      if (is.data.frame(value)) return(value)
      if (is_vector_like(value)) return(data.frame(label = value, stringsAsFactors = FALSE))
    }
  }
  for (i in seq_along(obj)) {
    value <- obj[[i]]
    if (is.data.frame(value)) return(value)
  }
  NULL
}

parse_labels_from_metadata <- function(metadata) {
  if (is.null(metadata) || nrow(metadata) == 0) {
    return(list(labels = NULL, raw = NULL, source = NA_character_, message = "metadata is unavailable"))
  }
  lower_cols <- tolower(colnames(metadata))
  for (candidate in label_candidates) {
    idx <- which(lower_cols == tolower(candidate))
    for (i in idx) {
      raw <- metadata[[i]]
      parsed <- tryCatch(parse_label_vector(raw), error = function(e) NULL)
      if (!is.null(parsed)) return(list(labels = parsed, raw = raw, source = colnames(metadata)[[i]], message = "success"))
      if (grepl("doublet|label|class|multiplet|singlet", tolower(colnames(metadata)[[i]]))) {
        return(list(labels = NULL, raw = raw, source = colnames(metadata)[[i]], message = paste("ambiguous label values:", paste(unique(as.character(raw)), collapse = ", "))))
      }
    }
  }
  list(labels = NULL, raw = NULL, source = NA_character_, message = paste("no parseable label column; available columns:", paste(colnames(metadata), collapse = ", ")))
}

extract_generic_object <- function(obj) {
  metadata <- NULL
  counts <- NULL
  count_source <- NA_character_
  label_source <- NA_character_
  raw_labels <- NULL
  labels <- NULL
  rds_structure <- paste(class(obj), collapse = ",")

  if (inherits(obj, "SingleCellExperiment")) {
    if (!requireNamespace("SummarizedExperiment", quietly = TRUE)) {
      fail("SingleCellExperiment input requires SummarizedExperiment package", list(object_class = class(obj)))
    }
    assay_names <- SummarizedExperiment::assayNames(obj)
    assay_name <- if ("counts" %in% assay_names) "counts" else assay_names[[1]]
    counts <- SummarizedExperiment::assay(obj, assay_name)
    count_source <- paste0("SingleCellExperiment_assay_", assay_name)
    metadata <- as.data.frame(SummarizedExperiment::colData(obj))
    rds_structure <- "SingleCellExperiment"
  } else if (inherits(obj, "Seurat")) {
    if (!requireNamespace("Seurat", quietly = TRUE)) {
      fail("Seurat input requires Seurat package", list(object_class = class(obj)))
    }
    counts <- tryCatch(Seurat::GetAssayData(obj, slot = "counts"), error = function(e) NULL)
    count_source <- "Seurat_counts_slot"
    if (is.null(counts) || length(counts) == 0) {
      counts <- tryCatch(Seurat::GetAssayData(obj, slot = "data"), error = function(e) NULL)
      count_source <- "Seurat_data_slot"
    }
    metadata <- obj@meta.data
    rds_structure <- "Seurat"
  } else if (is.list(obj) && !is.data.frame(obj)) {
    counts <- extract_from_named_list(obj, matrix_candidates)
    count_source <- "generic_list_matrix"
    metadata <- extract_metadata_from_list(obj)
    rds_structure <- if (is.null(names(obj))) paste0("unnamed_list_length_", length(obj)) else "named_list"
  } else if (is_matrix_like(obj)) {
    counts <- obj
    count_source <- "matrix_object"
    rds_structure <- "matrix_object"
  }

  label_result <- parse_labels_from_metadata(metadata)
  labels <- label_result$labels
  raw_labels <- label_result$raw
  label_source <- label_result$source
  list(
    counts = counts,
    labels = labels,
    raw_labels = raw_labels,
    metadata = metadata,
    count_source = count_source,
    label_source = label_source,
    rds_structure = rds_structure,
    label_message = label_result$message
  )
}

log_msg("input RDS: ", normalizePath(input_rds, winslash = "/", mustWork = FALSE))
log_msg("output directory: ", normalizePath(output_dir, winslash = "/", mustWork = FALSE))

base_fields <- list()
obj <- tryCatch(readRDS(input_rds), error = function(e) {
  fail(paste("readRDS failed:", conditionMessage(e)))
})
object_class <- class(obj)
base_fields$object_class <- object_class
log_msg("object class: ", paste(object_class, collapse = ", "))

extracted <- NULL
unnamed_list <- is.list(obj) && length(obj) == 2 && (is.null(names(obj)) || all(!nzchar(names(obj))))
if (unnamed_list && is_matrix_like(obj[[1]]) && is_vector_like(obj[[2]])) {
  log_msg("detected structure: unnamed_list_matrix_labels")
  parsed_labels <- tryCatch(parse_label_vector(obj[[2]]), error = function(e) {
    fail(
      paste("could not parse labels from unnamed list element 2:", conditionMessage(e)),
      c(base_fields, list(rds_structure = "unnamed_list_matrix_labels", label_source = "unnamed_list_element_2"))
    )
  })
  extracted <- list(
    counts = obj[[1]],
    labels = parsed_labels,
    raw_labels = obj[[2]],
    metadata = NULL,
    count_source = "unnamed_list_element_1",
    label_source = "unnamed_list_element_2",
    rds_structure = "unnamed_list_matrix_labels",
    label_message = "success"
  )
} else {
  log_msg("detected structure: generic")
  extracted <- extract_generic_object(obj)
}

counts_original <- coerce_matrix(extracted$counts)
if (is.null(counts_original)) {
  fail("could not extract a count matrix from RDS object", base_fields)
}
matrix_dims <- get_matrix_dims(counts_original)
if (is.null(matrix_dims$selected)) {
  fail("could not determine count matrix dimensions", base_fields)
}
labels <- extracted$labels
if (is.null(labels)) {
  fields <- c(base_fields, list(rds_structure = extracted$rds_structure, label_source = extracted$label_source))
  fail(paste("could not extract experimental doublet labels:", extracted$label_message), fields)
}

n_rows <- matrix_dims$selected[[1]]
n_cols <- matrix_dims$selected[[2]]
n_labels <- length(labels)
orientation <- NA_character_
counts_out <- counts_original
if (n_labels == n_cols) {
  orientation <- "genes_by_cells"
  counts_out <- counts_original
} else if (n_labels == n_rows) {
  orientation <- "cells_by_genes_transposed"
  counts_out <- Matrix::t(counts_original)
} else {
  fields <- c(base_fields, list(
    rds_structure = extracted$rds_structure,
    matrix_dim_original = dim_label(matrix_dims$selected),
    dim_from_dim = dim_label(matrix_dims$dim_from_dim),
    dim_from_nrow_ncol = dim_label(matrix_dims$dim_from_nrow_ncol),
    dim_from_S4_Dim = dim_label(matrix_dims$dim_from_S4_Dim),
    dim_from_dimnames = dim_label(matrix_dims$dim_from_dimnames),
    label_source = extracted$label_source
  ))
  fail(paste("label length", n_labels, "does not match matrix rows", n_rows, "or columns", n_cols), fields)
}

counts_out <- coerce_dgc(counts_out)
genes <- rownames(counts_out)
cells <- colnames(counts_out)
gene_source <- "rownames"
cell_source <- "colnames"
if (is.null(genes)) {
  genes <- paste0("gene_", seq_len(nrow(counts_out)))
  gene_source <- "generated"
}
if (is.null(cells)) {
  cells <- paste0("cell_", seq_len(ncol(counts_out)))
  cell_source <- "generated"
}
genes <- make.unique(as.character(genes))
cells <- make.unique(as.character(cells))
rownames(counts_out) <- genes
colnames(counts_out) <- cells

if (length(labels) != ncol(counts_out)) {
  fail(paste("label length", length(labels), "does not match written cell count", ncol(counts_out)), base_fields)
}
raw_labels <- extracted$raw_labels
if (is.null(raw_labels)) raw_labels <- as.character(labels)
raw_labels <- as.character(raw_labels)

label_values <- unique(raw_labels)
label_table <- label_table_text(raw_labels)
n_doublets <- sum(labels == 1L)
n_singlets <- sum(labels == 0L)

log_msg("selected count matrix source: ", extracted$count_source)
log_msg("selected label source: ", extracted$label_source)
log_msg("matrix class: ", paste(class(counts_original), collapse = ", "))
log_msg("matrix dimensions original: ", n_rows, " x ", n_cols)
log_msg("selected orientation: ", orientation)
log_msg("matrix dimensions written: ", nrow(counts_out), " x ", ncol(counts_out))
log_msg("label table: ", label_table)

invisible(Matrix::writeMM(counts_out, file.path(output_dir, "matrix.mtx")))
write.table(data.frame(gene_id = rownames(counts_out)), file.path(output_dir, "genes.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(data.frame(cell_id = colnames(counts_out)), file.path(output_dir, "barcodes.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(data.frame(cell_id = colnames(counts_out), experimental_doublet = labels), file.path(output_dir, "labels.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(
  data.frame(cell_id = colnames(counts_out), original_label = raw_labels, experimental_doublet = labels, stringsAsFactors = FALSE),
  file.path(output_dir, "metadata.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

fields <- list(
  status = "success",
  message = "converted RDS to matrix.mtx/genes.tsv/barcodes.tsv/labels.tsv",
  object_class = object_class,
  rds_structure = extracted$rds_structure,
  matrix_class = class(counts_original),
  matrix_dim_original = dim_label(matrix_dims$selected),
  matrix_dim_written = paste(dim(counts_out), collapse = " x "),
  dim_from_dim = dim_label(matrix_dims$dim_from_dim),
  dim_from_nrow_ncol = dim_label(matrix_dims$dim_from_nrow_ncol),
  dim_from_S4_Dim = dim_label(matrix_dims$dim_from_S4_Dim),
  dim_from_dimnames = dim_label(matrix_dims$dim_from_dimnames),
  selected_orientation = orientation,
  n_genes = nrow(counts_out),
  n_cells = ncol(counts_out),
  n_doublets = n_doublets,
  n_singlets = n_singlets,
  doublet_rate = n_doublets / max(1, length(labels)),
  label_values = label_values,
  label_value_table = label_table,
  label_source = extracted$label_source,
  gene_id_source = gene_source,
  cell_id_source = cell_source
)
write_report(fields)
log_msg("output files written: matrix.mtx, genes.tsv, barcodes.tsv, labels.tsv, metadata.tsv, conversion_report.json")
