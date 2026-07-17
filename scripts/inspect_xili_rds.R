#!/usr/bin/env Rscript

print_line <- function(...) cat(paste0(...), "\n", sep = "")

dim_text <- function(x) {
  d <- tryCatch(dim(x), error = function(e) NULL)
  if (is.null(d)) return("NA")
  paste(d, collapse = " x ")
}

s4_dim_text <- function(x) {
  d <- tryCatch({
    if (isS4(x)) methods::slot(x, "Dim") else NULL
  }, error = function(e) NULL)
  if (is.null(d)) return("NA")
  paste(d, collapse = " x ")
}

head_text <- function(x, n = 6) {
  if (is.null(x) || length(x) == 0) return("NA")
  paste(utils::head(as.character(x), n), collapse = ", ")
}

table_text <- function(x, n = 10) {
  tbl <- tryCatch(sort(table(as.character(x), useNA = "ifany"), decreasing = TRUE), error = function(e) NULL)
  if (is.null(tbl) || length(tbl) == 0) return("NA")
  tbl <- utils::head(tbl, n)
  paste(paste(names(tbl), as.integer(tbl), sep = "="), collapse = ", ")
}

is_vector_like <- function(x) {
  is.character(x) || is.factor(x) || is.logical(x) || is.numeric(x) || is.integer(x)
}

is_matrix_like <- function(x) {
  inherits(x, "Matrix") || is.matrix(x) || is.data.frame(x) || (!is.null(tryCatch(dim(x), error = function(e) NULL)))
}

inspect_element <- function(x, label) {
  print_line("")
  print_line("element: ", label)
  print_line("  class: ", paste(class(x), collapse = ", "))
  print_line("  dim: ", dim_text(x))
  print_line("  S4 Dim slot: ", s4_dim_text(x))
  print_line("  length: ", length(x))
  if (is_matrix_like(x)) {
    rn <- tryCatch(rownames(x), error = function(e) NULL)
    cn <- tryCatch(colnames(x), error = function(e) NULL)
    print_line("  rownames head: ", head_text(rn))
    print_line("  colnames head: ", head_text(cn))
  }
  if (is_vector_like(x) && !is.matrix(x) && !is.data.frame(x)) {
    print_line("  values head: ", head_text(x))
    print_line("  value table: ", table_text(x))
  }
}

inspect_rds <- function(path) {
  print_line("input path: ", normalizePath(path, winslash = "/", mustWork = FALSE))
  print_line("file exists: ", file.exists(path))
  if (file.exists(path)) {
    info <- file.info(path)
    print_line("file size bytes: ", info$size)
  }
  print_line("loading RDS...")
  if (requireNamespace("Matrix", quietly = TRUE)) {
    suppressPackageStartupMessages(loadNamespace("Matrix"))
  }
  obj <- tryCatch(
    readRDS(path),
    error = function(e) {
      print_line("readRDS failed: ", conditionMessage(e))
      return(NULL)
    }
  )
  if (is.null(obj)) return(invisible(NULL))

  print_line("loaded successfully")
  print_line("class(object): ", paste(class(obj), collapse = ", "))
  nms <- tryCatch(names(obj), error = function(e) NULL)
  print_line("names(object): ", if (is.null(nms)) "NULL" else paste(nms, collapse = ", "))
  print_line("dim(object): ", dim_text(obj))
  print_line("")
  print_line("str(object, max.level = 2):")
  utils::str(obj, max.level = 2)

  if (is.list(obj)) {
    print_line("")
    print_line("list elements:")
    for (i in seq_along(obj)) {
      element_name <- if (!is.null(nms) && length(nms) >= i && nzchar(nms[[i]])) nms[[i]] else paste0("[[", i, "]]")
      inspect_element(obj[[i]], element_name)
    }
  }
  invisible(obj)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript scripts/inspect_xili_rds.R path/to/dataset.rds", call. = FALSE)
}

inspect_rds(args[[1]])
