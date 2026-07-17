#!/usr/bin/env Rscript

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

cran_packages <- c("Matrix", "Seurat")
missing_cran <- cran_packages[!vapply(cran_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_cran)) {
  install.packages(missing_cran)
}

bioc_packages <- c("SingleCellExperiment", "SummarizedExperiment", "scDblFinder", "scds")
missing_bioc <- bioc_packages[!vapply(bioc_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_bioc) > 0) {
  BiocManager::install(missing_bioc, ask = FALSE, update = FALSE)
}

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
if (!requireNamespace("DoubletFinder", quietly = TRUE)) {
  remotes::install_github("chris-mcginnis-ucsf/DoubletFinder")
}

packages <- c("Matrix", "Seurat", "SingleCellExperiment", "SummarizedExperiment", "scDblFinder", "scds", "DoubletFinder")
versions <- data.frame(
  package = packages,
  version = vapply(packages, function(package) {
    if (requireNamespace(package, quietly = TRUE)) as.character(utils::packageVersion(package)) else "NOT_INSTALLED"
  }, character(1)),
  stringsAsFactors = FALSE
)
utils::write.csv(versions, "reproducibility/environment/r_package_versions.csv", row.names = FALSE)
print(versions)
