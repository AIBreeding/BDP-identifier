#!/usr/bin/env Rscript
install.packages(c("data.table", "optparse"), repos = "https://cloud.r-project.org")
# NuPoP is only required for scripts/08_03_nupop_profiles.R.
if (!requireNamespace("NuPoP", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
  BiocManager::install("NuPoP", ask = FALSE, update = FALSE)
}
