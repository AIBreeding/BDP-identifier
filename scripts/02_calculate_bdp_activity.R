#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(data.table); library(optparse)})

opts <- parse_args(OptionParser(option_list = list(
  make_option("--expression", type = "character", help = "Genes x samples table"),
  make_option("--pairs", type = "character", help = "Candidate-pair TSV"),
  make_option("--gene-column", type = "character", default = "gene_id"),
  make_option("--min-observations", type = "integer", default = 5),
  make_option("--uppercase-ids", action = "store_true", default = FALSE),
  make_option("--output", type = "character", help = "Output TSV")
)))
stopifnot(!is.null(opts$expression), !is.null(opts$pairs), !is.null(opts$output))

expr <- fread(opts$expression)
if (!opts$`gene-column` %in% names(expr)) stop("Missing gene column: ", opts$`gene-column`)
setnames(expr, opts$`gene-column`, "gene_id")
pairs <- fread(opts$pairs)
if (opts$`uppercase-ids`) {
  expr[, gene_id := toupper(gene_id)]
  pairs[, `:=`(geneA = toupper(geneA), geneB = toupper(geneB))]
}
mat <- as.matrix(expr[, setdiff(names(expr), "gene_id"), with = FALSE])
storage.mode(mat) <- "double"
rownames(mat) <- expr$gene_id

activity <- function(a, b) {
  if (!a %in% rownames(mat) || !b %in% rownames(mat)) return(list(eff_obs = 0L, cor_log = NA_real_))
  x <- mat[a, ]; y <- mat[b, ]
  keep <- is.finite(x) & is.finite(y) & x > 0 & y > 0
  n <- sum(keep)
  if (n < opts$`min-observations`) return(list(eff_obs = n, cor_log = NA_real_))
  list(eff_obs = n, cor_log = cor(log10(x[keep]), log10(y[keep]), method = "pearson"))
}
ans <- Map(activity, pairs$geneA, pairs$geneB)
out <- data.table(geneA = pairs$geneA, geneB = pairs$geneB,
                  eff_obs = vapply(ans, `[[`, integer(1), "eff_obs"),
                  cor_log = vapply(ans, `[[`, numeric(1), "cor_log"))
dir.create(dirname(opts$output), recursive = TRUE, showWarnings = FALSE)
fwrite(out, opts$output, sep = "\t")
message(sprintf("Wrote activity for %d pairs to %s", nrow(out), opts$output))
