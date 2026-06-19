#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(data.table); library(optparse)})
opts <- parse_args(OptionParser(option_list = list(
  make_option("--manifest", type = "character", help = "TSV: species,pairs,activity"),
  make_option("--output", type = "character")
)))
stopifnot(!is.null(opts$manifest), !is.null(opts$output))
m <- fread(opts$manifest)
stopifnot(all(c("species", "pairs", "activity") %in% names(m)))
merged <- rbindlist(lapply(seq_len(nrow(m)), function(i) {
  p <- fread(m$pairs[i]); a <- fread(m$activity[i])
  if (tolower(m$species[i]) == "rice") {
    p[, `:=`(geneA = toupper(geneA), geneB = toupper(geneB))]
    a[, `:=`(geneA = toupper(geneA), geneB = toupper(geneB))]
  }
  merge(p, a, by = c("geneA", "geneB"), all.x = TRUE)[, species := m$species[i]][]
}), fill = TRUE)
setcolorder(merged, c("species", setdiff(names(merged), "species")))
dir.create(dirname(opts$output), recursive = TRUE, showWarnings = FALSE)
fwrite(merged, opts$output, sep = "\t")
