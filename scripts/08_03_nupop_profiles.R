#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(NuPoP); library(data.table); library(optparse)})
opts <- parse_args(OptionParser(option_list = list(
  make_option("--manifest", type = "character", help = "TSV: species,fasta,nupop_species,model"),
  make_option("--output-dir", type = "character")
)))
stopifnot(!is.null(opts$manifest), !is.null(opts$`output-dir`))
m <- fread(opts$manifest); dir.create(opts$`output-dir`, recursive = TRUE, showWarnings = FALSE)
old_wd <- getwd()
on.exit(setwd(old_wd), add = TRUE)
setwd(opts$`output-dir`)
for (i in seq_len(nrow(m))) {
  fasta <- normalizePath(file.path(old_wd, m$fasta[i]), mustWork = TRUE)
  predNuPoP(file = fasta, species = m$nupop_species[i], model = m$model[i])
}
