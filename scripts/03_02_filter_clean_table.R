#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(data.table); library(optparse)})
opts <- parse_args(OptionParser(option_list = list(
  make_option("--input", type = "character"), make_option("--output", type = "character"),
  make_option("--max-distance", type = "integer", default = 1000),
  make_option("--min-observations", type = "integer", default = 1000)
)))
stopifnot(!is.null(opts$input), !is.null(opts$output))
x <- fread(opts$input)
x <- x[distance > 0 & distance <= opts$`max-distance` & eff_obs >= opts$`min-observations` &
       is.finite(cor_log) & !grepl("^(Mt|Pt|Un)", chr, ignore.case = TRUE)]
fwrite(x, opts$output, sep = "\t")
message(sprintf("Retained %d high-confidence BDP candidates", nrow(x)))
