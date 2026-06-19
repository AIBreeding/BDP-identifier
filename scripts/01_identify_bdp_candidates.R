#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(data.table); library(optparse)})

opts <- parse_args(OptionParser(option_list = list(
  make_option("--gff", type = "character", help = "GFF3 or GFF3.gz file"),
  make_option("--species", type = "character", help = "Species label"),
  make_option("--max-distance", type = "integer", default = 20000),
  make_option("--output", type = "character", help = "Output TSV")
)))
stopifnot(!is.null(opts$gff), !is.null(opts$species), !is.null(opts$output))

gff <- fread(opts$gff, sep = "\t", header = FALSE, quote = "", fill = TRUE,
             col.names = c("chr", "source", "type", "start", "end", "score", "strand", "phase", "attributes"))
genes <- gff[type == "gene" & strand %chin% c("+", "-"), .(
  gene_id = sub("^.*(?:ID=gene:|ID=)([^;]+).*$", "\\1", attributes, perl = TRUE),
  chr, start = as.integer(start), end = as.integer(end), strand
)]
setorder(genes, chr, start, end)
genes[, `:=`(next_chr = shift(chr, type = "lead"), geneB = shift(gene_id, type = "lead"),
             startB = shift(start, type = "lead"), endB = shift(end, type = "lead"),
             strandB = shift(strand, type = "lead"))]
pairs <- genes[chr == next_chr & strand == "-" & strandB == "+"]
pairs[, distance := startB - end]
pairs <- pairs[distance <= opts$`max-distance`, .(
  species = opts$species, chr, geneA = gene_id, startA = start, endA = end,
  strandA = strand, geneB, startB, endB, strandB, distance
)]
dir.create(dirname(opts$output), recursive = TRUE, showWarnings = FALSE)
fwrite(pairs, opts$output, sep = "\t")
message(sprintf("Wrote %d candidate pairs to %s", nrow(pairs), opts$output))
