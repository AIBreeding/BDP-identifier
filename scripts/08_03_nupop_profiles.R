library(optparse)
library(data.table)
library(NuPoP)

option_list <- list(
  make_option("--manifest", type = "character", help = "TSV: species,fasta,nupop_species,model"),
  make_option("--output-dir", type = "character", help = "Output directory")
)
opts <- parse_args(OptionParser(option_list = option_list))
m <- fread(opts$manifest)
dir.create(opts$`output-dir`, recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(opts$`output-dir`)
old_wd <- getwd()

# IUPAC 模糊码 → N
iupac_to_n <- function(seq_str) {
  gsub("[RYKMSWBDHV]", "N", seq_str)
}

for (i in seq_len(nrow(m))) {
  raw_fasta <- normalizePath(file.path(old_wd, m$fasta[i]), mustWork = TRUE)
  lines <- readLines(raw_fasta)
  header_pos <- which(startsWith(lines, ">"))
  
  for (k in seq_along(header_pos)) {
    h_idx <- header_pos[k]
    header <- sub("^>", "", lines[h_idx])
    header <- gsub("[|:]", "_", header)
    header <- gsub(" ", "", header)
    
    if (k < length(header_pos)) {
      seq_lines <- lines[(h_idx + 1):(header_pos[k + 1] - 1)]
    } else {
      seq_lines <- lines[(h_idx + 1):length(lines)]
    }
    seq_text <- paste(seq_lines, collapse = "")
    seq_text <- iupac_to_n(seq_text)
    
    tmp_fasta <- tempfile(fileext = ".fasta")
    writeLines(c(paste0(">", header), seq_text), tmp_fasta)
    
    # 切到输出目录再跑（predNuPoP 写当前目录）
    setwd(out_dir)
    predNuPoP(file = tmp_fasta, species = m$nupop_species[i], model = m$model[i])
    setwd(old_wd)
    
    unlink(tmp_fasta)
  }
  
  cat("Done:", m$species[i], "\n")
}
