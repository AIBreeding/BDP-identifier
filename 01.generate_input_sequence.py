import csv
import pandas as pd
from Bio import SeqIO

# Module-level genome dictionaries (loaded once)
AT_GENOME = None
OS_GENOME = None
ZM_GENOME = None
TA_GENOME = None
GENOME_DB = None

def load_genome_databases(path):
    """Load genome sequences into dictionaries (one-time operation)."""
    global AT_GENOME, OS_GENOME, ZM_GENOME, TA_GENOME, GENOME_DB
    
    if GENOME_DB is None:  # Only load if not already loaded
        print("Loading genome databases (one-time operation)...")
        AT_GENOME = SeqIO.to_dict(SeqIO.parse(''.join([path, "Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"]), "fasta"))
        OS_GENOME = SeqIO.to_dict(SeqIO.parse(''.join([path, "/00.past_chr_wrong_version/sequence_all/os_sequence/all.con.fa"]), "fasta"))
        ZM_GENOME = SeqIO.to_dict(SeqIO.parse(''.join([path, "/00.past_chr_wrong_version/sequence_all/zm_sequence/Zea_mays.B73_RefGen_v4.dna.toplevel.fa"]), "fasta"))
        TA_GENOME = SeqIO.to_dict(SeqIO.parse(''.join([path, "/00.past_chr_wrong_version/sequence_all/ta_sequence/Triticum_aestivum.IWGSC.dna.toplevel.fa"]), "fasta"))
        
        GENOME_DB = {
            'arabidopsis': AT_GENOME,
            'rice': OS_GENOME,
            'maize': ZM_GENOME,
            'wheat': TA_GENOME
        }
    else:
        print("Genome databases already loaded, skipping reload.")
    
    return GENOME_DB

def filter_dataframe(df, max_distance, min_eff_obs):
    """Filter the dataframe based on distance and effective observations."""
    return df[(df['distance'] <= max_distance) & 
             (df['distance'] > 0) & 
             (df['eff_obs'] > min_eff_obs)].reset_index(drop=True)

def extract_sequences(df, genome_db, target_length, output_file):
    """Extract sequences with specified target length and write to output file."""
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['gene_id', 'species', 'chrom', 'endA', 'startB', 
                         'expanded_start', 'expanded_end', 'sequence', 'sequence_length'])

        for index, row in df.iterrows():
            gene_id = row['geneA']
            chrom = row['chr']
            endA = int(row['endA'])
            startB = int(row['startB'])
            species = row['species'].lower()
            print(f"{gene_id} is processing...")

            if species not in genome_db:
                print(f"Skipping {gene_id}: Unsupported species '{species}'")
                continue
                
            genome = genome_db[species]
            chrom_key = str(chrom)
            
            if chrom_key not in genome:
                print(f"Skipping {gene_id}: Chromosome {chrom_key} not found")
                continue
                
            seq_record = genome[chrom_key]
            seq_len = len(seq_record.seq)

            if startB <= endA:
                print(f"Skipping {gene_id}: startB ({startB}) <= endA ({endA})")
                continue

            midpoint = (endA + startB) // 2  # 计算中点坐标
                
            # 计算扩展长度
            left_extend = target_length // 2
            right_extend = target_length - left_extend

            # 计算新坐标
            new_start = midpoint - left_extend
            new_end = midpoint + right_extend

            # 处理边界情况
            safe_start = max(1, new_start)
            safe_end = min(seq_len, new_end)

            # Extract sequence (automatically handles out-of-bound cases)
            expanded_seq = str(seq_record.seq[max(0, new_start-1):min(seq_len, new_end)])
            ext_seq_len = len(expanded_seq)

            print(f"{gene_id} interval sequence length is {ext_seq_len}")

            writer.writerow([
                gene_id,
                species,
                chrom_key,
                endA,
                startB,
                new_start,
                new_end,
                expanded_seq,
                ext_seq_len
            ])
            print(f"{gene_id} is done!!!!!!!")

def process_sequences(input_path, output_path, target_lengths, max_distance=1000, min_eff_obs=1000):
    """Main function to process sequences with different target lengths."""
    # Load genome databases (will only load once)
    genome_db = load_genome_databases(input_path)
    
    # Read input file
    input_file = f"{input_path}all_pair_info_evo2.csv"
    df = pd.read_csv(input_file)
    
    # Filter dataframe
    df_filtered = filter_dataframe(df, max_distance, min_eff_obs)
    
    # Save filtered pair info
    pair_info_file = f"{output_path}interseq_dist{max_distance}_effobs{min_eff_obs}_pair_info.csv"
    df_filtered.to_csv(pair_info_file)
    
    # Process each target length
    for length in target_lengths:
        print(f"\nProcessing target length: {length}bp")
        output_file = f"{output_path}interseq_dist{max_distance}_effobs{min_eff_obs}_{length}bp.csv"
        extract_sequences(df_filtered, genome_db, length, output_file)
    print(f"{target_lengths}-bp interval sequence generation is done!!!!!!!")


# Example usage
if __name__ == "__main__":
    # Define paths
    input_path = "/home/user/03.shang_gao/data/"
    output_path = "/home/user/03.shang_gao/12.semi_supervised_learning/01.generate_input_sequence/"
    
    # Define parameters
    target_lengths = [128, 256, 512, 1024]  # Sequence lengths to extract
    max_distance = 1000  # Maximum distance between gene pairs
    min_eff_obs = 1000   # Minimum effective observations
    
    # Run processing
    process_sequences(input_path, output_path, target_lengths, max_distance, min_eff_obs)
