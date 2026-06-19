from pathlib import Path


def read_fasta(path):
    records, name, chunks = [], None, []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line: continue
        if line.startswith(">"):
            if name is not None: records.append((name, "".join(chunks).upper()))
            name, chunks = line[1:].split()[0], []
        else: chunks.append(line)
    if name is not None: records.append((name, "".join(chunks).upper()))
    return records


def pair_from_id(sequence_id):
    parts = str(sequence_id).split("|")
    if len(parts) < 2: raise ValueError(f"Expected species|geneA-geneB header: {sequence_id}")
    return parts[1]
