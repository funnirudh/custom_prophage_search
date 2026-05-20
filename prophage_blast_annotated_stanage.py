#!/usr/bin/env python3
"""

Optimized for many genomes (tens of thousands).
- Parallel makeblastdb + blastn per genome (configurable workers)
- Summarizes best hits per genome
- Writes a quick-look TSV summary matrix
- Splits heatmap into multiple numbered image files (configurable rows per image)
- Writes genome -> image_number mapping (TSV)

Requirements:
    - Python 3.8+
    - Biopython, pandas, matplotlib, seaborn
    - NCBI BLAST+ installed (makeblastdb, blastn)
"""

import os
import subprocess
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# -----------------------------
# USER CONFIGURATION (tune as needed)
# -----------------------------
PROPHAGE_FASTA = "/mnt/parscratch/users/bip23aj/prophages/phiMMP01_in_Ox1533.fasta"
GENOME_FOLDER = "/mnt/parscratch/users/bip23aj/genomes/26kgenomes"
OUTDIR = "/mnt/parscratch/users/bip23aj/code/prophage_blast_output"
DB_FOLDER = Path(OUTDIR) / "blast_dbs"
SUMMARY_FILE = Path(OUTDIR) / "summary_prevalence.csv"            # full summary csv
SUMMARY_TSV_QUICK = Path(OUTDIR) / "summary_prevalence_matrix.txt"  # quick-look TSV
GENOME_IMAGE_MAP = Path(OUTDIR) / "genome_image_map.txt"        # genome -> image number
HEATMAP_PREFIX = Path(OUTDIR) / "prevalence_heatmap_part"      # e.g. prevalence_heatmap_part_1.png

# Performance / plotting params
MAX_WORKERS = min(16, max(1, cpu_count() - 1))  # processes to run in parallel; tune for cluster
BLAST_THREADS_PER_JOB = 1                        # keep per-job threads low when parallelizing across genomes
HEATMAP_ROWS_PER_IMAGE = 1000                    # how many genomes per heatmap image (tweak)
HEATMAP_DPI = 150
SKIP_EXISTING = True                              # skip db/result if already present

# BLAST outfmt (tabular)
BLAST_OUTFMT = "6 qseqid sseqid pident length qcovhsp bitscore evalue"

# -----------------------------
# Utility functions
# -----------------------------
def run_cmd(cmd_list, capture_output=True):
    """Run a shell command list. Raises on non-zero return."""
    try:
        proc = subprocess.run(cmd_list, check=True, capture_output=capture_output, text=True)
        return proc.stdout if capture_output else ""
    except subprocess.CalledProcessError as e:
        print(f"ERROR running: {' '.join(cmd_list)}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        raise

def make_blast_db(genome_fasta, db_folder):
    """
    Create BLAST DB for genome_fasta and return the db base path.
    Safe: will skip if DB files already exist (unless SKIP_EXISTING False).
    """
    genome_fasta = Path(genome_fasta)
    db_base = Path(db_folder) / genome_fasta.stem
    db_base.parent.mkdir(parents=True, exist_ok=True)

    # check for typical blast db files (.nhr/.nin/.nsq) — if present, skip
    expected_file = db_base.with_suffix(".nsq")
    if SKIP_EXISTING and expected_file.exists():
        print(f"[make_blast_db] DB exists, skipping: {db_base}")
        return str(db_base)

    cmd = [
        "makeblastdb",
        "-in", str(genome_fasta),
        "-dbtype", "nucl",
        "-parse_seqids",
        "-out", str(db_base)
    ]
    print(f"[make_blast_db] Running makeblastdb for {genome_fasta.name}")
    run_cmd(cmd)
    return str(db_base)

def run_blastn_single(prophage_fasta, db_base, genome_name, out_folder):
    """
    Run blastn between prophage_fasta and db_base.
    Returns path to TSV output.
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_file = out_folder / f"{Path(prophage_fasta).stem}_vs_{genome_name}.tsv"

    # Skip if exists
    if SKIP_EXISTING and out_file.exists() and out_file.stat().st_size > 0:
        print(f"[blastn] result exists, skipping: {out_file.name}")
        return str(out_file)

    cmd = [
        "blastn",
        "-query", str(prophage_fasta),
        "-db", str(db_base),
        "-outfmt", BLAST_OUTFMT,
        "-out", str(out_file),
        "-num_threads", str(BLAST_THREADS_PER_JOB)
    ]
    print(f"[blastn] Running blastn for genome {genome_name}")
    run_cmd(cmd)
    return str(out_file)

# Worker function that will be used in parallel executor
def process_genome(genome_path):
    """
    For a single genome FASTA file:
    - makeblastdb
    - run blastn
    Returns (genome_name, result_file_path)
    """
    genome_path = Path(genome_path)
    name = genome_path.stem
    try:
        db_base = make_blast_db(genome_path, DB_FOLDER)
        result = run_blastn_single(PROPHAGE_FASTA, db_base, name, OUTDIR)
        return (name, result, None)
    except Exception as e:
        return (name, None, str(e))

# -----------------------------
# Summarize BLAST results
# -----------------------------
def summarize_results(blast_results_map, outdir):
    """
    blast_results_map: dict genome_name -> tsv_path (some entries can be None)
    Produces a DataFrame with best hit per genome (or zeros if no hits).
    Saves summary CSV and a quick TSV matrix for 'quick look'.
    """
    rows = []
    for genome_name, file_path in blast_results_map.items():
        if file_path and Path(file_path).exists() and Path(file_path).stat().st_size > 0:
            try:
                df = pd.read_csv(file_path, sep="\t", header=None, comment="#", engine="c")
                # Ensure expected columns count
                if df.shape[1] < 7:
                    # some unexpected format; skip with zeros
                    print(f"[summarize] Unexpected format in {file_path}; filling zeros")
                    rows.append((genome_name, 0.0, 0.0, 0.0, "N/A"))
                    continue
                df.columns = ["qseqid", "sseqid", "pident", "length", "qcovhsp", "bitscore", "evalue"]
                best = df.sort_values("bitscore", ascending=False).iloc[0]
                rows.append((
                    genome_name,
                    float(best["pident"]),
                    float(best["qcovhsp"]),
                    float(best["bitscore"]),
                    best["evalue"]
                ))
            except Exception as e:
                print(f"[summarize] Error reading {file_path}: {e}")
                rows.append((genome_name, 0.0, 0.0, 0.0, "N/A"))
        else:
            rows.append((genome_name, 0.0, 0.0, 0.0, "N/A"))

    summary_df = pd.DataFrame(rows, columns=["Genome", "Identity(%)", "Coverage(%)", "Bitscore", "E-value"])

    # Save full CSV
    summary_df.to_csv(Path(outdir) / "summary_prevalence.csv", index=False)

    # Save quick-look TSV (tab separated)
    summary_df.to_csv(Path(outdir) / "summary_prevalence_matrix.txt", sep="\t", index=False)

    # Print quick stats
    total = len(summary_df)
    high_cov = (summary_df["Coverage(%)"] > 50).sum()
    print(f"\n[summarize] {high_cov}/{total} genomes ({(high_cov/total)*100:.1f}%) contain >50% of the prophage sequence.")
    return summary_df

# -----------------------------
# Heatmap splitting & plotting
# -----------------------------
def split_and_plot_heatmaps(summary_df, out_prefix, rows_per_image=HEATMAP_ROWS_PER_IMAGE, dpi=HEATMAP_DPI):
    """
    Splits the summary_df (sorted by Coverage desc) into chunks of rows_per_image,
    writes genome->image mapping, saves chunked heatmap images.
    Returns number_of_images.
    """

    # Sort by Coverage descending
    df = summary_df.sort_values("Coverage(%)", ascending=False).reset_index(drop=True)

    # Data for plotting: we'll plot Identity and Coverage as two columns (heatmap will have two columns)
    heatmap_data = df.set_index("Genome")[["Identity(%)", "Coverage(%)"]]

    n_rows = heatmap_data.shape[0]
    num_images = int(np.ceil(n_rows / rows_per_image))
    print(f"[heatmap] Splitting {n_rows} genomes into {num_images} image(s) ({rows_per_image} rows per image)")

    genome_map_records = []

    # Precompute vmin/vmax for consistent color scaling across chunks
    vmin = 0.0
    vmax = 100.0

    for i in range(num_images):
        start = i * rows_per_image
        end = min((i + 1) * rows_per_image, n_rows)
        chunk = heatmap_data.iloc[start:end]

        # record genome->image mapping
        for genome in chunk.index:
            genome_map_records.append((genome, i + 1))

        # Plot chunk heatmap
        if chunk.shape[0] == 0:
            continue

        # figure height heuristic: keep rows readable but not huge
        height_per_row = 0.12  # inches per row (tweak)
        fig_height = max(2.5, chunk.shape[0] * height_per_row)
        fig_width = 6  # two columns

        plt.figure(figsize=(fig_width, fig_height))
        ax = sns.heatmap(chunk,
                         cmap="viridis",
                         vmin=vmin, vmax=vmax,
                         cbar_kws={"label": "%"},
                         annot=False,  # turn off per-cell numbers for performance; annotate if desired
                         linewidths=0.2)

        ax.set_title(f"Prophage prevalence (part {i+1}) — genomes {start+1} to {end}")
        plt.tight_layout()

        out_png = Path(f"{out_prefix}_{i+1}.png")
        plt.savefig(out_png, dpi=dpi)
        plt.close()
        print(f"[heatmap] Saved: {out_png}")

    # Save genome->image mapping
    genome_map_df = pd.DataFrame(genome_map_records, columns=["Genome", "Image_Number"])
    genome_map_df.to_csv(GENOME_IMAGE_MAP, sep="\t", index=False)
    print(f"[heatmap] Genome->image mapping saved: {GENOME_IMAGE_MAP}")

    return num_images

# -----------------------------
# Main workflow
# -----------------------------
def main():
    print("===============================================")
    print("Prophage vs. genomes BLAST Tool (large-scale)")
    print("===============================================")
    print(f"Genomes folder: {GENOME_FOLDER}")
    print(f"Output folder:  {OUTDIR}")
    print(f"DB folder:      {DB_FOLDER}")
    print(f"Workers:        {MAX_WORKERS}")
    print()

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(DB_FOLDER, exist_ok=True)

    # find genome files
    genome_paths = list(Path(GENOME_FOLDER).glob("*.fna")) \
                 + list(Path(GENOME_FOLDER).glob("*.fa")) \
                 + list(Path(GENOME_FOLDER).glob("*.fasta"))

    if not genome_paths:
        print("[main] No genome FASTA files found in:", GENOME_FOLDER)
        return

    # Use parallel executor to create DBs and run blastn per genome
    blast_results = {}  # genome_name -> result_file_path or None
    errors = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(process_genome, str(gp)): gp for gp in genome_paths}

        for fut in as_completed(futures):
            gp = futures[fut]
            try:
                genome_name, result_path, error = fut.result()
                if error:
                    errors.append((genome_name, error))
                    blast_results[genome_name] = None
                    print(f"[main] ERROR for {genome_name}: {error}")
                else:
                    blast_results[genome_name] = result_path
            except Exception as e:
                # fallback: log and continue
                name = gp.stem
                errors.append((name, str(e)))
                blast_results[name] = None
                print(f"[main] Exception processing {gp}: {e}")

    if errors:
        print(f"\n[main] Completed with {len(errors)} errors (see console). You can re-run for failed genomes.")
    else:
        print("\n[main] All genomes processed successfully.")

    # Summarize results
    print("\n[main] Summarizing BLAST results...")
    summary_df = summarize_results(blast_results, OUTDIR)

    # Save quick-look text matrix path output
    print(f"[main] Quick-look matrix saved to: {SUMMARY_TSV_QUICK}")

    # Make heatmaps (split into parts)
    print("\n[main] Generating split heatmaps...")
    num_images = split_and_plot_heatmaps(summary_df, str(HEATMAP_PREFIX), rows_per_image=HEATMAP_ROWS_PER_IMAGE)

    print("\n[main] Finished.")
    print(f" Summary CSV: {Path(OUTDIR) / 'summary_prevalence.csv'}")
    print(f" Quick matrix: {SUMMARY_TSV_QUICK}")
    print(f" Genome->image map: {GENOME_IMAGE_MAP}")
    print(f" Heatmap images: {HEATMAP_PREFIX}_1..{num_images}.png")

if __name__ == "__main__":
    main()