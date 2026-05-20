import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- PARAMETERS ---
FILE = "/Users/anirudhjakhmola/Documents/Codes/Prophage_Detection_Code/Breath_vs_Identity/phiMMP01_summary.csv"  # your input file
COVERAGE_THRESHOLD = 80
IDENTITY_THRESHOLD = 95
OUTPUT = "breadth_vs_identity.png"
# ------------------

# Load data
df = pd.read_csv(FILE)

# Define whether prophage is considered present
df["Prophage_present"] = (df["coverage"] >= COVERAGE_THRESHOLD) & (df["identity"] >= IDENTITY_THRESHOLD)
print(f"Number of genomes crossing threshold: {df['Prophage_present'].sum()}")

# Plot style
sns.set(style="whitegrid", context="talk")

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="coverage", y="identity",
    hue="Prophage_present",
    palette={True: "orange", False: "gray"},
    alpha=0.7,
    edgecolor="none"
)

# Add threshold lines
plt.axvline(COVERAGE_THRESHOLD, color="black", linestyle="--", linewidth=1)
plt.axhline(IDENTITY_THRESHOLD, color="black", linestyle="--", linewidth=1)

# Labels & styling
plt.xlabel("Prophage alignment breadth (%)", fontsize=14)
plt.ylabel("Nucleotide identity (%)", fontsize=14)
plt.title("Distribution of prophage alignment breadth and identity", fontsize=16, pad=15)
plt.legend(title="Genomes detected", loc="lower right", frameon=True)
plt.tight_layout()

# Save
plt.savefig(OUTPUT, dpi=600)
plt.show()

# Second graph with no text

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="coverage", y="identity",
    hue="Prophage_present",
    palette={True: "orange", False: "gray"},
    alpha=0.7,
    edgecolor="none"
)

# Add threshold lines
plt.axvline(COVERAGE_THRESHOLD, color="black", linestyle="--", linewidth=1)
plt.axhline(IDENTITY_THRESHOLD, color="black", linestyle="--", linewidth=1)

# Labels & styling
plt.xlabel(".", fontsize=14)
plt.ylabel(".", fontsize=14)
plt.title(".", fontsize=16, pad=15)
ax = plt.gca()  # or use the 'ax' variable returned by sns.scatterplot
legend = ax.get_legend()
if legend is not None:
    legend.remove()
plt.tight_layout()

# Save
plt.savefig(OUTPUT, dpi=600)
plt.show()

