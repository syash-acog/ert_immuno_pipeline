# Immunogenicity Pipeline

A computational pipeline for predicting immunogenicity risk of therapeutic proteins. This tool analyzes MHC-II binding, cleavage likelihood, structural accessibility, and T-cell receptor facing residues to identify potential immunogenic hotspots.

## 🧬 Overview

This pipeline helps biologists and protein engineers:
- Identify immunogenic epitopes in therapeutic proteins
- Predict MHC-II binding affinity for multiple HLA alleles
- Map risk scores to individual residues for targeted de-immunization
- Generate population-weighted risk assessments

## 📋 Requirements

- Docker (recommended) OR Python 3.12
- Protein sequence in FASTA format
- (Optional) PDB structure file for accessibility analysis

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/ert_immuno_pipeline.git
cd ert_immuno_pipeline

# 2. Download ImmuScope model weights from Zenodo
# The data and model weights are available at: https://doi.org/10.5281/zenodo.14184201
# Download and extract to the tools/ImmuScope directory:

tar -xvzf ImmuScope-data.tar.gz -C tools/ImmuScope/
tar -xvzf ImmuScope-weights.tar.gz -C tools/ImmuScope/

# 3. Add your input files
# Place your protein FASTA file in data/input/
# (Optional) Place your PDB structure file in data/input/
cp /path/to/your/protein.fasta data/input/
cp /path/to/your/structure.pdb data/input/  # Optional

# 4. Build the Docker image
make build

# 5. Run the pipeline
docker run --rm \
  -v $(pwd)/data/input:/app/data/input \
  -v $(pwd)/data/output:/app/data/output \
  immuno-pipeline \
  --fasta /app/data/input/your_protein.fasta \
  --pdb /app/data/input/structure.pdb \
  --alleles "HLA-DRB1*03:01,HLA-DRB1*01:01"

# 6. Check results in data/output/
```

### Using Python Directly

```bash
# 1. Clone and setup

# 2. Download ImmuScope data/weights (see Zenodo link above)

# 3. Add your input files to data/input/

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the pipeline
python src/main.py \
  --fasta data/input/protein.fasta \
  --pdb data/input/structure.pdb \
  --alleles "HLA-DRB1*03:01,HLA-DRB1*01:01"
```

## 📖 Usage

### Command Line Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--fasta` | `-f` | ✅ | Path to input FASTA file |
| `--pdb` | `-p` | ❌ | Path to PDB structure file |
| `--alleles` | `-a` | ❌ | Comma-separated HLA alleles (default: HLA-DRB1*03:01,HLA-DRB1*01:01) |
| `--output` | `-o` | ❌ | Output directory (default: data/output) |
| `--peptide-length` | `-l` | ❌ | Peptide length (default: 15) |

### Supported Alleles

The pipeline currently supports **HLA-DRB1** alleles only. Examples:
- `HLA-DRB1*01:01`
- `HLA-DRB1*03:01`
- `HLA-DRB1*04:01`
- `HLA-DRB1*07:01`
- `HLA-DRB1*11:01`
- `HLA-DRB1*13:01`
- `HLA-DRB1*15:01`

> ⚠️ **Note**: HLA-DQ and HLA-DP alleles are not supported by the current TEPITOPE method.

### Input File Formats

#### FASTA File
```
>Protein_Name
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIA
```

#### PDB File (Optional)
Standard PDB format. If not provided, the pipeline will use default accessibility values.

## 📊 Output Files

The pipeline generates the following files in the output directory:

| File | Description |
|------|-------------|
| `rav-pal_enumeration.csv` | All enumerated 15-mer peptides with start positions |
| `rav-pal_cleavage.csv` | Cleavage likelihood scores from NetCleave |
| `rav-pal_structural_accessibility.csv` | Solvent accessible surface area (SASA) values |
| `rav-pal_mhc_binding.csv` | MHC-II binding predictions for all alleles |
| `rav-pal_register_resolution.csv` | Binding core, anchor positions, and TCR-facing residues |
| `rav-pal_immunogenicity_score.csv` | ImmuScope immunogenicity predictions per peptide |
| `rav-pal_residue_risk.csv` | Per-residue risk scores with contributing peptides |
| `rav-pal_protein_risk.csv` | Final aggregated protein risk score |


## 🔬 Pipeline Steps

1. **Peptide Enumeration**: Generate overlapping peptides from protein sequence
2. **Cleavage Likelihood**: Predict proteasomal/lysosomal cleavage sites (NetCleave)
3. **Structural Accessibility**: Calculate solvent-accessible surface area (FreeSASA)
4. **MHC Binding**: Predict MHC-II binding affinity (TEPITOPE)
5. **Register Resolution**: Identify 9-mer binding core and TCR-facing residues
6. **Immunogenicity Scoring**: Predict T-cell response probability (ImmuScope)
7. **Residue Risk Mapping**: Aggregate scores to protein coordinates
8. **Population Weighting**: Calculate final weighted risk score

## 🐳 Docker Commands

```bash
# Build image
make build

# Run with example data
make run

# Show help
make help

# Clean up
make clean
```

## 📁 Project Structure

```
ert_immuno_pipeline/
├── src/
│   ├── main.py              # Entry point with CLI
│   ├── pipeline.py          # Pipeline orchestration
│   └── steps/               # Individual analysis steps
├── tools/
│   ├── NetCleave/           # Cleavage prediction tool
│   └── ImmuScope/           # Immunogenicity prediction tool
├── data/
│   ├── input/               # Place your input files here
│   └── output/              # Results will be saved here
├── Dockerfile
├── Makefile
└── requirements.txt
```


## 📚 References

- **TEPITOPE**:  [Epitopepredict GitHub Repository](https://github.com/dmnfarrell/epitopepredict)
- **NetCleave**: [NetCleave GitHub Repository](https://github.com/pepamengual/NetCleave)
- **ImmuScope**: [ImmuScope GitHub Repository](https://github.com/RamuLab/ImmuScope)
- **FreeSASA**: [FreeSASA GitHub Repository](https://github.com/mittinatten/freesasa)
