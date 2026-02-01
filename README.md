# vhold

Viral protein annotation using structural homology.

vhold annotates viral proteins by comparing predicted 3D structures against BFVD and Viro3D reference databases using ProstT5 and Foldseek.

## Installation

```bash
pip install -e .
```

### External Dependencies

- **Foldseek**: Required for structural similarity search
  ```bash
  conda install -c conda-forge -c bioconda foldseek
  ```

## Quick Start

```bash
# 1. Install reference databases
vhold install

# 2. Run the full annotation pipeline
vhold run -i proteins.fasta -o results/ -t 4

# 3. Check results
cat results/vhold_results.tsv
```

## Commands

### `vhold install`

Download and install BFVD and Viro3D reference databases.

```bash
vhold install                    # Install all databases
vhold install --no-viro3d        # Install only BFVD
vhold install -d /custom/path    # Custom database location
```

### `vhold predict`

Predict 3Di structural sequences using ProstT5.

```bash
vhold predict -i proteins.fasta -o predictions/
```

### `vhold compare`

Search 3Di predictions against reference databases.

```bash
vhold compare -p predictions/ -o search_results/
```

### `vhold run`

Run the complete annotation pipeline.

```bash
vhold run -i proteins.fasta -o results/ -t 4
```

## Output Files

- `vhold_results.tsv` - Main annotation results
- `vhold_summary.json` - Summary statistics
- `predictions/` - 3Di sequence predictions
- `foldseek/` - Raw Foldseek search results

## Databases

- **BFVD**: Bacterial and Fungal Virus Database
- **Viro3D**: Comprehensive viral protein structure database

## License

MIT
