# Case Study 5: Eukaryotic Virus Proteins

Demonstrates vHold's ability to annotate proteins from **mammalian viruses** - the key capability that distinguishes vHold from pHold (which targets phages).

## Background

While pHold excels at phage protein annotation, many important viruses infect eukaryotic hosts:
- Emerging zoonotic viruses (Nipah, Hendra, Ebola)
- Vaccine-preventable diseases (Measles, Rabies)
- Pandemic threats (Coronaviruses, Flaviviruses)

This case study tests vHold on proteins from multiple mammalian virus families.

## Virus Families Included

| Family | Viruses | Genome | Proteins |
|--------|---------|--------|----------|
| **Paramyxoviridae** | Nipah, Hendra, Measles | (-) ssRNA | 11 |
| **Filoviridae** | Ebola | (-) ssRNA | 6 |
| **Rhabdoviridae** | Rabies, VSV | (-) ssRNA | 5 |
| **Hantaviridae** | Hantaan | (-) ssRNA | 2 |
| **Togaviridae** | Chikungunya | (+) ssRNA | 1 |
| **Coronaviridae** | MERS-CoV | (+) ssRNA | 2 |

## Test Dataset

| Metric | Value |
|--------|-------|
| Total proteins run | 27 (of 29 total; 2 skipped) |
| Virus families | 7 |
| Length range | 166-1248 aa |
| Structural proteins | 18 |
| Regulatory/accessory | 5 |
| Replication | 2 |
| Skipped (CPU scaling) | MERS spike (7078aa), Dengue polyprotein (3478aa) |

## Results Summary (2026-02-21)

Pipeline: triage + Foldseek + MLP classifier + disorder (CPU, `--triage --fast --device cpu`). LLM classification not active (no API key). STARLING search not available. MPS failed on 1248aa Chikungunya protein (OOM).

| Metric | Value |
|--------|-------|
| Annotated | 27/27 (100%) |
| Dark matter | 0 |
| Overall accuracy | 21/27 (77.8%) |
| Annotation source | All 27 via embedding triage (Viro3D) |
| Mean consensus score | 0.999 |

**Misclassifications (6):**
| Protein | Virus | Ground Truth | Predicted | Issue |
|---------|-------|:---:|:---:|-------|
| Nipah V | Nipah | host_interaction | regulatory | Category boundary (interferon antagonist) |
| Measles V | Measles | host_interaction | regulatory | Category boundary (interferon antagonist) |
| Ebola VP30 | Ebola | regulatory | replication | Category boundary (transcription activator) |
| Rabies G | Rabies | structural | host_interaction | GO term override |
| Chikungunya SP | CHIKV | structural | host_interaction | GO term override |
| MERS-CoV M | MERS | structural | unknown | DB curation issue |

**Key observations:**
- All 27 proteins matched via embedding triage — no decoder/Foldseek needed
- Misclassifications are category boundary issues and GO term overrides, not structural search failures
- V/C proteins classified as "regulatory" instead of "host_interaction" — LLM reclassification expected to resolve
- MPS OOM on 1248aa protein; use `--device cpu` for large eukaryotic virus proteins

Previous results archived to `_archive_2026-02-21/`.

## Skipped Proteins

Two proteins were excluded due to CPU performance constraints:

| Protein | Virus | Size | Est. Time (CPU) | Reason |
|---------|-------|------|-----------------|--------|
| YP_009047202.1 | MERS-CoV | 7078aa | Days | O(n^2) scaling makes this impractical on CPU |
| NP_059433.1 | Dengue | 3478aa | ~12-24 hours | Also has corrupted sequence data (API rate limit error) |

These would require GPU (CUDA or MPS) to process in reasonable time.

## Running the Case Study

```bash
cd case_studies/eukaryotic_viruses

# Run on the 27-protein subset (auto-selects MPS on Apple Silicon)
uv run vhold run -i test_proteins_small.fasta -o results/ -t 8

# Force CPU if needed
uv run vhold run -i test_proteins_small.fasta -o results/ -t 8 --device cpu

# Analyze results
python analyze_results.py results/
```

### With GPU (for all 29 proteins)
```bash
# CUDA (Linux/Windows)
uv run vhold run -i test_proteins.fasta -o results/ --device cuda

# MPS (Apple Silicon) - auto-selected with --device auto
uv run vhold run -i test_proteins.fasta -o results/
```

## Files

| File | Description |
|------|-------------|
| `all_eukaryotic_proteins.fasta` | Raw proteins from 19 virus genomes |
| `test_proteins.fasta` | 29 curated test proteins (full set) |
| `test_proteins_small.fasta` | 27 proteins excluding large ones |
| `ground_truth.json` | Expected annotations from literature |
| `remaining_proteins.fasta` | 4 remaining large proteins |
| `fetch_proteins.py` | NCBI protein fetcher |
| `prepare_test_set.py` | Test set curation script |
| `analyze_results.py` | Results analysis script |
| `results/` | vHold output directory |

## Recommendations

1. **Use GPU for proteins >1000aa** - CPU scaling is quadratic
2. **Process polyproteins as individual proteins** - Split before annotation
3. **Use `--triage` with `--llm-classify`** - Embedding triage + LLM reclassification expected to improve both speed and accuracy for eukaryotic viruses

## References

1. Wang LF, et al. (2000). The exceptionally large genome of Hendra virus. Virology 278:587-600.
2. Chua KB, et al. (2000). Nipah virus: a recently emergent deadly paramyxovirus. Science 288:1432-1435.
3. Feldmann H, et al. (2020). Ebola virus disease. Nat Rev Dis Primers 6:13.
4. Rupprecht CE, et al. (2002). Rabies re-examined. Lancet Infect Dis 2:327-343.
