# Case Study 2: Remote Homology Discovery

## Purpose

Demonstrate vHold's ability to annotate divergent viral proteins where sequence-based methods fail. This is vHold's core value proposition: **structure is more conserved than sequence**.

## The Challenge

Sequence-based annotation fails when:
- Sequence identity drops below ~30% ("twilight zone")
- BLAST/DIAMOND E-values become insignificant
- Viral proteins evolve too rapidly for sequence detection

Structure-based annotation succeeds because:
- Protein folds are 3-10× more conserved than sequences
- Foldseek can detect structural similarity at <20% sequence identity
- ProstT5 predicts structure from sequence alone (no MSA needed)

## Key Insight: We Already Have Identity Data

Foldseek reports sequence identity (`fident`) for every hit. We don't need to run BLAST - we can simply **stratify results by identity** to prove the point:

```
Sequence Identity    Detection Method
─────────────────    ────────────────
> 50%                BLAST works fine
30-50%               BLAST unreliable
20-30%               BLAST fails, structure works
< 20%                "Twilight zone" - structure only
```

## Experimental Design

### Approach 1: Identity-Stratified Analysis

For any vHold run, analyze hits by sequence identity bins:

```python
# Pseudocode
bins = {
    "easy": (0.5, 1.0),      # BLAST would work
    "moderate": (0.3, 0.5),   # BLAST marginal
    "remote": (0.2, 0.3),     # BLAST fails
    "twilight": (0.0, 0.2),   # Structure only
}

for protein in results:
    for hit in protein.hits:
        bin = get_bin(hit.fident)
        if hit.has_annotation:
            counts[bin]["annotated"] += 1
```

**Success metric**: Significant annotation rate in "remote" and "twilight" bins.

### Approach 2: Divergent Virus Panel

Select proteins from rapidly-evolving virus families:

| Virus Family | Mutation Rate | Expected Identity |
|--------------|---------------|-------------------|
| RNA phages | Very high | <30% |
| Insect viruses | High | 20-40% |
| Giant viruses | Moderate | 30-50% |
| Plant viruses | Moderate | 30-50% |

### Approach 3: Synthetic Holdout

1. Take Viro3D proteins with Pfam annotations
2. Find proteins with NO close homologs in BFVD (<30% identity)
3. Test if vHold can still recover their function via structure

## Implementation Plan

### Phase 1: Analyze Existing Data

Extract identity-stratified statistics from any vHold run:

```bash
# From foldseek hits, calculate identity distribution
cut -f1,2,3 results/foldseek/bfvd_hits.tsv | \
  awk -F'\t' '{
    if ($3 < 0.2) print "twilight"
    else if ($3 < 0.3) print "remote"
    else if ($3 < 0.5) print "moderate"
    else print "easy"
  }' | sort | uniq -c
```

### Phase 2: Create Divergent Virus Dataset

Sources for divergent proteins:
1. **Serratus palmdb** - Novel RNA viruses from metagenomes
2. **IMG/VR** - Uncultivated viral genomes
3. **Viro3D outliers** - Proteins distant from cluster centers

### Phase 3: Controlled Benchmark

Create gold standard with:
- Known function (Pfam/GO annotation)
- Low sequence identity to database (<30%)
- Sufficient structural quality (pLDDT > 70)

## Expected Results

### Hypothesis

vHold can annotate proteins at <30% sequence identity where BLAST would return no significant hits.

### Success Criteria

| Identity Bin | BLAST Expected | vHold Target |
|--------------|----------------|--------------|
| >50% | >90% | >90% |
| 30-50% | ~50% | >80% |
| 20-30% | <10% | >50% |
| <20% | ~0% | >30% |

### Key Figures

1. **Annotation rate vs. sequence identity** - Show vHold maintains performance at low identity
2. **Identity distribution of successful annotations** - Histogram showing annotations at <30%
3. **Example proteins** - Specific cases where structure succeeded, sequence failed

## Analysis Script

```python
#!/usr/bin/env python3
"""Analyze vHold results stratified by sequence identity."""

import pandas as pd
from pathlib import Path

def analyze_identity_distribution(results_dir: Path):
    """Stratify hits by sequence identity."""

    # Load foldseek hits
    bfvd = pd.read_csv(results_dir / "foldseek/bfvd_hits.tsv",
                       sep="\t", header=None,
                       names=["query", "target", "fident", ...])

    # Define bins
    bins = [0.0, 0.2, 0.3, 0.5, 1.0]
    labels = ["twilight", "remote", "moderate", "easy"]

    bfvd["identity_bin"] = pd.cut(bfvd["fident"], bins=bins, labels=labels)

    # Count annotations per bin
    annotated = bfvd[bfvd["has_annotation"]]

    for bin_name in labels:
        bin_hits = annotated[annotated["identity_bin"] == bin_name]
        print(f"{bin_name}: {len(bin_hits)} annotated hits")

    return bfvd
```

## Data Requirements

### Input Proteins

For a compelling demonstration, need proteins that are:
1. **Functionally characterized** - Known Pfam domains or GO terms
2. **Structurally predictable** - pLDDT > 70 for reliable 3Di
3. **Sequence-divergent** - <30% identity to closest database entry

### Ground Truth

| Field | Source |
|-------|--------|
| True function | Pfam domain assignment |
| True category | GO biological process |
| Sequence identity | Foldseek fident to best hit |
| Structure quality | ColabFold/ESMFold pLDDT |

## Comparison to BLAST (Optional)

If explicit BLAST comparison desired:

```bash
# Run DIAMOND (faster BLAST)
diamond blastp \
  -d viral_proteins \
  -q query.fasta \
  -o blast_hits.tsv \
  --very-sensitive \
  -e 1e-5

# Compare: proteins with vHold hits but no BLAST hits
comm -23 <(cut -f1 vhold_annotated.txt | sort) \
         <(cut -f1 blast_hits.txt | sort) > vhold_only.txt
```

## Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Implement identity stratification analysis | 1 day |
| 2 | Curate divergent virus panel | 2-3 days |
| 3 | Run benchmark and analyze | 1 day |
| 4 | Document results | 1 day |

## References

1. Holm L (2020). Using Dali for Protein Structure Comparison. *Methods Mol Biol* 2112:29-42.
2. van Kempen M, et al. (2023). Fast and accurate protein structure search with Foldseek. *Nat Biotechnol*.
3. Edgar RC (2022). Serratus: Massively-parallel pandemic virus detection. *Nature* 602:142-145.
