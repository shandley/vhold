# Case Study 3: Metagenomic Dark Matter

Demonstrates vHold's ability to annotate truly novel viral proteins discovered from metagenomes.

## Background

Metagenomic sequencing has revealed vast viral diversity, but 40-70% of predicted proteins have no detectable homologs in reference databases. These "dark matter" proteins cannot be annotated by BLAST/DIAMOND. Structure-based search offers a solution because protein structure is more conserved than sequence.

## Data Source: palmdb

[palmdb](https://github.com/ababaian/palmdb) contains 513,176 RNA-dependent RNA polymerase (RdRp) "palmprint" sequences discovered from petabase-scale mining of the Sequence Read Archive.

**Key features**:
- Discovered using RdRp-specific HMMs (ground truth: these ARE RdRps)
- Many sequences have no close match in GenBank/UniProt
- Palmprints are ~100 aa conserved core sequences containing catalytic motifs A, B, C
- Source: Edgar et al., Nature 2022

## Test Dataset

30 diverse palmprint sequences selected from palmdb:

| Metric | Value |
|--------|-------|
| Total sequences | 30 |
| Length range | 90-141 aa |
| Mean length | 109 aa |
| Expected function | RNA-dependent RNA polymerase |
| Expected category | replication |

## Experimental Design

### Hypothesis

vHold can identify RdRp structural homologs for novel metagenomic sequences that have no BLAST hits.

### Success Criteria

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Annotation rate | >80% | vHold finds hits for most sequences |
| Correct category | >70% | Classified as "replication" |
| RdRp keywords | >50% | Description contains polymerase/RdRp |
| Novelty level | remote/twilight | Hits at <50% identity (true discovery) |

### Controls

1. **Positive control**: All sequences ARE RdRps (discovered by RdRp HMM)
2. **Novelty validation**: Check hit identity - expect <50% (not in database)
3. **Structural validation**: RdRp palm domain fold is conserved

## Running the Case Study

```bash
cd case_studies/metagenomic_dark_matter

# Select test sequences (already done)
python select_test_sequences.py

# Run vHold annotation
vhold run -i test_palmprints.fasta -o results/ -t 4 --device cpu

# Analyze results
python analyze_results.py results/
```

## Expected Results

### If vHold Works Well

```
Annotation rate: 25-30/30 (83-100%)
Category accuracy: 20-25/30 (67-83%) as "replication"
Novelty distribution:
  - twilight_zone (<30%): ~60%
  - remote_homolog (30-70%): ~30%
  - close_homolog (>70%): ~10%
```

### If vHold Fails

```
Annotation rate: <50%
Many "no hits" - sequences too divergent
Or: Hits but wrong category (not RdRp)
```

## Files

| File | Description |
|------|-------------|
| `sotus.palmprint.faa` | Full palmdb dataset (513K sequences) |
| `label_sotu.tsv` | Source labels for palmdb sequences |
| `test_palmprints.fasta` | 30 selected test sequences |
| `ground_truth.json` | Expected annotations |
| `select_test_sequences.py` | Sequence selection script |
| `analyze_results.py` | Results analysis script |
| `results/` | vHold output directory |

## Interpretation Guide

### High Annotation Rate + Correct Category

vHold successfully transfers RdRp function from structural homologs. This demonstrates the value of structure-based annotation for metagenomic dark matter.

### High Annotation Rate + Wrong Category

vHold finds hits but misclassifies them. May indicate:
- Keyword gaps in functional categories
- Hits to non-RdRp proteins with similar folds
- Need to improve category classification

### Low Annotation Rate

Sequences are too divergent even for structural search. May indicate:
- Truly novel protein folds
- palmprints are too short for reliable structure prediction
- Database gaps

## References

1. Edgar RC, et al. (2022). Petabase-scale sequence alignment catalyses viral discovery. *Nature* 602:142-145.
2. Babaian A, Edgar RC (2022). Ribovirus classification by a polymerase barcode sequence. *PeerJ* 10:e14055.
3. Wolf YI, et al. (2018). Origins and Evolution of the Global RNA Virome. *mBio* 9:e02329-18.
