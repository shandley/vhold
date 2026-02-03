# Case Study 4: crAssphage ORFans

Demonstrates vHold's ability to annotate unknown proteins from crAssphage, the most abundant virus in the human gut.

## Background

crAssphage (Carjivirus communis) was discovered in 2014 through cross-assembly of human fecal metagenomes. Despite being extraordinarily abundant (present in >70% of humans, comprising up to 40% of gut viral reads), it remained uncultured for years and ~86% of its proteins were annotated as "hypothetical."

The 2023 Nature paper "Structural atlas of a human gut crassvirus" provided cryo-EM structures and mass spectrometry characterization of many proteins, giving us ground truth for evaluation.

## Data Source

- **Genome**: [NC_024711.1](https://www.ncbi.nlm.nih.gov/nuccore/NC_024711.1) (Carjivirus communis)
- **Size**: 97 kb dsDNA
- **Total proteins**: 80 (69 hypothetical = 86% ORFans)
- **Test set**: 37 proteins (diverse selection)

## Test Dataset

| Category | Count | Description |
|----------|-------|-------------|
| Structural proteins | 8 | Capsid, tail (cryo-EM confirmed) |
| Replication proteins | 6 | Helicase, polymerase, etc. |
| Transcription proteins | 3 | Virion RNA polymerase complex |
| Nuclease proteins | 3 | DNA processing enzymes |
| True ORFans | 13 | No known function |
| Other | 4 | Cargo proteins, BACON domain |
| **Total** | **37** | |

## Experimental Design

### Hypothesis

vHold can identify structural homologs for crAssphage ORFans, particularly:
1. Capsid proteins should match other phage capsid structures
2. Polymerase complex should match known RNA polymerases
3. True ORFans may reveal unexpected homologies

### Success Criteria

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Annotation rate | >60% | vHold finds hits for most proteins |
| Category accuracy | >50% | Correct functional category assignment |
| Structural proteins | >70% | Capsid/tail identified as "structural" |
| ORFan discovery | Any | New insights for previously unknown proteins |

### Key Proteins to Watch

| Protein | NCBI Annotation | Literature Function | Interest |
|---------|-----------------|---------------------|----------|
| gp74 | hypothetical | major capsid protein | Should find capsid homologs |
| gp50 | hypothetical | virion RNA polymerase | Unique phage RNAP |
| gp64 | hypothetical | BACON domain protein | Unique to crAssphage |
| gp61 | hypothetical | portal protein | Packaging machinery |
| gp40 | hypothetical | unknown (1957 aa) | Largest ORFan |

## Running the Case Study

```bash
cd case_studies/crass_phage_orfans

# Prepare dataset (already done)
python prepare_dataset.py

# Run vHold annotation
vhold run -i crass_orfans.fasta -o results/ -t 4 --device cpu

# Analyze results
python analyze_results.py
```

## Expected Results

### If vHold Works Well

```
Annotation rate: 60-80%
Structural proteins: Correctly identified as structural/capsid
RNA polymerase: Matches to known RNAP structures
ORFan discoveries: Novel homologies for some hypothetical proteins
```

### Challenging Cases

- **gp40** (1957 aa): Very large protein, may be slow to process
- **gp64**: BACON domains are crAssphage-specific, may not find homologs
- **Small ORFans**: Short hypothetical proteins may lack structural signal

## References

1. Dutilh BE, et al. (2014). A highly abundant bacteriophage discovered in the unknown sequences of human faecal metagenomes. *Nature Communications* 5:4498.

2. Yutin N, et al. (2018). Discovery of an expansive bacteriophage family that includes the most abundant viruses from the human gut. *Nature Microbiology* 3:38-46.

3. Structural atlas of a human gut crassvirus. *Nature* (2023). DOI: 10.1038/s41586-023-06019-2

4. Shkoporov AN, et al. (2018). ΦCrAss001 represents the most abundant bacteriophage family in the human gut. *Nature Microbiology* 3:1168-1177.

5. Drobysheva AV, et al. (2021). Structure and function of virion RNA polymerase of a crAss-like phage. *Nature* 589:306-309.

## Files

| File | Description |
|------|-------------|
| `crass_proteins_raw.fasta` | All 80 crAssphage proteins from NCBI |
| `crass_orfans.fasta` | 37 selected test proteins |
| `ground_truth.json` | Expected annotations from literature |
| `prepare_dataset.py` | Dataset preparation script |
| `analyze_results.py` | Results analysis script |
| `results/` | vHold output directory |
