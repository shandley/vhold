# vHold: Supplementary Materials

## S1. Algorithm Pseudocode

### S1.1 Main Pipeline

```
ALGORITHM vHold_Pipeline
INPUT: proteins (FASTA file), databases (BFVD, Viro3D)
OUTPUT: annotations (TSV), summary (JSON), dark_matter (TSV)

1. sequences ← parse_fasta(proteins)
2. lengths ← {seq.id: len(seq) for seq in sequences}

3. // ProstT5 3Di Prediction
4. model ← load_prostt5()
5. predictions ← []
6. FOR batch IN batch_sequences(sequences, batch_size):
7.     formatted ← ["<AA2fold> " + space_join(seq) for seq in batch]
8.     tokens ← tokenize(formatted)
9.     outputs ← model.generate(tokens, return_scores=True)
10.    FOR i, output IN enumerate(outputs):
11.        3di ← decode_3di(output.sequences)
12.        confidence ← extract_confidence(output.scores)
13.        predictions.append(Prediction(id, seq, 3di, confidence))

14. // Confidence Masking
15. FOR pred IN predictions:
16.    pred.3di_masked ← mask_low_confidence(pred.3di, pred.confidence, threshold=0.7)

17. // Foldseek Search
18. query_db ← create_foldseek_db(predictions)
19. all_hits ← []
20. FOR db IN [BFVD, Viro3D]:
21.    hits ← foldseek_search(query_db, db, evalue=1e-3, sensitivity=9.5)
22.    FOR hit IN hits:
23.        hit.source_db ← db.name
24.    all_hits.extend(hits)

25. // Annotation Transfer
26. annotations ← load_database_annotations()
27. uniprot_ids ← collect_uniprot_ids(all_hits, "bfvd")
28. uniprot_cache ← batch_fetch_uniprot(uniprot_ids)

29. // Consensus Scoring
30. results ← {}
31. FOR query_id IN lengths.keys():
32.    query_hits ← group_by_database(filter_by_query(all_hits, query_id))
33.    scored_hits ← {}
34.    FOR db, hits IN query_hits:
35.        FOR hit IN hits:
36.            ann ← get_annotation(hit, annotations, uniprot_cache)
37.            score ← calculate_hit_quality(hit)
38.            weighted ← score × DATABASE_WEIGHT[db] × functional_bonus(ann)
39.            scored_hits[db].append(HitScore(hit, score, weighted, ann))
40.
41.    consensus ← build_consensus(query_id, lengths[query_id], scored_hits)
42.    consensus.category ← classify_protein(consensus.description, consensus.gene)
43.    results[query_id] ← consensus

44. // Dark Matter Analysis
45. dark_matter ← []
46. FOR result IN results.values():
47.    IF is_dark_matter(result):
48.        dark_matter.append(classify_dark_matter(result))

49. // Output Generation
50. write_consensus_tsv(results)
51. write_summary_json(results, dark_matter)
52. IF dark_matter.count > 0:
53.    write_dark_matter_tsv(dark_matter)

54. RETURN results
```

### S1.2 Consensus Scoring Algorithm

```
ALGORITHM build_consensus
INPUT: query_id, query_length, scored_hits (dict of db → list of HitScore)
OUTPUT: ConsensusResult

1. result ← ConsensusResult(query_id, query_length)

2. // Get best hit per database
3. best_by_db ← {}
4. FOR db, scores IN scored_hits:
5.     IF scores not empty:
6.         best_by_db[db] ← max(scores, key=weighted_score)

7. IF best_by_db is empty:
8.     result.confidence_level ← "none"
9.     result.agreement ← "none"
10.    RETURN result

11. // Sort databases by best hit score
12. sorted_dbs ← sort(best_by_db.keys(), by=best_by_db[db].weighted_score, descending=True)

13. // Primary hit (best overall)
14. primary_db ← sorted_dbs[0]
15. primary ← best_by_db[primary_db]
16. result.primary_hit ← primary.hit
17. result.primary_annotation ← primary.annotation
18. result.primary_source ← primary_db

19. // Secondary hit (if available)
20. IF len(sorted_dbs) > 1:
21.    secondary_db ← sorted_dbs[1]
22.    secondary ← best_by_db[secondary_db]
23.    result.secondary_hit ← secondary.hit
24.    result.secondary_annotation ← secondary.annotation
25.    result.secondary_source ← secondary_db
26.
27.    // Check agreement
28.    result.agreement ← check_annotation_agreement(primary.annotation, secondary.annotation)
29.
30.    // Calculate consensus score with agreement bonus
31.    base_score ← primary.weighted_score
32.    IF result.agreement = "agree":
33.        result.consensus_score ← min(1.0, base_score × 1.3)
34.    ELSE IF result.agreement = "partial":
35.        result.consensus_score ← min(1.0, base_score × 1.15)
36.    ELSE:
37.        result.consensus_score ← base_score
38. ELSE:
39.    result.agreement ← "single"
40.    result.consensus_score ← primary.weighted_score

41. // Assign confidence level
42. IF result.consensus_score ≥ 0.8:
43.    result.confidence_level ← "high"
44. ELSE IF result.consensus_score ≥ 0.5:
45.    result.confidence_level ← "medium"
46. ELSE IF result.consensus_score ≥ 0.3:
47.    result.confidence_level ← "low"
48. ELSE:
49.    result.confidence_level ← "very_low"

50. // Upgrade confidence for strong agreement
51. IF result.agreement = "agree" AND result.confidence_level IN ["medium", "low"]:
52.    result.confidence_level ← upgrade_one_level(result.confidence_level)

53. RETURN result
```

### S1.3 Hit Quality Calculation

```
ALGORITHM calculate_hit_quality
INPUT: hit (FoldseekHit)
OUTPUT: quality (float, 0-1)

1. // E-value contribution (0-1)
2. // E-value 1e-50 → 1.0, E-value 1e-10 → 0.2, E-value 1 → 0
3. IF hit.evalue > 0:
4.     evalue_score ← min(1.0, -log10(hit.evalue) / 50)
5. ELSE:
6.     evalue_score ← 1.0

7. // Identity (already 0-1)
8. identity_score ← hit.fident

9. // Coverage (already 0-1)
10. coverage_score ← hit.qcov

11. // Weighted combination
12. quality ← 0.5 × evalue_score + 0.3 × identity_score + 0.2 × coverage_score

13. RETURN quality
```

## S2. Database Descriptions

### S2.1 BFVD (Big Fantastic Virus Database)

**Source**: https://bfvd.foldseek.com/

**Contents**:
- 351,242 viral protein structures
- AlphaFold2-predicted structures with confidence metrics
- Pre-computed 3Di sequences for Foldseek compatibility

**Metadata Fields**:
| Field | Description |
|-------|-------------|
| uniprot_id | UniProt accession (primary key) |
| structure_id | Internal structure identifier |
| plddt | Mean predicted LDDT score (0-100) |
| ptm | Predicted TM-score |
| source | Structure prediction source |

**Annotation Retrieval**:
UniProt accessions are used to query the UniProt REST API for functional annotations:
```
GET https://rest.uniprot.org/uniprotkb/{accession}?fields=accession,protein_name,gene_names,organism_name
```

### S2.2 Viro3D

**Source**: https://viro3d.cvr.gla.ac.uk/

**Contents**:
- 85,162 high-confidence viral protein structures
- Coverage: 4,400+ virus species
- ColabFold-predicted structures

**Metadata Fields**:
| Field | Description |
|-------|-------------|
| Viro3D ID | Internal identifier (e.g., QHD43423.2_10195) |
| GenBank Protein ID | NCBI protein accession |
| ICTV Species | Viral taxonomy |
| Viro3D Name | Parsed gene/product name |
| UniProt ID | UniProt accession (if available) |
| Protein Length | Amino acid sequence length |

**Target ID Normalization**:
Foldseek returns target IDs with prefixes and suffixes that must be normalized:
```
Foldseek returns: CF-QHD43423.2_10195_relaxed
Metadata contains: QHD43423.2_10195
Normalization: Remove [CE]F- prefix and _relaxed/_unrelaxed suffix
```

### S2.3 Combined Database Statistics

| Metric | BFVD | Viro3D | Combined |
|--------|------|--------|----------|
| Total structures | 351,242 | 85,162 | 436,404 |
| Unique proteins | ~351,000 | ~85,000 | ~420,000* |
| Disk size | 533.8 MB | 568.7 MB | 1.1 GB |
| Coverage | Primarily phages | All viral taxa | Comprehensive |

*Some overlap exists between databases

## S3. Functional Category Keywords

### S3.1 Complete Keyword Lists

```python
FUNCTIONAL_CATEGORIES = {
    "structural": [
        "capsid", "coat", "envelope", "spike", "matrix", "tail",
        "fiber", "portal", "shell", "head", "virion", "glycoprotein",
        "membrane protein", "nucleocapsid", "tegument", "baseplate"
    ],
    "replication": [
        "polymerase", "replicase", "helicase", "primase", "rdrp",
        "reverse transcriptase", "rep protein", "nsp12", "nsp13"
    ],
    "protease": [
        "protease", "peptidase", "maturase", "3cl", "mpro", "nsp5"
    ],
    "nuclease": [
        "nuclease", "endonuclease", "exonuclease", "integrase",
        "recombinase", "ligase", "rnase", "dnase"
    ],
    "packaging": [
        "terminase", "packaging", "scaffold", "portal protein"
    ],
    "regulatory": [
        "transcription", "repressor", "activator", "regulator",
        "anti-repressor", "antirepressor"
    ],
    "movement": [
        "movement", "cell-to-cell", "transport protein"
    ],
    "lysis": [
        "lysin", "holin", "endolysin", "spanin", "lysis"
    ],
}

UNKNOWN_TERMS = [
    "hypothetical", "uncharacterized", "unknown", "duf", "uniref"
]
```

### S3.2 Classification Priority

Categories are checked in order: structural → replication → protease → nuclease → packaging → regulatory → movement → lysis → unknown

First matching category is assigned. If no category matches and no unknown terms are found, "unknown" is assigned as default.

## S4. Dark Matter Classification Criteria

### S4.1 Thresholds

```python
WEAK_HIT_EVALUE_THRESHOLD = 1e-5    # E-values worse than this are "weak"
WEAK_HIT_IDENTITY_THRESHOLD = 0.3   # Identity below 30% is "weak"
WEAK_CONFIDENCE_THRESHOLD = 0.3     # Below "low" confidence
```

### S4.2 Classification Logic

```
IF no hits detected:
    category = "no_hits"
    reason = "No structural homologs detected in any database"

ELSE IF functional_category == "unknown":
    IF evalue > 1e-5 OR identity < 0.3:
        category = "weak_hits"
        reason = "Only weak structural homologs (e-value: {evalue}, identity: {identity})"
    ELSE:
        category = "unknown_function"
        reason = "Structural homologs found but function is uncharacterized"

ELSE IF consensus_score < 0.3:
    category = "weak_hits"
    reason = "Low confidence annotation (score: {consensus_score})"

ELSE:
    NOT dark matter
```

## S5. Output File Specifications

### S5.1 Main Results TSV Schema

| Column | Type | Description |
|--------|------|-------------|
| query_id | string | Input protein identifier |
| query_length | integer | Amino acid sequence length |
| description | string | Functional annotation |
| confidence_level | enum | high/medium/low/very_low/none |
| consensus_score | float | 0-1 quality score |
| agreement | enum | agree/partial/disagree/single/none |
| functional_category | string | Assigned category |
| primary_source | string | Database name |
| primary_target | string | Hit identifier |
| primary_evalue | float | E-value |
| primary_identity | float | Sequence identity (0-1) |
| primary_coverage | float | Query coverage (0-1) |
| secondary_source | string | Second database (optional) |
| secondary_target | string | Second hit identifier (optional) |
| secondary_evalue | float | Second E-value (optional) |
| secondary_identity | float | Second identity (optional) |
| organism | string | Source organism |
| gene | string | Gene name |
| uniprot_id | string | UniProt accession |
| bfvd_hits | integer | Count of BFVD hits |
| viro3d_hits | integer | Count of Viro3D hits |

### S5.2 Summary JSON Schema

```json
{
  "vhold_version": "string",
  "timestamp": "ISO8601 datetime",
  "input_file": "string",
  "databases_searched": ["string"],
  "parameters": {
    "threads": "integer",
    "batch_size": "integer",
    "device": "string",
    "evalue": "float",
    "sensitivity": "float",
    "confidence_threshold": "float"
  },
  "statistics": {
    "total_proteins": "integer",
    "annotated": "integer",
    "unannotated": "integer",
    "annotation_rate": "float",
    "with_multi_db_consensus": "integer",
    "consensus_rate": "float",
    "confidence_distribution": {
      "high": "integer",
      "medium": "integer",
      "low": "integer",
      "very_low": "integer",
      "none": "integer"
    },
    "agreement_distribution": {
      "agree": "integer",
      "partial": "integer",
      "disagree": "integer",
      "single": "integer",
      "none": "integer"
    },
    "category_distribution": {"category": "integer"},
    "primary_source_distribution": {"database": "integer"},
    "consensus_score_stats": {
      "min": "float",
      "max": "float",
      "mean": "float",
      "median": "float"
    },
    "evalue_stats": {
      "min": "float",
      "max": "float",
      "median": "float"
    }
  },
  "dark_matter": {
    "total_dark_matter": "integer",
    "dark_matter_rate": "float",
    "by_category": {
      "no_hits": "integer",
      "unknown_function": "integer",
      "weak_hits": "integer"
    },
    "by_confidence": {"level": "integer"},
    "length_stats": {
      "min": "integer",
      "max": "integer",
      "mean": "float",
      "median": "integer"
    }
  }
}
```

### S5.3 Dark Matter TSV Schema

| Column | Type | Description |
|--------|------|-------------|
| query_id | string | Protein identifier |
| query_length | integer | Sequence length |
| category | enum | no_hits/unknown_function/weak_hits |
| reason | string | Human-readable explanation |
| best_evalue | float | Best hit E-value (if any) |
| best_identity | float | Best hit identity (if any) |
| confidence_level | string | Confidence level |
| consensus_score | float | Consensus score (if any) |
| description | string | Annotation (if any) |
| databases_with_hits | string | Comma-separated list |

## S6. Performance Benchmarks

### S6.1 Computational Performance

| Component | Hardware | Throughput |
|-----------|----------|------------|
| ProstT5 prediction | NVIDIA V100 | ~1,000 proteins/hour |
| ProstT5 prediction | Apple M1 Pro | ~500 proteins/hour |
| ProstT5 prediction | CPU (8 cores) | ~50 proteins/hour |
| Foldseek search | CPU (8 cores) | ~10,000 proteins/hour |

### S6.2 Memory Requirements

| Component | Memory |
|-----------|--------|
| ProstT5 model (GPU FP16) | 3 GB |
| ProstT5 model (CPU FP32) | 6 GB |
| BFVD database | 534 MB |
| Viro3D database | 569 MB |
| Working memory | ~2 GB per 1000 proteins |

## S7. Software Dependencies

### S7.1 Python Dependencies

```
torch >= 2.0.0
transformers >= 4.30.0
pandas >= 2.0.0
requests >= 2.28.0
biopython >= 1.80
click >= 8.0.0
loguru >= 0.7.0
```

### S7.2 External Tools

- **Foldseek**: Version 8.x or later
  - Installation: `conda install -c conda-forge -c bioconda foldseek`
  - Or: Download binary from https://github.com/steineggerlab/foldseek

### S7.3 ProstT5 Model

- **Model**: Rostlab/ProstT5
- **Source**: HuggingFace Hub
- **Size**: ~3 GB download, ~6 GB extracted
- **Auto-download**: On first use via transformers library

## S8. Example Analysis

### S8.1 Test Dataset

Three well-characterized viral proteins were used for validation:

1. **T4 phage gp23 major capsid protein** (P04535)
   - Length: 521 aa
   - Expected: Structural (capsid)

2. **HIV-1 protease** (O90777)
   - Length: 99 aa
   - Expected: Protease

3. **SARS-CoV-2 nucleocapsid** (P0DTC9)
   - Length: 419 aa
   - Expected: Structural (nucleocapsid)

### S8.2 Results Summary

| Protein | Category | Confidence | Consensus | Agreement |
|---------|----------|------------|-----------|-----------|
| T4 gp23 | structural | high | 1.03 | single |
| HIV protease | protease | high | 1.00 | agree |
| SARS-CoV-2 N | structural | high | 1.00 | agree |

- **Annotation rate**: 100% (3/3)
- **Multi-database consensus**: 67% (2/3)
- **Dark matter**: 0% (0/3)

All proteins correctly classified to expected functional categories with high confidence.
