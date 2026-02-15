# vHold: Methods and Algorithms

## Overview

vHold (Viral Homology-based Annotation Tool) is a computational pipeline for functional annotation of viral proteins using structural homology. The method leverages the observation that protein tertiary structure is 3-10 times more conserved than primary sequence during evolution, enabling detection of remote homology relationships that sequence-based methods miss.

## Biological Rationale

### The Challenge of Viral Protein Annotation

Viral proteins present unique challenges for functional annotation:

1. **Rapid sequence evolution**: RNA viruses and many DNA viruses evolve at rates 10^4-10^6 times faster than their hosts, leading to extensive sequence divergence even among functionally related proteins.

2. **Limited sequence similarity**: Many viral proteins share <20% sequence identity with characterized homologs, below the threshold for reliable BLAST-based annotation.

3. **Viral "dark matter"**: 40-70% of viral proteins in metagenomic datasets lack any functional annotation using sequence-based methods.

4. **Convergent evolution**: Some viral protein functions have evolved independently multiple times, creating functional analogs with no sequence similarity.

### Structural Homology as a Solution

Protein structure constrains sequence evolution more strongly than function alone. Key observations supporting structural homology approaches:

- Proteins with <20% sequence identity often maintain the same fold and function
- The 3Di structural alphabet captures local geometric features that are conserved across divergent sequences
- Structure-based searches can detect homology at sequence identities as low as 10-15%

## Pipeline Architecture

```
Input FASTA → ProstT5 Prediction → Confidence Masking → Foldseek Search →
    → Multi-Database Consensus → Functional Classification → Dark Matter Analysis → Output
```

### Step 1: 3Di Structural Sequence Prediction

**Model**: ProstT5 (Rostlab/ProstT5)

ProstT5 is a protein language model based on the T5 architecture that translates amino acid sequences directly to 3Di structural alphabet sequences without requiring explicit structure prediction.

**3Di Alphabet**: The 3Di alphabet encodes local structural geometry using 20 characters representing discrete structural states. Each 3Di character captures:
- Backbone dihedral angles (φ, ψ)
- Cα-Cα distances
- Local secondary structure context

**Input Preparation**:
```
Sequence: MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
Formatted: <AA2fold> M V L S P A D K T N V K A A W G K V G A H A G E ...
```

**Generation Parameters**:
- Method: Sampled generation with nucleus sampling
- Temperature: 1.2
- Top-p (nucleus): 0.95
- Top-k: 6
- Beam candidates: 3
- Repetition penalty: 1.2

**Confidence Score Extraction**:
Per-residue confidence scores are extracted from the generation logits:

```
confidence[i] = max(softmax(logits[i]))
```

where `logits[i]` is the model's output distribution over the 3Di alphabet at position `i`.

**Mean confidence** typically ranges from 0.70-0.95 for well-folded globular proteins, with lower values indicating disordered regions or unusual structural features.

### Step 2: Confidence-Based Masking

Low-confidence 3Di positions are masked to improve search specificity:

```
3Di_masked[i] = 'X' if confidence[i] < threshold else 3Di[i]
```

Default threshold: 0.7

Masked positions are treated as wildcards during Foldseek searches, preventing spurious matches at unreliable positions.

### Step 3: Structural Homology Search

**Search Tool**: Foldseek

Foldseek searches are performed against two complementary databases:

#### BFVD (Big Fantastic Virus Database)
- **Size**: 351,242 viral protein structures
- **Source**: AlphaFold2-predicted structures with quality metrics
- **Coverage**: Primarily bacteriophages with some eukaryotic viruses
- **Annotations**: UniProt accessions linked to functional descriptions

#### Viro3D
- **Size**: 85,162 high-confidence viral structures
- **Source**: Curated subset of AlphaFold predictions
- **Coverage**: 4,400+ virus species across all viral taxa
- **Annotations**: Direct Pfam domains, ICTV taxonomy, gene/product names

**Search Parameters**:
- E-value threshold: 1e-3 (default)
- Sensitivity: 9.5
- Maximum hits per query: 1000
- Mode: Exhaustive search (no prefiltering)

**Output Metrics**:
- E-value and bit score
- Sequence identity in aligned region (fident)
- Query and target coverage (qcov, tcov)
- Alignment length

### Step 4: Multi-Database Consensus Scoring

The consensus scoring algorithm integrates results from multiple databases to improve annotation confidence.

#### 4.1 Hit Quality Score

For each Foldseek hit, a quality score is calculated:

```
quality = 0.5 × evalue_score + 0.3 × identity_score + 0.2 × coverage_score

where:
    evalue_score = min(1.0, -log₁₀(evalue) / 50)
    identity_score = fident  (0-1)
    coverage_score = qcov    (0-1)
```

E-value contributes most strongly (50%) as it best captures statistical significance of structural similarity.

#### 4.2 Database Weighting

Databases are weighted based on annotation quality:

| Database | Weight | Rationale |
|----------|--------|-----------|
| Viro3D   | 1.2    | Curated functional annotations |
| BFVD     | 1.0    | Broader coverage, UniProt-derived annotations |

A functional annotation bonus (1.1×) is applied when the annotation contains a real functional description rather than just identifiers or "hypothetical protein".

```
weighted_score = quality × database_weight × functional_bonus
```

#### 4.3 Structure Quality Weighting

The quality of the AlphaFold/ESMFold/ColabFold structure prediction used for the database hit influences the reliability of the homology match. Structure quality is assessed using pLDDT (predicted Local Distance Difference Test) and pTM (predicted Template Modeling) scores.

**Quality Metrics Sources (in priority order)**:

1. **ColabFold** (preferred when MSA depth ≥ 100): Uses multiple sequence alignments for more accurate predictions
2. **ESMFold**: Single-sequence predictions, fast but may be less accurate
3. **BFVD AlphaFold2**: Original AlphaFold2 predictions from the BFVD database

**Structure Quality Score Calculation**:

```
pLDDT Score (0-1):
  - High (≥70):     0.9-1.0 (normalized to range)
  - Medium (50-70): 0.7-0.9
  - Low (30-50):    0.5-0.7
  - Very Low (<30): 0.3-0.5

pTM Score (0-1):
  - High (≥0.5):    0.85-1.0
  - Medium (0.3-0.5): 0.7-0.85
  - Low (<0.3):     0.5-0.7

Combined Structure Quality = 0.7 × pLDDT_score + 0.3 × pTM_score
```

**MSA Depth Bonus** (ColabFold only):
- MSA depth ≥1000: 5% bonus
- MSA depth ≥500: 2% bonus

**Integration with Hit Quality**:

Structure quality adjusts the alignment quality score:

```
combined_quality = alignment_quality × (0.85 + 0.15 × structure_quality)
```

This means structure quality can reduce the final score by up to 15% for low-quality structure predictions, while high-quality predictions provide scores very close to the alignment-based quality.

#### 4.4 Cross-Database Agreement

When hits exist in multiple databases, annotation agreement is assessed:

1. **Keyword Extraction**: Significant terms (>3 characters, excluding stop words like "protein", "viral", "hypothetical") are extracted from each description.

2. **Jaccard Similarity**:
```
similarity = |terms₁ ∩ terms₂| / |terms₁ ∪ terms₂|
```

3. **Agreement Classification**:
   - **Agree**: similarity ≥ 0.5 (consensus bonus: 1.3×)
   - **Partial**: similarity ≥ 0.2 (consensus bonus: 1.15×)
   - **Disagree**: similarity < 0.2 (no bonus)
   - **Single**: only one database has hits

#### 4.5 Final Consensus Score

```
consensus_score = primary_weighted_score × agreement_bonus
```

The consensus score is capped at 1.0.

### Step 5: Confidence Level Assignment

Confidence levels guide interpretation of annotation reliability:

| Level | Consensus Score | Interpretation |
|-------|----------------|----------------|
| High | ≥ 0.8 | Strong structural evidence, reliable annotation |
| Medium | 0.5-0.8 | Good structural match, annotation likely correct |
| Low | 0.3-0.5 | Marginal structural similarity, use with caution |
| Very Low | < 0.3 | Weak evidence, consider as potential dark matter |

**Agreement upgrade**: Proteins with "agree" status in multi-database consensus are upgraded one confidence level (e.g., medium → high).

### Step 6: Functional Category Classification

Annotated proteins are classified into functional categories using a hierarchical evidence-based approach. Classification leverages multiple annotation sources when available:

#### 6.1 Evidence Hierarchy (Priority Order)

1. **Pfam Domain Annotations** (highest priority): Direct domain-level functional classification
2. **SUPERFAMILY Annotations**: Structural superfamily membership
3. **GO Biological Process**: Biological process involvement
4. **GO Molecular Function**: Molecular activity classification
5. **Keyword Matching** (fallback): Text-based classification from descriptions

#### 6.2 Functional Categories

| Category | Description | Evidence Sources |
|----------|-------------|------------------|
| **Structural** | Virion structural proteins | Pfam: capsid, coat, envelope, spike; GO BP: viral capsid assembly |
| **Replication** | Genome replication machinery | Pfam: polymerase, helicase, primase, thymidine kinase; GO BP: DNA replication |
| **Protease** | Proteolytic enzymes | Pfam: peptidase, assemblin; GO MF: peptidase activity |
| **Nuclease** | Nucleic acid processing | Pfam: nuclease, integrase; GO MF: nuclease activity |
| **Packaging** | Genome packaging | Pfam: terminase, UL6; GO BP: genome packaging |
| **Regulatory** | Gene regulation | Pfam: kinase domain; GO BP: regulation of transcription |
| **Movement** | Cell-to-cell movement | Pfam: movement protein |
| **Lysis** | Host cell lysis | Pfam: lysin, holin, endolysin |
| **Host Interaction** | Host interaction/immune evasion | GO BP: perturbation of host defense |
| **Entry** | Host cell entry | GO BP: fusion with host membrane |
| **Unknown** | Uncharacterized function | No functional evidence available |

#### 6.3 Classification Source Tracking

Each classification includes a `classification_source` field indicating what evidence was used:
- `pfam:<domain_name>` - Pfam domain match
- `superfamily:<name>` - SUPERFAMILY match
- `go_bp:<term>` - GO Biological Process match
- `go_mf:<term>` - GO Molecular Function match
- `keywords` - Text-based keyword matching (fallback)
- `mlp_classifier` - MLP embedding classifier prediction
- `llm:<model>` - LLM-based reclassification

### Step 6b: MLP Embedding Classifier (Automatic)

When a trained classifier model is installed, proteins classified as "unknown" by the keyword/Pfam/GO system are automatically re-evaluated using a lightweight MLP trained on ProstT5 encoder embeddings.

**Architecture**: MLP with input dimension 1024, hidden layers [512, 256], LayerNorm + ReLU + Dropout(0.3) at each layer, and 11 output classes (10 functional categories + unknown).

**Training Data**: 84,250 proteins with labels from three sources:
1. **Keyword/Pfam/GO annotations** (69K proteins): High-confidence labels from the evidence hierarchy
2. **Agreement-filtered LLM labels** (15K proteins): Protein descriptions batch-classified by Claude Haiku, filtered to retain only labels where the structural model independently agreed

**Training Details**:
- 85/15 stratified train/val split
- Inverse-frequency class weights in CrossEntropyLoss
- AdamW optimizer, lr=1e-3, weight_decay=1e-4, cosine annealing
- Batch size 2048, early stopping on val macro F1 (patience=5)

**Performance** (validation set, macro F1 = 0.692):

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| structural | 0.931 | 0.737 | 0.823 |
| replication | 0.937 | 0.801 | 0.863 |
| protease | 0.717 | 0.862 | 0.783 |
| nuclease | 0.761 | 0.863 | 0.809 |
| packaging | 0.428 | 0.845 | 0.568 |
| regulatory | 0.613 | 0.819 | 0.701 |
| movement | 0.342 | 0.785 | 0.477 |
| lysis | 0.607 | 0.892 | 0.722 |
| host_interaction | 0.540 | 0.879 | 0.669 |
| entry | 0.368 | 0.835 | 0.511 |

**Integration**: The classifier only overrides "unknown" proteins -- it never overrides Pfam, SUPERFAMILY, or GO evidence. Predictions below the confidence threshold (default 0.5) are not applied. Reclassified proteins receive `classification_source: "mlp_classifier"`.

**CLI**: Enabled by default (`--classify`), disable with `--no-classify`. Confidence threshold adjustable via `--classifier-confidence`.

### Step 7: Dark Matter Analysis

Proteins lacking confident functional annotation are flagged as viral "dark matter" for targeted investigation.

#### Dark Matter Categories

1. **No Hits**: No structural homologs detected in any database
   - Truly novel proteins or highly divergent sequences
   - Priority targets for experimental characterization

2. **Unknown Function**: Strong structural hits but uncharacterized function
   - Conserved structure suggests important biological role
   - Candidates for functional genomics studies

3. **Weak Hits**: Only marginal structural homologs
   - E-value > 1e-5 OR sequence identity < 30%
   - Ambiguous homology relationships
   - May represent rapidly evolving proteins

#### Dark Matter Statistics

The dark matter rate is calculated as:
```
dark_matter_rate = (no_hits + unknown_function + weak_hits) / total_proteins
```

Length distribution statistics help identify whether dark matter proteins have characteristic sizes.

#### Metagenomic Characterization (Optional)

Dark matter proteins can be further characterized using large-scale metagenomic resources:

**1. Serratus Integration** (serratus.io)
- Ultra-deep search results from 5.7 million SRA datasets
- Identifies viral families with related sequences
- Provides environmental and geographic distribution context
- API-based search with rate limiting

**2. Logan Integration** (logan-search.org)
- Planetary-scale genome assembly from 27.3M SRA datasets
- k-mer based sequence search for environmental distribution
- Identifies conserved sequences across metagenomes

**Metagenomic Context Output**:
- `serratus_family_hits`: Viral families with related sequences
- `serratus_total_runs`: Number of SRA datasets with matches
- `is_conserved_in_metagenomes`: Boolean indicating conservation
- `conservation_score`: 0-1 score based on metagenomic prevalence

**Priority Scoring**:
Dark matter proteins are assigned priority scores (0-1) for follow-up analysis based on:
- No structural homologs (+0.4)
- Conservation in metagenomes (+0.3)
- Unknown function with strong structural match (+0.2)
- Optimal length for single-domain protein (100-500 aa) (+0.1)

**Recommendations**:
Automated recommendations are generated for each dark matter protein:
- Experimental characterization for conserved novel proteins
- AlphaFold/ESMFold prediction for novel folds
- InterProScan domain parsing for large proteins
- HHpred/DALI remote homology search

#### ESM Metagenomic Atlas Search (Optional)

For dark matter proteins without viral database hits, vHold can search the ESM Metagenomic Atlas to find structural homologs in metagenomic protein space.

**Database**: ESMAtlas30
- 617+ million protein structures predicted from metagenomic sequences
- Pre-clustered at 30% sequence identity for efficient searching
- Includes proteins from MGnify and other metagenomic sources
- Structures predicted by ESMFold with quality metrics

**Analysis Pipeline**:
1. Extract 3Di predictions for dark matter proteins
2. Create Foldseek database from dark matter sequences
3. Search against ESMAtlas30 with configured E-value threshold
4. Parse results and identify novel fold candidates

**Novel Fold Detection**:
Proteins with ESMAtlas hits but no BFVD/Viro3D hits may represent:
- Novel viral folds not yet characterized in cultured isolates
- Horizontally transferred domains from host or environmental microbes
- Conserved viral proteins with metagenomic evidence

**ESMAtlas Output**:
- `has_metagenomic_homolog`: Boolean indicating ESMAtlas hits
- `has_novel_fold_match`: Potential novel fold (ESM/MGY-prefixed targets)
- `unique_clusters`: Number of distinct structural clusters
- `best_evalue`: Best E-value from ESMAtlas search
- `best_identity`: Best sequence identity from ESMAtlas search

**High Priority Targets**:
Proteins flagged as high priority for follow-up:
- No hits in BFVD or Viro3D (truly novel to viruses)
- BUT has hits in ESMAtlas (conserved in metagenomes)
- Suggests functionally important but uncharacterized viral proteins

### Step 8: Cross-Database Validation

vHold includes a validation framework that assesses annotation reliability by comparing results across databases.

#### Validation Status

Each protein receives a validation status based on cross-database support:

| Status | Description | Reliability |
|--------|-------------|-------------|
| **Validated** | Consistent across multiple databases (similarity ≥50%) | High |
| **Partial** | Some agreement (similarity 20-50%) | Medium |
| **Conflicting** | Databases disagree on annotation | Low |
| **Single Source** | Only one database has annotation | Moderate |
| **Unvalidated** | No validation possible | Unknown |

#### Conflict Detection

Conflicts are detected when databases provide different annotations and classified by severity:

| Severity | Description | Criteria |
|----------|-------------|----------|
| **Critical** | Completely different functions | Different functional categories (e.g., structural vs. replication) |
| **Major** | Different related categories | Related categories or unknown with low similarity |
| **Minor** | Same category, different details | Same category, similarity 20-50% |
| **Negligible** | Terminology differences | Same category, similarity ≥50% |

#### Database Consistency Metrics

Pairwise consistency between databases is quantified:

```
consistency_score = 0.6 × agreement_rate + 0.3 × category_agreement_rate + 0.1 × overlap_bonus
```

Where:
- `agreement_rate` = annotations with similarity ≥50% / queries in both databases
- `category_agreement_rate` = same functional category / queries in both databases
- `overlap_bonus` = min(1.0, overlap_rate × 2)

#### Reliability Scoring

Each annotation receives a reliability score (0-1) based on:
- Number of supporting databases (30%)
- Cross-database agreement (50%)
- Functional category consistency (20%)

#### Validation Output

The validation report includes:
- Validation status distribution
- Reliability score statistics (mean, median, distribution)
- Database consistency metrics for each pair
- List of critical conflicts requiring manual review

## Output Formats

### Main Results Table (TSV)

| Column | Description |
|--------|-------------|
| query_id | Input protein identifier |
| query_length | Protein length (amino acids) |
| description | Transferred functional annotation |
| confidence_level | high/medium/low/very_low/none |
| consensus_score | Combined quality score (0-1) |
| agreement | agree/partial/disagree/single/none |
| functional_category | Assigned functional class |
| classification_source | Evidence source for classification (pfam/go_bp/go_mf/superfamily/keywords) |
| structure_quality_score | Structure prediction quality (0-1) |
| structure_quality_source | Source of structure quality (colabfold/esmfold/bfvd_af2/none) |
| primary_source | Database providing primary annotation |
| primary_target | Best hit identifier |
| primary_evalue | E-value of primary hit |
| primary_identity | Sequence identity of primary hit |
| primary_coverage | Query coverage of primary hit |
| secondary_source | Second database (if available) |
| secondary_target | Second-best hit identifier |
| organism | Source organism of primary hit |
| gene | Gene name (if available) |
| uniprot_id | UniProt accession |
| bfvd_hits | Number of BFVD hits |
| viro3d_hits | Number of Viro3D hits |

### Summary Statistics (JSON)

- Annotation rate and consensus rate
- Confidence level distribution
- Agreement distribution
- Functional category distribution
- Primary source distribution
- E-value and consensus score statistics
- Structure quality statistics (min, max, mean, median)
- Structure quality source distribution
- Dark matter summary

### Dark Matter Table (TSV)

Separate file listing all dark matter proteins with category, reason, best hit statistics, and confidence metrics.

## Implementation Notes

### Computational Requirements

- **ProstT5 Prediction**: GPU-accelerated (CUDA/MPS) or CPU
  - GPU: ~1 second per protein
  - CPU: ~30-60 seconds per protein

- **Foldseek Search**: CPU-only
  - ~0.1 seconds per protein against combined databases

### Memory Requirements

- ProstT5 model: ~3 GB GPU memory (FP16) or ~6 GB CPU memory (FP32)
- Foldseek databases: ~1.1 GB combined (BFVD + Viro3D)

### Scalability

The pipeline is designed for:
- Single genomes: Complete in minutes
- Metagenomic datasets: 100,000+ proteins feasible with batch processing

## Validation Approach

### Benchmark Datasets

Performance should be evaluated on:
1. Well-characterized viral genomes with experimental annotations
2. Recently characterized proteins not in training databases
3. Synthetic tests with known structural homologs at varying sequence identities

### Metrics

- Sensitivity: Fraction of known functions correctly annotated
- Specificity: Fraction of annotations that are correct
- Dark matter reduction: Decrease in unannotated proteins vs. sequence-only methods

## References

1. Heinzinger M, et al. (2023). ProstT5: Bilingual Language Model for Protein Sequence and Structure. bioRxiv.

2. van Kempen M, et al. (2023). Fast and accurate protein structure search with Foldseek. Nature Biotechnology.

3. Terzian P, et al. (2021). BFVD: Big Fantastic Virus Database.

4. Oughtred R, et al. (2023). Viro3D: A comprehensive resource for virus protein structures.

5. Bouras G, et al. (2023). Phold: Phage annotation using protein structural homology.

6. Edgar RC, et al. (2022). Petabase-scale sequence alignment catalyses viral discovery. Nature.

7. Roux S, et al. (2024). Logan: Planetary-Scale Genome Assembly Surveys Life's Diversity. bioRxiv.

8. Lin Z, et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science.
