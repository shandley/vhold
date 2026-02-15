# vHold Roadmap: Strategic Directions

Last updated: 2026-02-14

## Context

vHold has a working pipeline: ProstT5 3Di prediction, Foldseek structural search against BFVD + Viro3D, multi-database consensus scoring, keyword + LLM functional classification, and dark matter reporting. Five case studies validate the approach across SARS-CoV-2, remote homology, metagenomic dark matter, crAssphage ORFans, and eukaryotic viruses.

This document evaluates strategic directions for the project, grounded in what we've learned and where the highest-value opportunities are.

## Comparison with pHold

pHold searches 1.36 million phage-specific structures and integrates with pharokka (gene prediction) and cross-references PHROGs, VFDB, CARD, acrDB, and DefenseFinder. Its value chain is: pharokka finds genes, pHold tells you what they do.

vHold differs in scope (all viruses, not just phage) and in approach (multi-database consensus, LLM classification, dark matter reporting). The strategic question is where vHold's unique strengths create the most value.

## Gene Prediction: Not a Good Fit

Gene prediction was considered and rejected for vertebrate viruses. The situations are fundamentally different from phage:

- **Polyproteins** (picornaviruses, flaviviruses): one ORF, many proteins, protease cleavage sites
- **RNA editing** (paramyxoviruses): P gene encodes P, V, and W proteins through mRNA editing
- **Ribosomal frameshifting** (coronaviruses, retroviruses): overlapping frames read as fusion proteins
- **Splicing** (retroviruses, adenoviruses, herpesviruses): eukaryotic-style introns
- **Ambisense coding** (arenaviruses, bunyaviruses): both sense and antisense from same segment

Prodigal achieves ~71% F1 on eukaryotic viruses and drops to ~60% on RNA viruses. No single gene finder handles all strategies. Building genome-type-aware gene prediction would be a separate project orthogonal to vHold's core strength.

More importantly, for most vertebrate viruses people care about, gene models already exist in NCBI. The bottleneck is not finding genes -- it is understanding what the proteins do.

## The Value Framework: White, Grey, and Dark Matter

| Matter | Identity | vHold Classification | BLAST | vHold Value |
|--------|----------|---------------------|-------|-------------|
| **White** | >70% | database_match, close_homolog | Works well | Low -- confirmatory only |
| **Grey** | 30-70% | remote_homolog | Marginal/unreliable | **High** -- confident annotation where BLAST wavers |
| **Dark** | <30% or no hits | twilight_zone, no_hits | Fails | **Highest** -- only structural methods work |

The grey zone (remote homolog, 30-70% identity) is where vHold is most practically valuable. Pure dark matter is scientifically interesting but has nothing to transfer. Grey matter is where structural similarity enables confident functional assignment but BLAST gives nothing useful. Case Study 3's mean identity of 53% is in this sweet spot.

For well-annotated proteins (white matter), vHold provides cross-validation, structural context, and misannotation detection, but these are secondary benefits that rarely justify the compute cost as a standalone use case.

## Proposed Directions

### 1. BLAST Pre-Filtering (`--blast-first` mode)

**Status**: Not started
**Value**: High -- makes vHold practical for larger datasets
**Effort**: Moderate

Run DIAMOND/BLAST as a fast pre-screen. Only invoke ProstT5 + Foldseek on proteins that BLAST cannot confidently annotate. This:

- Respects compute budgets (ProstT5 is expensive; BLAST is cheap)
- Focuses structural search on proteins that actually need it
- Produces unified output with provenance tracking (annotated_by: blast vs structure)
- Could be implemented as `vhold run --blast-first` or as a separate `vhold tiered` subcommand

**Design questions**:
- What BLAST e-value threshold defines "confidently annotated"? (1e-10? 1e-5?)
- Should we use DIAMOND for speed, or standard BLAST for compatibility?
- Do we include the BLAST annotations in the output alongside structural annotations?
- Should proteins with BLAST hits still get structural search for cross-validation?

**Implementation sketch**:
1. Run DIAMOND against UniRef90 or NR (viral subset)
2. Partition proteins into "annotated" (below e-value threshold) and "unannotated"
3. Run existing vHold pipeline on unannotated proteins only
4. Merge results into unified output with source column

### 2. Metagenomic Pipeline Integration

**Status**: Not started
**Value**: High -- meets users where they are
**Effort**: Moderate to high

The real market for vHold is viromics. Integration points:

**Input formats to accept**:
- VirSorter2 output (viral contigs + predicted proteins)
- VIBRANT output (viral bins)
- geNomad output (viral sequences with protein predictions)
- Generic: any FASTA of predicted viral proteins

**Output formats to produce**:
- DRAM-v compatible annotations
- anvi'o compatible functional annotations
- vConTACT2 compatible protein clusters
- Standard GFF3 with structural annotation features

**Design questions**:
- Which pipeline integration is most impactful first?
- Do we need to handle nucleotide input (run prodigal ourselves) or always require protein FASTA?
- How do we handle the scale? Metagenomic datasets can have 10,000+ viral proteins.

### 3. Pre-Computed 3Di for Reference Proteomes

**Status**: Not started
**Value**: High -- removes the biggest practical barrier
**Effort**: Moderate

Pre-compute and distribute 3Di structural representations for all RefSeq viral proteomes. This would:

- Eliminate the ProstT5 bottleneck for known viruses
- Enable instant structural comparison between any virus and the reference set
- Make vHold practical for large-scale surveillance

**Design questions**:
- How many proteins in RefSeq viral? (~500K? Need to check)
- Storage format? (FASTA of 3Di sequences + confidence scores)
- Distribution mechanism? (Same as database install, or separate?)
- How to detect when a query protein matches a pre-computed reference?
- Update cadence? (RefSeq updates quarterly)

**Implementation sketch**:
1. Download all RefSeq viral protein sequences
2. Run ProstT5 on a GPU cluster (one-time cost)
3. Package as a downloadable 3Di database
4. At runtime, hash-match query proteins against reference; skip ProstT5 for matches
5. Only predict 3Di for truly novel sequences

### 4. Emerging Pathogen Rapid Characterization

**Status**: Not started (Case Study 1 is a proof of concept)
**Value**: High for public health, niche audience
**Effort**: Low (mostly documentation and workflow packaging)

When a novel virus emerges, the first question is: what do its proteins do? Sequence databases may have no close matches. vHold can provide same-day structural annotations of an entire novel viral proteome.

**Use case**: Public health agencies, pandemic preparedness programs, outbreak response.

**What's needed**:
- A streamlined "rapid characterization" workflow document/mode
- Pre-built container image (Docker/Singularity) for deployment
- Example using a real outbreak scenario (CS1 with SARS-CoV-2 is a start)
- Integration with pathogen genomics pipelines (Nextstrain, Terra/Cromwell)

### 5. Structural Novelty as a Discovery Tool

**Status**: Partially implemented (dark matter report exists)
**Value**: Medium-high for research, lower for routine annotation
**Effort**: Moderate

Beyond annotation, vHold's dark matter report identifies proteins with no structural homologs. These are candidates for:

- Novel viral protein families
- Potential drug targets (truly unique viral folds)
- Evolutionary studies (when did this fold originate?)

**Enhancements**:
- Cluster dark matter proteins by predicted structure (are the unknowns related to each other?)
- Cross-reference against non-viral structural databases (PDB, AFDB) to find host mimicry
- Report predicted structural features (disorder, transmembrane, secondary structure) for unknowns

### 6. Structural Distance-Based Viral Taxonomy

**Status**: Not started
**Value**: Medium -- scientifically interesting, niche audience
**Effort**: High

Sequence-based phylogenetics breaks down for rapidly-evolving RNA viruses. Structural distances decay much more slowly. A tool that computes pairwise structural distances between viral proteomes could reveal evolutionary relationships invisible to sequence alignment.

**Design questions**:
- Which structural distance metric? (TM-score, 3Di alignment score, LDDT)
- How to handle multi-domain proteins?
- Can Foldseek all-vs-all provide the distance matrix?
- How does this relate to ICTV's increasing use of structure-informed taxonomy?

## Completed Since Last Update

- **MLP functional classifier** (`--classify`): Lightweight MLP (1024→512→256→11, 660K params) trained on frozen ProstT5 encoder embeddings. Macro F1 0.692 on validation set. Auto-runs in pipeline Step 4a.5 to reclassify "unknown" proteins. Three training iterations: v1 (keyword labels, F1=0.635), v2 (raw LLM labels, F1=0.531 — worse due to text/structure mismatch), v3 (agreement-filtered LLM labels, F1=0.692 — best). Key finding: filtering LLM labels through structural model agreement removes noise from text-based annotations.
- **LLM training label generation**: Batch LLM classification of 32,312 unique protein descriptions via Claude Haiku (808 API calls, $2.52, 38 min). 55,941 proteins labeled across 10 categories. Agreement filtering with structural model retained 18,746 high-quality labels (33.5% agreement rate).
- **Embedding triage infrastructure**: ProstT5 encoder embedding DB complete (436K proteins, 822 MB, 59ms search). Subsumes BLAST pre-filtering (Direction 1) and pre-computed 3Di references (Direction 3) — see `docs/EMBEDDING_TRIAGE.md`.
- **FoldMason integration (`vhold align`)**: Multiple structural alignment via ProstT5 → FoldMason fastMode bridge. Enables structural phylogenetics (Direction 6) as a practical tool. Note: refinement incompatible with coordinate-free mode (segfaults).
- **Triage threshold calibration**: Empirical calibration across 85 proteins from 4 case studies. Optimal threshold: 0.90 (100% recall, 83.5% precision, F1=0.910). Enriched BFVD metadata (345K UniProt annotations) dramatically improved annotation transfer quality.
- **End-to-end `--triage` wiring**: Full pipeline integration — embedding search → annotation transfer → keyword classification → LLM reclassification. CLI flags: `--triage` and `--triage-threshold`. Decoder + Foldseek steps automatically skipped when all proteins matched.
- **LLM reclassification of triage unknowns**: Extended `--llm-classify` to reclassify embedding-matched proteins with "unknown" category. Claude Haiku correctly resolves accessory/regulatory proteins (e.g., SARS-CoV-2 ORF7a, ORF6, ORF3a → host_interaction). Improved precision from 77.6% to 83.5%.

## Updated Priority Ranking

| # | Direction | Value | Effort | Status |
|---|-----------|-------|--------|--------|
| 1 | ~~BLAST pre-filtering~~ | ~~High~~ | ~~Moderate~~ | **Superseded** by embedding triage |
| 3 | ~~Pre-computed 3Di~~ | ~~High~~ | ~~Moderate~~ | **Superseded** by embedding triage |
| -- | ~~Triage threshold calibration~~ | ~~High~~ | ~~Low~~ | **COMPLETE** — threshold 0.90, F1=0.910 |
| -- | ~~Wire `--triage` end-to-end~~ | ~~High~~ | ~~Moderate~~ | **COMPLETE** — `--triage` and `--llm-classify` working |
| -- | ~~MLP functional classifier~~ | ~~High~~ | ~~Moderate~~ | **COMPLETE** — macro F1 0.692, `--classify` in pipeline |
| 2 | Metagenomic integration | High | High | **Next priority** |
| -- | Iterative label refinement | Medium | Low | Use v3 as filter for another agreement round |
| -- | Contrastive encoder fine-tuning | High | High | Would improve both triage and classifier |
| 4 | Emerging pathogen workflow | High (niche) | Low | Do when packaging for release |
| 5 | Structural novelty discovery | Medium-high | Moderate | Research feature, iterate |
| 6 | Structural taxonomy | Medium | Moderate | **Partially enabled** by `vhold align` |

## References

- pHold: https://github.com/gbouras13/phold
- Pharokka: https://github.com/gbouras13/pharokka
- Viro3D: https://www.embopress.org/doi/full/10.1038/s44320-025-00147-9
- Viral Dark Matter review: https://pubmed.ncbi.nlm.nih.gov/41264852/
- Global metagenomics functional dark matter: https://www.nature.com/articles/s41586-023-06583-7
- Gene prediction comparison: https://www.biorxiv.org/content/10.1101/2021.12.11.472104v1.full
- RNA virus structurome dark matter: https://pubmed.ncbi.nlm.nih.gov/39714180/
