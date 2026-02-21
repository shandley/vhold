# vHold Project State - Claude Code Context

Last updated: 2026-02-21

## Project Overview

**vHold** is a viral protein annotation tool using structural homology (ProstT5 + Foldseek). It extends pHold's phage-focused capabilities to include eukaryotic viruses.

### Key Differentiator
- **pHold**: Targets bacteriophages
- **vHold**: Targets ALL viruses including mammalian/eukaryotic viruses

## Architecture

```
vHold Pipeline:
0. (Optional) Gene calling from nucleotide contigs (--gene-caller auto|pyrodigal)
   - Pyrodigal-gv: ab initio viral ORF calling in metagenomic mode
   - miniprot: splice-aware protein-to-genome alignment for eukaryotic viruses
   - Merge: Pyrodigal primary, miniprot supplementary (non-overlapping spliced genes)
1. Read input (FASTA, GenBank, or GFF3) — auto-detected by extension
   - GenBank/GFF: extracts proteins with genomic coordinates (contig, start, end, strand)
1b. (Optional) Embedding triage: ProstT5 encoder cosine similarity against 436K references
    - LoRA adapter auto-loaded if installed (--lora/--no-lora)
    - Alignment validation: rejects false positives with <15% sequence identity
    - Matched proteins: skip decoder, transfer annotation directly
    - Unmatched proteins: continue to step 2
2. Predict 3Di with ProstT5 decoder (--fast for greedy, default beam search)
   - Supports --backend onnx for 3-6x CPU speedup
3. Search BFVD + Viro3D with Foldseek
4. Transfer annotations with multi-database consensus scoring
4a. Merge triage results with structural search results
4a.2. (Auto) Disorder annotation via metapredict (--disorder/--no-disorder, auto-enabled)
4a.5. (Auto) MLP classifier: reclassify "unknown" proteins using trained embedding classifier
4a.6. (Auto) STARLING similarity search: search 35M UniRef50 IDRs for disordered unknowns
4a.7. (Auto) Disorder classifier: STARLING embeddings MLP for remaining disordered unknowns
4b. (Auto) LLM reclassification of unknown proteins (--llm/--no-llm, auto-enabled with API key)
4c. (Auto) Neighborhood voting: gene-order-based reclassification (GenBank/GFF input only)
5. Generate output: TSV (with GO IDs), summary JSON, dark matter report
6. (Optional) Export to metagenomic formats: anvi'o, vConTACT2/3, DRAM-v, GFF3
```

## Key Source Files

| File | Purpose |
|------|---------|
| `src/vhold/features/prostt5.py` | 3Di prediction (ProstT5 model, MPS/CUDA/CPU), `get_predictor()` factory |
| `src/vhold/features/embeddings.py` | Embedding triage (EmbeddingExtractor with LoRA auto-load, EmbeddingDatabase, triage_proteins) |
| `src/vhold/features/contrastive.py` | Contrastive loss (SupConHardLoss, MultiGranularityLoss), CategoryBalancedSampler, ContrastiveDataset |
| `src/vhold/features/alignment_validation.py` | Triage match validation via pairwise sequence alignment (BLOSUM62) |
| `src/vhold/features/foldseek.py` | Structural search against BFVD/Viro3D + `create_query_db()` |
| `src/vhold/features/foldmason.py` | FoldMason MSA wrapper (`run_foldmason_msa()`) |
| `src/vhold/features/classifier.py` | MLP functional classifier (model definition + batch inference) |
| `src/vhold/features/disorder.py` | Disorder prediction (metapredict) + STARLING embeddings |
| `src/vhold/features/disorder_classifier.py` | Disorder-aware MLP classifier (STARLING 512-dim inputs) |
| `src/vhold/features/starling_search.py` | STARLING FAISS similarity search + consensus building |
| `src/vhold/features/uniprot_lookup.py` | UniProt REST API client for resolving UniRef50 hits |
| `src/vhold/features/genecall.py` | Unified gene calling: Pyrodigal-gv + miniprot + merge logic |
| `src/vhold/features/onnx_export.py` | ONNX INT8 export + quantization |
| `src/vhold/features/onnx_predictor.py` | ONNX ProstT5 predictor (OnnxProstT5Predictor) |
| `src/vhold/features/onnx_embeddings.py` | ONNX embedding extractor (OnnxEmbeddingExtractor) |
| `src/vhold/subcommands/run.py` | Main pipeline orchestration |
| `src/vhold/subcommands/align.py` | Multiple structural alignment pipeline |
| `src/vhold/databases/bfvd.py` | BFVD metadata loading + enriched annotation lookup |
| `src/vhold/results/export.py` | Metagenomic format export (anvi'o, vConTACT2/3, DRAM-v, GFF3) |
| `src/vhold/subcommands/export.py` | Export subcommand orchestration |
| `src/vhold/databases/embeddings.py` | Embedding DB path/install (Zenodo record 18652045) |
| `src/vhold/databases/classifier.py` | Classifier checkpoint path management |
| `src/vhold/databases/lora.py` | LoRA adapter path management |
| `src/vhold/databases/disorder_classifier.py` | Disorder classifier checkpoint path management |
| `src/vhold/results/annotations.py` | Consensus annotation transfer |
| `src/vhold/results/categories.py` | Keyword-based functional classification |
| `src/vhold/results/go_terms.py` | GO term ID resolution (name→ID from bundled OBO map) |
| `src/vhold/results/neighborhood.py` | Neighborhood voting (gene-order-based classification) |
| `src/vhold/results/llm_classify.py` | LLM-based functional classification (keywords + embedding unknowns) |
| `src/vhold/utils/constants.py` | Thresholds, DB URLs, embedding config |
| `src/vhold/io/genbank.py` | GenBank CDS feature extraction with coordinates |
| `src/vhold/io/gff.py` | GFF3 parsing with embedded ##FASTA support |
| `src/vhold/io/nucleotide.py` | Nucleotide FASTA detection and reading |
| `src/vhold/databases/miniprot_refs.py` | Bundled miniprot reference DB path management |
| `src/vhold/cli.py` | Click CLI (run, predict, compare, install, align, export, export-onnx) |
| `scripts/train_contrastive.py` | Contrastive LoRA training (SupCon-Hard + multi-granularity) |
| `scripts/evaluate_contrastive.py` | Before/after evaluation (MAP@k, NDCG, silhouette) |
| `scripts/rebuild_embedding_db.py` | Re-extract 436K embeddings with LoRA-merged encoder |
| `scripts/train_classifier.py` | MLP classifier training with stratified split + class weights |
| `scripts/generate_llm_labels.py` | Batch LLM label generation for training data expansion |
| `scripts/calibrate_triage_threshold.py` | Triage threshold calibration across 4 case studies |
| `scripts/build_go_term_map.py` | Build GO term name→ID JSON from OBO file |
| `scripts/train_disorder_classifier.py` | Disorder classifier training (STARLING embeddings) |
| `tests/` | 854+ tests (pytest) |

## Implemented Features

### Core Pipeline
- ProstT5 3Di prediction with per-residue confidence scoring
- Foldseek structural search against BFVD (351K structures) and Viro3D (85K structures)
- Multi-database consensus scoring with Viro3D 1.2x quality bonus
- Functional classification via Pfam, GO, SUPERFAMILY, and keyword matching
- Dark matter analysis for unannotated proteins
- Novelty classification (database_match, close_homolog, remote_homolog, twilight_zone)

### Gene Calling (`--gene-caller`) -- COMPLETE

Unified gene calling module for nucleotide contig input. Pyrodigal-gv handles standard viral ORFs (~97%), while miniprot provides splice-aware protein-to-genome alignment for eukaryotic viral genes with introns (~3%).

**Status**: Complete. 83 tests (38 unit + 22 nucleotide IO + 23 integration with real T4 phage genome).

| Component | Purpose | Install |
|-----------|---------|---------|
| Pyrodigal-gv | Ab initio viral ORF calling (metagenomic mode) | `pip install vhold[genecalling]` |
| miniprot | Splice-aware protein-to-genome alignment | `conda install -c bioconda miniprot` |
| Reference DB | 74 curated Swiss-Prot proteins from 9 virus families | Bundled (`data/miniprot_viral_refs.faa`, 39KB) |

**CLI options**: `--gene-caller` (auto\|pyrodigal\|none, default: none), `--translation-table` (default: 11), `--closed/--no-closed`, `--min-gene` (default: 90nt)

**Merge strategy**: All Pyrodigal genes kept. miniprot genes added only when <50% reciprocal overlap with any Pyrodigal gene on the same contig/strand — catches spliced genes missed by ab initio calling.

**Reference protein families**: Adenoviridae, Papillomaviridae, Polyomaviridae, Retroviridae, Herpesviridae, Orthomyxoviridae, Bornaviridae, Circoviridae, Caulimoviridae. 29 truly multi-exon CDS + 3 frameshift products + 6 leader-spliced + 35 contiguous ORFs across 34 organisms.

**Validation results**:
- SARS-CoV-2: 12 Pyrodigal genes (correct — no spliced genes)
- HAdV-5: 31 Pyrodigal + 11 miniprot hits (all overlapping >50% → 31 total)
- HPV-16: 12 miniprot hits (cross-species matches included)
- T4 phage: 265 Pyrodigal genes from full genome

**Output**: Called genes written as GFF3 + protein FASTA to output directory. `CalledGene` → `ProteinRecord` conversion preserves genomic coordinates for downstream neighborhood voting.

### Embedding-Based Triage (`--triage`) -- COMPLETE

Fast pre-screening using ProstT5 encoder-only embeddings (~0.1s/protein) to skip the expensive decoder for known proteins. End-to-end wired in CLI and pipeline.

**Status**: Complete. Hosted on Zenodo (record 18652045). Install via `vhold install --embeddings`.

| Metric | Value |
|--------|-------|
| Reference proteins | 436,237 (351K BFVD + 85K Viro3D) |
| Embedding dimensions | 1,024 (float16, L2 normalized) |
| File size | 822 MB |
| Search latency | 59ms for 436K proteins (brute-force cosine) |
| Default threshold | 0.90 (calibrated) |
| Location | `~/.vhold/databases/embeddings/vhold_embeddings.npz` |

**Triage calibration results** (85 proteins across 4 case studies):

| Configuration | Precision | F1 | Notes |
|--------------|-----------|------|-------|
| Bare embeddings | 29.4% | - | Before enriched metadata |
| + Enriched BFVD metadata | 67.1% | 0.803 | 345K UniProt annotations |
| + Keyword fixes | 77.6% | 0.874 | Structural, protease, regulatory terms |
| + LLM reclassification | **83.5%** | **0.910** | Claude Haiku resolves ORF/VP unknowns |

At threshold 0.90: 100% recall (all 85 test proteins matched, min similarity 0.904). Remaining 14 false positives are annotation quality issues (deleted UniProt entries, UniParc-only IDs, ground truth category debates).

**Enriched BFVD metadata**: `~/.vhold/databases/bfvd/bfvd_metadata_enriched.tsv` (79 MB, 345,730 entries). Contains protein names, gene names, GO BP/MF, Pfam domains, keywords, function_cc, lineage from UniProt REST API. Hosted on Zenodo (record 18652045), auto-downloaded with `vhold install`.

### MLP Functional Classifier (`--classify`) -- COMPLETE

Lightweight MLP trained on frozen ProstT5 encoder embeddings to classify viral proteins into functional categories. Runs automatically in the pipeline (Step 4a.5) for any protein classified as "unknown" by keyword/Pfam/GO matching.

**Status**: Complete. Model hosted on Zenodo (record 18652045). Install via `vhold install --classifier`. Auto-runs when installed; disable with `--no-classify`.

| Metric | Value |
|--------|-------|
| Architecture | MLP (1024 -> 512 -> 256 -> 11) with LayerNorm, ReLU, Dropout(0.3) |
| Parameters | 660,491 |
| Training samples | 84,250 (from 436K total, excluding 343K "unknown") |
| Validation macro F1 | 0.692 |
| Validation accuracy | 78.8% |
| Training time | 62s on CPU |
| Checkpoint size | 2.6 MB |
| Location | `~/.vhold/models/classifier/vhold_classifier.pt` |

**Training label sources**: Keyword/Pfam/GO annotations (69K proteins) + agreement-filtered LLM labels (15K additional proteins where both text-based LLM classification and structural model predictions agreed).

**Per-category performance** (validation F1):

| Category | F1 | Support | Notes |
|----------|:--:|:--:|-------|
| replication | 0.863 | 3,681 | Best performer |
| structural | 0.823 | 5,262 | High precision (93.1%) |
| nuclease | 0.809 | 854 | |
| protease | 0.783 | 666 | |
| lysis | 0.722 | 185 | |
| regulatory | 0.701 | 1,089 | |
| host_interaction | 0.669 | 231 | |
| packaging | 0.568 | 206 | |
| entry | 0.511 | 393 | Low precision (36.8%) |
| movement | 0.477 | 65 | Rarest category |

**Agreement filtering insight**: Text-based labels and structure-based predictions are complementary signals. Labels where both agree are high-quality; labels where they disagree are noisy. This principle could be applied iteratively.

### Contrastive LoRA Fine-Tuning (`--lora/--no-lora`) -- CODE COMPLETE, NEEDS CUDA

Fine-tunes ProstT5 encoder with LoRA + supervised contrastive loss (SupCon-Hard, based on CLEAN, Yu et al. Science 2023) so that functionally similar viral proteins produce closer embeddings. Improves triage and classifier quality.

**Status**: Code complete, integrated into pipeline, unit tests pass. **Training requires CUDA GPU** -- MPS is too slow (~60 sec/batch, estimated 35+ hours for 1000 proteins). Adapter not yet produced.

| Metric | Value |
|--------|-------|
| LoRA rank | 16 |
| Target layers | Top 6 of 24 encoder layers (q + v attention) |
| Trainable parameters | 983,040 / 2,819,835,904 (0.03%) |
| Adapter size | ~5 MB |
| Loss | Multi-granularity: category (weight 0.7) + Pfam (weight 0.3) |
| Temperature | 0.07 (category), 0.049 (Pfam) |
| Hard negative mining | Hardest negative per anchor upweighted |
| Batch sampling | Category-balanced (equal representation per batch) |
| Inference | `merge_and_unload()` -- zero overhead, no peft runtime dependency |
| Location | `~/.vhold/models/contrastive_lora/` |
| Requires | `pip install 'vhold[contrastive]'` (peft>=0.14.0) for training only |

**Training configuration**:
- AdamW lr=2e-5, weight_decay=0.01
- Linear warmup (500 steps) + cosine decay
- Gradient accumulation 4 steps, effective batch 256
- Max sequence length 1024 tokens
- Early stopping on MAP@1 (patience 3)

**MPS training attempt results** (Apple M4, 24GB):
- 1000 proteins (855 train / 145 val), batch_size=8, seq_length=512
- Model loading: ~25 min (11GB pytorch_model.bin deserialization)
- Per-batch speed: ~60 sec (forward through 24 encoder layers + backward through 6 LoRA layers)
- Batch 50/432 loss=0.9844 after ~49 min of training
- Estimated epoch time: ~7 hours; 5 epochs: ~35 hours
- **Verdict**: Impractical on MPS. Needs CUDA A100/H100 where batches would be ~5-10 sec

**Cost-benefit assessment**: Current triage achieves 83.5% precision / 100% recall / F1=0.910 without contrastive fine-tuning. Remaining 14 false positives are annotation quality issues (deleted UniProt entries, UniParc-only IDs), not embedding quality. Marginal improvement from contrastive training (~2-7% precision) does not justify the compute cost on available hardware.

**Integration**: `EmbeddingExtractor.load_model()` auto-detects installed adapter, loads via peft, and calls `merge_and_unload()` to permanently fuse weights. Falls back gracefully if peft not installed or adapter corrupt. Disable with `--no-lora`.

### ONNX INT8 Quantization (`--backend onnx`) -- VALIDATED

Optional ONNX INT8 backend for faster CPU inference. One-time export via `vhold export-onnx`, then use with `--backend onnx` on run/predict/align commands.

**Status**: Complete. Exported, quantized, and validated on Apple Silicon (arm64).

| Metric | Value |
|--------|-------|
| Export command | `vhold export-onnx` |
| Quantization | INT8 dynamic (auto-detected: avx512_vnni, avx512, avx2, arm64) |
| Model format | ONNX via Optimum `ORTModelForSeq2SeqLM` |
| FP32 model size | ~16.8 GB (3 ONNX files) |
| INT8 model size | **4.23 GB** (75% reduction) |
| Location | `~/.vhold/models/onnx_int8/` |
| Requires | `pip install 'vhold[onnx]'` (optimum>=1.17.0 + onnxruntime>=1.17.0,<1.24) |
| Export time | ~25 min on Apple M4 (model loading + ONNX tracing + INT8 quantization) |

**Validation results** (Apple M4, arm64 INT8, greedy decoding):
- SARS-CoV-2 ORF10 (38aa): 3Di prediction correct, confidence 0.963, 1.3s
- SARS-CoV-2 NS7B (43aa): 3Di prediction correct, confidence 0.952, 1.4s
- SARS-CoV-2 NS6 (61aa): 3Di prediction correct, confidence 0.971, 1.7s
- Embeddings: 1024-dim, L2-normalized, correct shape and dtype
- All 3Di characters in valid alphabet

**Integration**: `get_predictor(backend="onnx")` returns `OnnxProstT5Predictor`. Triage and MLP classifier steps route to `OnnxEmbeddingExtractor` when backend is ONNX. Model reuse between triage and decoder is skipped for ONNX (separate model instances).

**Note**: onnxruntime>=1.24 requires Python 3.11+. Pinned to <1.24 for Python 3.10 compatibility.

### GenBank/GFF3 Input (`--input-format`) -- COMPLETE

Accepts annotated genome files in addition to FASTA. Auto-detects format from file extension; override with `--input-format`. GenBank/GFF input preserves genomic coordinates for neighborhood voting.

| Format | Extensions | Parser |
|--------|-----------|--------|
| Protein FASTA | `.fasta`, `.fa`, `.faa` | BioPython SeqIO |
| Nucleotide FASTA | `.fna`, `.ffn` | Nucleotide detection + gene calling |
| GenBank | `.gb`, `.gbk`, `.gbf`, `.genbank` | BioPython CDS extraction |
| GFF3 | `.gff`, `.gff3` | Custom parser + nucleotide translation |

- GenBank: extracts CDS features with `/translation` qualifiers, preserves product/gene/locus_tag
- GFF3: supports embedded `##FASTA` (Prodigal output) or separate `--genome-fasta`
- Position fields (`contig`, `start`, `end`, `strand`) added to `ProteinRecord` and `ConsensusResult`
- Positions appear in TSV output when present

### Alignment Validation of Triage Matches -- COMPLETE

Validates embedding triage matches with lightweight pairwise sequence alignment. Rejects matches where cosine similarity is high but global sequence identity is below 15% (convergent structural similarity). Reference sequences read directly from installed Foldseek BFVD/Viro3D databases.

- Uses BioPython PairwiseAligner + BLOSUM62 scoring
- `FoldseekSequenceIndex` reads sequences from MMseqs2/Foldseek database format
- Graceful degradation: skips validation if Foldseek DBs not installed
- Default threshold: `DEFAULT_TRIAGE_MIN_IDENTITY = 0.15` (15% global sequence identity)

### GO Term ID Propagation -- COMPLETE

Resolves GO IDs (GO:0039694) from GO term names for downstream tool compatibility (TopGO, DRAM-v, anvi'o). Adds `go_bp_ids` and `go_mf_ids` columns to consensus TSV output.

- Bundled `go_term_map.json` with 125K name→ID mappings (from Gene Ontology OBO + EXACT synonyms)
- Handles BFVD format (inline `term [GO:ID]` — extracts IDs directly)
- Handles Viro3D format (plain term names — looks up from map)
- Wired into `get_bfvd_annotation()` and `get_viro3d_annotation()` return paths
- Build script: `scripts/build_go_term_map.py --obo /path/to/go-basic.obo`

### Neighborhood Voting (auto-activates with GenBank/GFF) -- COMPLETE

Uses gene order to reclassify remaining unknown proteins when genomic positions are available. Runs as Step 4c after LLM classification, before output generation.

- Distance + confidence weighted majority voting over ±5 nearest genes on same contig
- Weight = `confidence_weight * (1 / (1 + rank_distance))`
- Winner needs ≥2 votes and ≥50% of total weight
- Cross-contig isolation: genes on different contigs do not vote for each other
- No CLI flag needed — activates automatically when positions are available

### Metagenomic Export (`vhold export` / `--export-format`) -- COMPLETE

Export vHold results to formats compatible with the viral metagenomics ecosystem. Dual interface: standalone `vhold export` for post-hoc conversion, or `--export-format` on `vhold run` for integrated pipeline export.

| Format | Output File | Target Tool |
|--------|-----------|-------------|
| anvi'o | `*_anvio_functions.txt` | anvi'o functional annotation import |
| vConTACT2 | `*_vcontact2_gene2genome.csv` | vConTACT2 protein clustering |
| vConTACT3 | `vcontact3/gene2genome.tsv` + `genome_lengths.tsv` | vConTACT3 clustering |
| DRAM-v | `*_dramv_annotations.tsv` | DRAM-v metabolic annotation |
| GFF3 | `*_annotations.gff3` | General annotation viewers |

- Contig/scaffold IDs derived from protein IDs using Prodigal (`_N` suffix) and geNomad provirus patterns
- anvi'o: multi-row per gene (structural hit, Pfam, GO BP, GO MF, category)
- GFF3: only includes proteins with genomic coordinates (from GenBank/GFF input)
- No new dependencies (all formats are simple TSV/CSV/GFF3)

### Intrinsic Disorder Prediction (`--disorder/--no-disorder`) -- COMPLETE

Three-layer disorder integration for viral proteins poorly served by structural homology:

**Layer 1: metapredict** (Step 4a.2) — Fast per-residue disorder scoring. Auto-enabled when metapredict is installed. Classifies proteins as ordered (<15%), partially_disordered (15-40%), or highly_disordered (>40%).

**Layer 2: STARLING Similarity Search** (Step 4a.6) — Searches pre-built FAISS index of ~35M IDR sequences from UniRef50 for disordered unknowns. Resolves hits to functional annotations via UniProt REST API. Auto-enabled when STARLING SearchEngine is available.

**Layer 3: Disorder Classifier** (Step 4a.7) — MLP trained on STARLING 512-dim embeddings for remaining disordered unknowns. Architecture: 512→256→128→11 with LayerNorm, ReLU, Dropout(0.3).

| Component | Package | Install | Database Size |
|-----------|---------|---------|--------------|
| metapredict V3 | `vhold[disorder]` | auto | — |
| STARLING embeddings | `vhold[starling]` | auto | — |
| STARLING search DB | `vhold install --starling-db` | manual | ~18.6GB |
| Disorder classifier | `vhold install --disorder-classifier` | manual | ~1MB |

**ConsensusResult fields**: `disorder_fraction`, `disorder_regions`, `disorder_class` — propagated to TSV, summary JSON, and dark matter reports.

**Novelty values**: `ensemble_match` (STARLING search), existing values unchanged.

**UniProt API hardening** (completed):
- GET requests with URL-encoded query params (POST not supported by `/uniprotkb/search`)
- User-Agent identification: `vHold/1.0 (https://github.com/shandley/vhold)`
- 3 retries with exponential backoff (1s, 2s, 4s) on HTTP 429/503
- `Retry-After` header honored (delay-seconds format), clamped to [1, 300]s
- UniProt lookup cache persisted to `{output_dir}/uniprot_cache.json` across runs
- Atomic cache writes via `tempfile.mkstemp()` + `Path.replace()`
- Resilient cache loading (handles corrupt JSON, unreadable files)
- Module-level SearchEngine cache to avoid reloading 18.6GB FAISS index
- Classification evidence source tracking: `classification_source=f"starling_search:{evidence_source}"`

### LLM Classification (`--llm/--no-llm`)
- Post-processing step using Claude to reclassify proteins where keywords return "unknown"
- **Auto-enabled** when `ANTHROPIC_API_KEY` is set; `--no-llm` to opt out
- Targets `classification_source in ("keywords", "embedding")` AND `functional_category == "unknown"`
- Resolves: SARS-CoV-2 accessory proteins (ORF3a/6/7a/8), paramyxovirus V/C proteins, Ebola VP35, Rabies M2
- 3-layer graceful degradation: no anthropic package, no API key, per-protein API errors
- Default model: `claude-haiku-4-5-20251001` (configurable via `--llm-model`)
- Cost: ~$0.01 per run
- Requires: `pip install vhold[llm]` + `ANTHROPIC_API_KEY` environment variable

### Fast Mode (`--fast`)
- Greedy decoding (`num_beams=1, do_sample=False`) vs default beam search (`num_beams=3, do_sample=True`)
- ~2x faster on MPS, ~3x faster theoretical maximum
- 94.8% 3Di identity vs beam search (tested on 173aa protein)

### Apple Silicon (MPS) GPU Acceleration
- MPS auto-detected on Apple Silicon via `--device auto` (default)
- `PYTORCH_ENABLE_MPS_FALLBACK=1` set automatically for unsupported ops
- Previous instability fixed in transformers v4.43+
- Current stack: PyTorch 2.10.0 + transformers 5.0.0

**Benchmarks (Apple M4)**:
| Protein Size | CPU | MPS | Speedup |
|--------------|-----|-----|---------|
| 173aa | 88s | 40s | 2.2x |
| 435aa | 170s | 95s | 1.8x |
| 1135aa | ~132 min | 72 min | ~1.8x |

### `vhold align` -- Multiple Structural Alignment via FoldMason -- COMPLETE

Performs multiple structural alignment of viral protein families by bridging ProstT5 3Di predictions with FoldMason's alignment engine.

```bash
vhold align -i proteins.fasta -o alignment/ --device cpu --fast -t 8
```

**Pipeline**: Input FASTA -> ProstT5 3Di -> confidence masking -> Foldseek DB -> FoldMason `structuremsa` (fastMode) -> AA MSA + 3Di MSA + guide tree

## Completed Case Studies

| # | Name | Proteins | Results |
|---|------|----------|---------|
| 1 | SARS-CoV-2 | 18 | 55.6% annotated, 100% structural accuracy |
| 2 | Remote Homology | 10 | 100% annotated, 91.7% twilight zone hits |
| 3 | Metagenomic Dark Matter | 30 | 83.3% annotated, 70% RdRp classification |
| 4 | crAssphage ORFans | 37 | Setup only (deprioritized - phage focus) |
| 5 | Eukaryotic Viruses | 27 | 100% annotated, 81.5% consensus, 88.9% category accuracy (with LLM) |
| 6 | T7 Phage | 60 | Complete (triage + structural search results available) |

## Known Issues

- **LoRA adapter not yet trained**: Code complete but training requires CUDA GPU. MPS too slow (~60 sec/batch). Run on A100/H100: `scripts/train_contrastive.py --device cuda`.

## Remaining Work

### Immediate (before release)
- **Train contrastive LoRA adapter** (deferred -- needs CUDA): Code ready, run on A100/H100. Marginal value at current stage (83.5% precision limited by annotation quality, not embedding quality)
- **Manuscript / Application Note**: Bioinformatics Application Note with case studies

### Medium-term
- **Metagenomic input metadata**: Parse VirSorter2 score TSV, geNomad taxonomy TSV, VIBRANT quality TSV to carry upstream viral classification through to output. (Output export to anvi'o/vConTACT2/3/DRAM-v/GFF3 is COMPLETE via `vhold export`.)
- **Docker container**: Package vHold + ProstT5 + Foldseek + databases for Biocontainers/Docker Hub
- **Iterative label refinement**: Use v3 classifier as filter for another round of LLM label agreement filtering
- **Batch processing improvements**: Batch same-length sequences for ProstT5

### Research directions
- **PST (Protein Set Transformer)**: Genome-level attention over protein sets for improved annotation
- **Full ProstT5 seq2seq fine-tuning**: Fine-tune decoder on viral 3Di sequences
- **Structural distance-based viral taxonomy**: Pairwise structural distances for phylogenetics below twilight zone
- **BitNet 1.58-bit quantization-aware retraining**: Ternary ProstT5 for 3-6x CPU speedup
- **Metagenomic viral protein detection**: Repurpose embedding DB as fast structural-similarity-based viral protein detector

## Commands Reference

```bash
# Run vHold (auto-selects MPS on Apple Silicon, CUDA on Linux)
uv run vhold run -i input.fasta -o output/ -t 8

# GenBank input (auto-detected, preserves positions for neighborhood voting)
uv run vhold run -i genome.gb -o output/ -t 8

# GFF3 input with separate genome FASTA
uv run vhold run -i genes.gff3 --genome-fasta genome.fna -o output/

# Gene calling from nucleotide contigs (Pyrodigal-gv + miniprot)
uv run vhold run -i contigs.fna -o output/ --gene-caller auto -t 8

# Pyrodigal only (no miniprot splice-aware search)
uv run vhold run -i contigs.fna -o output/ --gene-caller pyrodigal

# With embedding triage (skip decoder for known proteins)
uv run vhold run -i input.fasta -o output/ --triage

# LLM classification auto-enabled when ANTHROPIC_API_KEY is set
# To disable: --no-llm
uv run vhold run -i input.fasta -o output/ --triage --no-llm

# Disable LoRA adapter (enabled by default when installed)
uv run vhold run -i input.fasta -o output/ --triage --no-lora

# Disable MLP classifier (enabled by default when model installed)
uv run vhold run -i input.fasta -o output/ --no-classify

# Disorder prediction (auto-enabled when metapredict installed)
# Disable: --no-disorder; STARLING search: --no-starling-search
uv run vhold run -i input.fasta -o output/ --disorder

# ONNX backend for faster CPU inference
uv run vhold export-onnx                              # one-time export
uv run vhold run -i input.fasta -o output/ --backend onnx

# Fast mode (greedy decoding, ~2x faster)
uv run vhold run -i input.fasta -o output/ --fast

# Two-step workflow (GPU predict, CPU search)
uv run vhold predict -i input.fasta -o predictions/ --device cuda
uv run vhold compare -p predictions/ -o results/ -t 32

# Multiple structural alignment
uv run vhold align -i proteins.fasta -o alignment/ --fast --device cpu -t 8

# Export to metagenomic tool formats (all formats by default)
uv run vhold export -i output/vhold_results.tsv -o exports/

# Export specific formats only
uv run vhold export -i output/vhold_results.tsv -o exports/ -f anvio -f dramv

# Integrated export during pipeline run
uv run vhold run -i input.fasta -o output/ --export-format anvio --export-format gff3

# Database setup
uv run vhold install
uv run vhold install --starling-db  # STARLING search database (~18.6GB)

# Run tests
uv run pytest tests/ -v

# Train contrastive LoRA adapter
uv run python scripts/train_contrastive.py --output results/contrastive/ --device mps --install

# Evaluate contrastive adapter
uv run python scripts/evaluate_contrastive.py --adapter results/contrastive/contrastive_lora/ --output results/contrastive/eval/

# Rebuild embedding DB with LoRA-enhanced encoder
uv run python scripts/rebuild_embedding_db.py --output ~/.vhold/databases/embeddings/vhold_embeddings.npz

# Train MLP classifier
uv run python scripts/train_classifier.py --output /tmp/classifier_output --install

# Train with LLM-expanded labels
uv run python scripts/train_classifier.py --output /tmp/classifier_output --extra-labels results/llm_labels_filtered.json --install

# Generate LLM training labels (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-... uv run python scripts/generate_llm_labels.py --output results/llm_labels.json
```

## Contact

This project documentation maintained for Claude Code session continuity.
