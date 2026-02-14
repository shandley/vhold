# Embedding-Based Triage for vHold

Last updated: 2026-02-14

## Problem

ProstT5 3Di prediction is the computational bottleneck in vHold. On CPU, it takes minutes per protein with O(n^2) scaling. Even on GPU, it dominates runtime. For many proteins in a typical query set, this compute is unnecessary -- they have close homologs in the reference databases that could be identified by simpler methods. The challenge is building a fast triage step that avoids introducing heavy new dependencies.

## Core Insight

ProstT5 is a T5 encoder-decoder model. The current pipeline uses the full model: the encoder processes the amino acid sequence, then the decoder autoregressively generates 3Di tokens. The decoder is where all the cost lives (beam search, autoregressive generation, O(n^2) scaling).

The encoder alone is fast. A single forward pass produces a per-residue embedding matrix (L x 1024 dimensions) that can be mean-pooled to a single 1024-dimensional protein vector. This takes ~0.1 seconds per protein on GPU -- orders of magnitude faster than the decoder.

These encoder embeddings capture deep evolutionary and functional relationships. ProtT5 embedding-based k-NN search has been shown to outperform MMseqs2-sensitive for protein homology detection (Littmann et al., Genome Research 2024).

## Proposed Architecture

```
Query FASTA (N proteins)
    |
    v
ProstT5 ENCODER only (~0.1s/protein GPU, seconds/protein CPU)
    |
    v
Per-protein embeddings (N x 1024 vectors)
    |
    v
Cosine similarity against pre-computed BFVD+Viro3D embeddings
    |
    +-- Close match (similarity > T) ----> Annotate from reference match
    |                                       Skip decoder + Foldseek
    |                                       Label: "embedding_match"
    |
    +-- No close match (similarity < T) -> Run full pipeline
                                            Decoder (3Di) + Foldseek
                                            Label: "structural_search"
    |
    v
Merge results with provenance
```

## Why This Approach

### Reuses existing infrastructure
- ProstT5 is already a required dependency. The encoder is loaded as part of the model.
- No new binaries (no MMseqs2, no DIAMOND, no BLAST).
- No new sequence database to build or index.
- The comparison step is pure Python (numpy matrix multiply or optionally FAISS).

### The databases already represent surveyed sequence space
- BFVD (351K structures) was built via AlphaFold2 from UniProt TrEMBL viral sequences.
- Viro3D (85K structures) was built via ColabFold from RefSeq eukaryotic viruses.
- ColabFold used MMseqs2 MSAs against UniRef30 and environmental databases.
- Any protein with detectable sequence relatives in these databases already has a pre-computed structure.
- Pre-computing embeddings for these proteins captures this surveyed space in a searchable vector format.

### More sensitive than sequence alignment
- Embeddings encode evolutionary relationships beyond raw sequence identity.
- ProtT5 k-NN outperforms MMseqs2-sensitive for homology detection.
- The triage step catches not just obvious matches but also some grey-zone proteins where sequence alignment is ambiguous.

### Scientifically coherent
- The encoder "understands" the protein at the level that informs 3Di prediction.
- If the encoder recognizes a protein (close embedding), it has enough information to transfer annotation.
- If the encoder doesn't recognize it (distant embedding), that's exactly when the full structural prediction adds value.

## Pre-Computed Embedding Database — COMPLETE

### Generation complete (2026-02-13)
- **436,237 proteins** (351,242 BFVD + 84,995 Viro3D)
- 1024 dimensions per protein, float16, L2 normalized
- **822 MB** on disk at `~/.vhold/databases/embeddings/vhold_embeddings.npz`
- Generated in **16.2 hours** on Apple M4 CPU (encoder-only, ~0.1-10s/protein depending on length)
- Keys: `embeddings` (436237, 1024), `protein_ids` (436237,), `source_dbs` (436237,)

### Validation results
- Zero NaN/Inf values, zero duplicate IDs, zero near-zero vectors
- All norms within 0.9998-1.0002 (properly L2 normalized)
- 2/1024 zero-variance dimensions (negligible)
- Brute-force cosine similarity search: **59ms** over all 436K proteins
- Nearest neighbor results biologically sensible (related proteins cluster together)

### Search at runtime
- For typical vHold queries (tens to hundreds of proteins): brute-force cosine similarity via numpy is sufficient (**59ms for 436K comparisons**).
- For large-scale batch queries (thousands+): FAISS approximate nearest neighbor as optional optimization.

## Dependency Impact

| Component | Current | With Triage |
|-----------|---------|-------------|
| ProstT5 model | Required | Required (encoder reused) |
| numpy | Required | Required (cosine similarity) |
| Foldseek | Required | Required (for grey/dark matter) |
| Embedding database | -- | New download (822 MB) |
| faiss-cpu | -- | Optional (large batch optimization) |
| MMseqs2 / DIAMOND | -- | Not needed |

## Empirical Answers (from calibration across 85 proteins, 4 case studies)

### 1. Similarity threshold: 0.90 is optimal
Calibrated across SARS-CoV-2 (18), remote homology (10), metagenomic dark matter (30), and eukaryotic viruses (27). At threshold 0.90: 100% recall (all 85 test proteins have similarity >= 0.904), 83.5% precision, F1=0.910. Higher thresholds (0.95, 0.97) sacrifice recall without meaningful precision gains. Lower thresholds (0.80, 0.85) add no recall but may introduce noise on other datasets. A single threshold works well across all case studies — no need for per-category thresholds.

### 2. ProstT5 encoder works well — no need for separate ProtT5
ProstT5 encoder embeddings produce biologically sensible clusters. Related proteins (e.g., spike proteins, polymerases) cluster together with high cosine similarity. The fine-tuning did not degrade embedding quality for homology detection. Using ProstT5 avoids a separate model download and reuses the model already loaded for 3Di prediction.

### 3. CPU performance: 0.5-5s per protein on Apple M4
Encoder-only inference on CPU: ~0.5s for short proteins (<100aa), ~5s for long proteins (~1000aa). For 85 test proteins, total encoder time was ~4.5 minutes. This is fast enough that triage adds negligible overhead compared to the decoder time it saves (minutes to hours per protein). The 59ms search time over 436K embeddings is negligible.

### 4. Embedding quality: good for most viral proteins, weak for short accessory proteins
Well-characterized structural proteins (spike, capsid, polymerase) match with high similarity (>0.95) and transfer correct annotations. Short accessory proteins (<100aa) with generic descriptions ("hypothetical protein", "ORF6 protein") match at lower similarity (0.90-0.95) and often get classified as "unknown" by keyword matching. LLM reclassification resolves most of these cases correctly.

### 5. Integration: opt-in via `--triage`, with `--llm-classify` for unknowns
Triage is opt-in (`vhold run --triage`). The `--triage-threshold` flag allows overriding the default (0.90). Embedding matches are labeled with `classification_source: "embedding"` or `"llm:<model>"` in output. The full pipeline (decoder + Foldseek) is automatically skipped for proteins matched by triage. Users wanting structural confirmation can omit `--triage` to run the full pipeline.

## Relationship to Other Roadmap Items

This approach subsumes two previously separate roadmap items:

- **BLAST pre-filtering**: Replaced by embedding triage. No BLAST/DIAMOND needed.
- **Pre-computed reference proteomes**: The embedding database IS the pre-computed reference, in a more useful form than raw 3Di sequences.

It also complements:

- **Fast mode (--fast)**: For proteins that DO need structural search, --fast reduces decoder time. Triage reduces the NUMBER of proteins going through the decoder.
- **LLM classification (--llm-classify)**: Still applies as a post-processing step for functional category assignment on all annotated proteins regardless of method.

## Measured Impact

**SARS-CoV-2 end-to-end test** (18 proteins, `--triage --llm-classify`):
- 18/18 proteins matched by embedding triage (100% at threshold 0.90)
- Decoder + Foldseek steps completely skipped
- Total runtime: ~4.5 minutes (vs ~45+ min without triage) = **~10x speedup**
- LLM reclassified 6/10 unknown proteins correctly
- Zero errors, zero dark matter

**Calibration across 85 proteins** (4 case studies):

| Configuration | Precision | F1 |
|--------------|-----------|------|
| Bare embeddings | 29.4% | — |
| + Enriched BFVD metadata (345K UniProt) | 67.1% | 0.803 |
| + Keyword classification fixes | 77.6% | 0.874 |
| + LLM reclassification (Claude Haiku) | **83.5%** | **0.910** |

For well-characterized virus proteomes (SARS-CoV-2, influenza, HIV): most proteins matched by embedding, decoder completely skipped.

For novel metagenomic viruses: most proteins go through full structural search, triage correctly identifies the few with known homologs.

## Model Quantization: Accelerating What Remains

For proteins that pass triage and require full 3Di prediction, model quantization can dramatically reduce decoder inference time on CPU. This addresses the same bottleneck from a different angle: triage reduces the NUMBER of decoder calls, quantization makes each call FASTER.

### The BitNet Inspiration

Microsoft's [BitNet](https://github.com/microsoft/BitNet) uses ternary weights (-1, 0, +1), converting matrix multiplications into additions/subtractions. Results: 2-6x CPU speedups, 55-82% energy reduction. A 100B parameter model runs at reading speed on a single CPU.

bitnet.cpp itself targets decoder-only architectures and natively-trained ternary models, so ProstT5 cannot be dropped in directly. However, [research on 1.58-bit quantization-aware training for T5 encoder-decoder models](https://arxiv.org/html/2411.05882v1) shows ternary T5 models outperform their 16-bit counterparts -- the extreme quantization acts as a regularizer. This is a stronger result than for decoder-only models.

### Three levels of quantization

**Level 1: ONNX INT8 post-training quantization (immediate)**
- Export existing ProstT5 to ONNX, apply INT8 static quantization
- Expected: ~3x CPU speedup, minimal accuracy loss
- No retraining required -- applied to the existing model weights
- Dependencies: onnxruntime (Python package)
- [ONNX INT8 achieves 3.08x speedup on CPU for transformers](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)

**Level 2: 4-bit weight quantization (near-term)**
- Use bitsandbytes or GPTQ-style quantization
- Expected: ~75% memory reduction, comparable speed to fp16
- [Protein language models are robust to 4-bit quantization](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/)
- Combined with FlashAttention: 4-9x faster inference possible

**Level 3: 1.58-bit quantization-aware retraining (research direction)**
- Retrain ProstT5 with BitLinear layers (ternary weights)
- Expected: 2-6x CPU speedup, potentially better accuracy (regularization effect)
- Requires GPU compute for training, validation against ProstT5 benchmarks
- [T5 models with 1.58-bit outperform 16-bit across configurations](https://arxiv.org/html/2411.05882v1)

### Compounding effects

These optimizations are multiplicative with each other and with the embedding triage:

| Optimization | What it does | Speedup |
|-------------|-------------|---------|
| Embedding triage | Skip decoder for known proteins | 2-5x fewer decoder calls |
| --fast greedy decoding | Reduce beams from 3 to 1 | ~2x per protein |
| ONNX INT8 quantization | Faster math on CPU | ~3x per protein |
| BitNet 1.58-bit (future) | Ternary weights | ~3-6x per protein |

Example for a 500aa protein currently at ~10 min on CPU:

| Configuration | Time |
|--------------|------|
| Current (beam search, fp32) | ~10 min |
| + --fast (greedy) | ~5 min |
| + ONNX INT8 | ~1.5 min |
| + BitNet 1.58-bit (future) | ~30-50 sec |

If that protein would have been caught by embedding triage, it's ~1 second instead.

### Recommendation

Start with ONNX INT8 (Level 1). It's pure engineering -- export, quantize, benchmark. No retraining, no GPU cluster, and the 3x CPU speedup immediately improves the experience for every protein that goes through the decoder. The encoder-only inference for triage would also benefit from INT8 quantization, making the triage step even faster.

Keep BitNet-style retraining (Level 3) as a research milestone. The T5 results showing ternary models outperforming 16-bit are compelling, and a 1.58-bit ProstT5 trained specifically for viral protein 3Di prediction could be a standalone contribution.

## Combined Architecture Vision

```
Query FASTA (N proteins)
    |
    v
ProstT5 ENCODER (quantized, fast)
    |
    v
Embedding vectors (N x 1024)
    |
    v
Cosine similarity vs reference embeddings (milliseconds)
    |
    +-- Known proteins (embedding match) --> Transfer annotation
    |   (~60-80% of well-characterized datasets)
    |   (~10-20% of novel metagenomic datasets)
    |
    +-- Unknown proteins (no match) ---------> ProstT5 DECODER (quantized)
        |                                       + Foldseek structural search
        |                                       + Consensus scoring
        v
Merge all annotations
    |
    v
Functional classification (keywords + LLM)
    |
    v
Output: unified results with provenance
```

This architecture means:
- Every protein gets an encoder pass (fast, seconds)
- Only grey/dark matter proteins get a decoder pass (slow, but quantized)
- No new binary dependencies beyond what exists
- The same model serves both triage and structural prediction
- Provenance tracking distinguishes annotation methods

## References

- BitNet inference framework: https://github.com/microsoft/BitNet
- BitNet 1.58-bit T5 results: https://arxiv.org/html/2411.05882v1
- Protein LM quantization: https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/
- ProtT5 embeddings for homology: https://pmc.ncbi.nlm.nih.gov/articles/PMC11529836/
- Embedding nearest neighbor search: https://pmc.ncbi.nlm.nih.gov/articles/PMC9714024/
- ProstT5 model: https://huggingface.co/Rostlab/ProstT5
- ONNX quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- BitNet CPU inference: https://arxiv.org/html/2410.16144v1
