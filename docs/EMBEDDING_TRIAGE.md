# Embedding-Based Triage for vHold

Last updated: 2026-02-08

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

## Pre-Computed Embedding Database

### Size estimate
- 436K proteins (BFVD + Viro3D combined)
- 1024 dimensions per protein
- fp16: 436,000 x 1024 x 2 bytes = ~850 MB
- fp32: 436,000 x 1024 x 4 bytes = ~1.7 GB
- Distributable alongside existing Foldseek databases

### Generation
- One-time compute on GPU (hours, not days)
- Distribute as a downloadable file via `vhold install`
- Update when databases update

### Search at runtime
- For typical vHold queries (tens to hundreds of proteins): brute-force cosine similarity via numpy is sufficient (milliseconds for 436K comparisons).
- For large-scale batch queries (thousands+): FAISS approximate nearest neighbor as optional optimization.

## Dependency Impact

| Component | Current | With Triage |
|-----------|---------|-------------|
| ProstT5 model | Required | Required (encoder reused) |
| numpy | Required | Required (cosine similarity) |
| Foldseek | Required | Required (for grey/dark matter) |
| Embedding database | -- | New download (~1-2 GB) |
| faiss-cpu | -- | Optional (large batch optimization) |
| MMseqs2 / DIAMOND | -- | Not needed |

## Open Questions Requiring Empirical Testing

### 1. Similarity threshold calibration
What cosine similarity threshold separates "can transfer annotation" from "needs structural search"? This requires:
- Computing embeddings for proteins with known pairwise identities
- Mapping cosine similarity to approximate sequence identity ranges
- Determining the threshold where annotation transfer is reliable
- Possibly different thresholds for different confidence levels

### 2. ProstT5 vs ProtT5 encoder behavior
ProstT5 was fine-tuned from ProtT5 for AA-to-3Di translation. The fine-tuning may have shifted the encoder representations. Questions:
- Do ProstT5 encoder embeddings cluster proteins as well as vanilla ProtT5?
- Is there a meaningful difference for homology detection?
- Should we use ProstT5 (already loaded) or the encoder-only ProtT5 model (separate download)?

### 3. CPU performance of encoder-only inference
The 0.1s/protein GPU benchmark is from the literature. We need to measure:
- Encoder-only forward pass time on Apple Silicon (MPS and CPU)
- Scaling with sequence length (encoder is O(n) attention, not O(n^2) autoregressive)
- Whether this is fast enough to make the triage step negligible compared to I/O

### 4. Embedding quality for viral proteins specifically
Most embedding benchmarks use general protein datasets. Viral proteins have unusual properties:
- High mutation rates
- Many disordered regions
- Short accessory proteins with generic descriptions
- Do embeddings capture functional similarity for these edge cases?

### 5. Integration with existing pipeline
- Should triage be the default (`vhold run` always triages) or opt-in (`--triage`)?
- What metrics to report for embedding matches vs structural search hits?
- How to handle the case where embedding says "close match" but the user wants structural confirmation?

## Relationship to Other Roadmap Items

This approach subsumes two previously separate roadmap items:

- **BLAST pre-filtering**: Replaced by embedding triage. No BLAST/DIAMOND needed.
- **Pre-computed reference proteomes**: The embedding database IS the pre-computed reference, in a more useful form than raw 3Di sequences.

It also complements:

- **Fast mode (--fast)**: For proteins that DO need structural search, --fast reduces decoder time. Triage reduces the NUMBER of proteins going through the decoder.
- **LLM classification (--llm-classify)**: Still applies as a post-processing step for functional category assignment on all annotated proteins regardless of method.

## Expected Impact

For a typical viromics query of 1,000 viral proteins:
- Current: ~1,000 ProstT5 predictions (hours to days on CPU)
- With triage: ~100-400 ProstT5 predictions (the grey/dark matter fraction)
- Estimated 2-5x overall speedup, with the improvement scaling with how well-characterized the input proteins are

For well-characterized virus proteomes (SARS-CoV-2, influenza, HIV): most proteins matched by embedding, minimal structural search needed.

For novel metagenomic viruses: most proteins go through full structural search, triage provides modest speedup but correctly identifies the few with known homologs.

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
