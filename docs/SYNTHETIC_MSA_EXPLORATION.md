# Synthetic MSA Augmentation for Viral Proteins

Last updated: 2026-02-08
Status: Exploration / future research direction

## Problem Statement

MSA (Multiple Sequence Alignment) quality is the primary determinant of structure prediction accuracy for AlphaFold2, ColabFold, and related methods. For viral proteins, MSAs are often shallow or empty due to:

1. **Rapid evolution**: viral proteins diverge quickly, fragmenting protein families across identity thresholds
2. **Sampling bias**: most viral diversity is unsequenced (environmental phages, invertebrate viruses, novel zoonotic reservoirs)
3. **Small genomes**: each viral protein family has fewer members than cellular families
4. **Database underrepresentation**: standard protein LM training data systematically underweights viral sequences

Approximately 11% of eukaryotic/viral proteins and 20% of metagenomic proteins are classified as "orphan" proteins with insufficient homologs for quality MSA construction.

## Core Idea

Given an evolutionary model that captures the distributional properties of a protein family, we should be able to generate plausible synthetic homologs to supplement real-world MSAs. This addresses sampling bias computationally: instead of waiting for nature to be sequenced, we simulate the evolutionary diversity that should exist.

## Existing Work

### MSA-Generator (MSAGen) -- NeurIPS 2024
- **Paper**: "MSA Generation with Seqs2Seqs Pretraining: Advancing Protein Structure Predictions"
- **Code**: https://github.com/lezhang7/MSAGen
- **Approach**: Seq2Seq pretraining to generate virtual MSAs from single/few sequences
- **Key result**: Virtual MSAs improved AlphaFold2 LDDT by up to +61 points for orphan proteins
- **Key result**: In some cases, synthetic MSAs surpass real MSAs for structure prediction

### MSAGPT -- NeurIPS 2024
- **Paper**: "MSAGPT: Neural Prompting Protein Structure Prediction via MSA Generative Pre-Training"
- **Approach**: Autoregressive MSA generation framed as a prompting task
- **Target**: De novo MSA generation for proteins with limited alignment data

### MSA Transformer Generative (eLife 2023)
- **Paper**: "Generative power of a protein language model trained on multiple sequence alignments"
- **Approach**: Iterative masked language modeling on MSA Transformer
- **Key result**: Synthetic sequences score comparably to natural on homology, coevolution, structure
- **Key result**: Fully synthetic MSAs fed to AlphaFold produce similar structure scores to natural MSAs

### MSAFlow (2025)
- **Paper**: OpenReview 2025
- **Approach**: Statistical Flow Matching conditioned on compressed MSA representations
- **Modes**: Zero-shot (single sequence + ESM2), few-shot (shallow MSA augmentation), family-based
- **Key result**: Synthetic shallow MSAs achieve pLDDT 89.0 vs 91.6 for deep natural MSAs (6.5% storage)

### Viral-Specific Fine-Tuning (PeerJ 2025)
- **Paper**: "Fine-tuning protein language models unlocks the potential of underrepresented viral proteomes" (Sawhney et al.)
- **Finding**: Standard pLMs have systematic bias against viral proteins
- **Solution**: Fine-tuning on viral sequences improves embedding quality for downstream tasks

## Why Viral Proteins Are the Ideal Test Case

Viral proteins represent the hardest and most informative case for synthetic MSA augmentation:

- **Highest divergence rates** in biology -- protein families span enormous sequence space
- **Functional conservation despite sequence divergence** -- same fold, same function, <20% identity
- **Most affected by sampling bias** -- most viral diversity is environmental and unsequenced
- **Clear evaluation metric** -- does augmentation move proteins from dark matter to annotated?
- **Practical impact** -- better viral protein annotation directly benefits pandemic preparedness, drug discovery, vaccine design

## Connection to vHold

This is a separate project but shares infrastructure with vHold:

### Shared components
- ProstT5/ProtT5 model ecosystem (encoder for embeddings, decoder for 3Di)
- BFVD + Viro3D as curated viral protein references (sequences + structures)
- Foldseek for structural search validation
- Ground truth from 5 case studies for benchmarking
- Dark matter report identifies target proteins (shallow/no MSAs)

### Experimental validation loop
1. Identify vHold dark matter proteins (no confident structural hit)
2. Generate synthetic MSA using MSAGen/MSAFlow/MSAGPT
3. Feed augmented MSA to ColabFold for improved structure prediction
4. Convert predicted structure to 3Di
5. Search with Foldseek -- does augmented MSA produce better structural hits?
6. Measure: how many proteins move from dark matter to annotated?

### Potential integration
- vHold could optionally accept pre-augmented MSAs as input
- Or: a sibling tool generates augmented MSAs, feeds them to ColabFold, and the resulting structures feed into vHold's structural search
- The embedding triage system could identify which proteins need MSA augmentation (those distant from all reference embeddings)

## Research Questions

1. **Which generative approach works best for viral proteins?** MSAGen, MSAGPT, MSAFlow, and MSA Transformer generation have different strengths. Viral proteins' extreme divergence may favor some approaches over others.

2. **Does viral-specific fine-tuning improve synthetic MSA quality?** Sawhney et al. show that generic pLMs underperform on viral proteins. Would fine-tuning the MSA generator on viral protein families produce better synthetic homologs?

3. **What is the relationship between embedding distance and MSA augmentation benefit?** The embedding triage framework could predict which proteins will benefit most from MSA augmentation -- those in the "dark zone" of embedding space.

4. **Can synthetic MSAs be validated structurally?** If we have Foldseek structural search results, we can check whether proteins predicted from augmented MSAs match the same structural families as their natural homologs.

5. **Is there a feedback loop?** Synthetic sequences that produce good structures could be added back to the reference database, iteratively improving the system.

## Project Structure

Proposed as a sibling project to vHold:

```
vHold ecosystem:
  vhold/          -- Viral protein annotation (current project)
  vmsagen/        -- Synthetic MSA augmentation for viral proteins (future)
    |
    +-- Uses vHold's databases and evaluation framework
    +-- Generates augmented MSAs for orphan viral proteins
    +-- Outputs feed into ColabFold or directly into vHold
```

## Relationship to Other vHold Directions

- **Embedding triage**: Identifies which proteins are in dark matter (candidates for MSA augmentation)
- **BitNet/quantization**: Faster ProstT5 inference makes the validation loop more practical
- **Pre-computed reference embeddings**: Provide the embedding space map showing where dark matter proteins fall relative to characterized families
- **LLM classification**: Can classify proteins that become annotatable after MSA augmentation

## References

- MSAGen: https://github.com/lezhang7/MSAGen
- MSAGen paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/694be3548697e9cc8999d45e8d16fe1e-Abstract-Conference.html
- MSAGPT: https://arxiv.org/html/2406.05347v1
- MSA Transformer generative: https://elifesciences.org/articles/79854
- MSAFlow: https://openreview.net/forum?id=74iQpDRwur
- Viral pLM fine-tuning: https://peerj.com/articles/19919/
- RNA virus structurome dark matter: https://journals.asm.org/doi/10.1128/mbio.03200-24
- Viral dark matter review: https://pubmed.ncbi.nlm.nih.gov/41264852/
