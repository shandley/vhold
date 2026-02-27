# vHold — SUNSET

Last updated: 2026-02-27

## Project Status: SUNSET

**vHold is no longer under active development.** After strategic review:
- **pHold** already handles structural annotation (ProstT5 + Foldseek)
- **Phynteny** does synteny-based functional prediction (LSTM/transformer on 280K genomes, AUC>0.84)
- vHold's unique contribution (FANTASIA-style GO transfer) does not justify a standalone tool

**Portable components have been migrated to ViroSense** (`~/Code/tools/virosense/annotate/`), which uses Evo2 DNA foundation model for viral detection and integrates structural annotation via ColabFold + BFVD + Foldseek + FoldMason — replacing the ProstT5 dependency entirely.

## Portable Modules (migrated to ViroSense)

| Module | vHold source | ViroSense target | Migration status |
|--------|-------------|-----------------|:----------------:|
| Structure acquisition | — | `annotate/structure.py` | COMPLETE (new) |
| Foldseek PDB search | `features/foldseek.py` | `annotate/foldseek.py` | COMPLETE |
| FoldMason alignment | `features/foldmason.py` | `annotate/foldmason.py` | COMPLETE |
| Functional classification | `results/categories.py` | `annotate/categories.py` | COMPLETE |
| Gene calling (Pyrodigal-gv) | `features/genecall.py` | `annotate/genecall.py` | Pending |
| Metagenomic export | `results/export.py` | `annotate/export.py` | Pending |
| GO term resolution | `results/go_terms.py` | — | Pending |

## NOT Migrating (ProstT5-dependent)

ProstT5 encoder/decoder, embedding triage (436K DB), MLP classifiers, FANTASIA GO transfer, ONNX backend, LoRA/contrastive training, disorder pipeline (STARLING/metapredict).

## Reference

The vHold codebase contains ~870 tests and extensive implementation across 50+ modules. Source files remain available for reference but are not being maintained. Case studies are in `case_studies/` with T7 phage and crAssphage as priority examples.

```bash
# Run tests (reference only)
uv run pytest tests/ -v

# Run vHold (reference only)
uv run vhold run -i input.fasta -o output/ -t 8
```
