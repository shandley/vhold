# Candidate Proteins for Remote Homology Case Study

## Selection Criteria

Proteins must be:
1. **Functionally characterized** - Known function for ground truth validation
2. **Sequence-divergent** - <30% identity to BFVD/Viro3D entries
3. **Structurally predictable** - Ordered domains (pLDDT > 70 expected)
4. **Experimentally validated** - Ideally with crystal/cryo-EM structure

## Priority 1: RNA Phages (Leviviridae)

**Rationale**: Extremely divergent from DNA phages that dominate viral databases.
Well-characterized since 1960s (first RNA genomes sequenced).

### MS2 Bacteriophage
| Protein | UniProt | Length | Function | PDB |
|---------|---------|--------|----------|-----|
| Coat protein | P03612 | 129 | Capsid assembly | 2MS2 |
| Maturation protein | P03610 | 393 | Host attachment | 5TC1 |
| Lysis protein | P03609 | 75 | Cell lysis | - |
| Replicase | P00585 | 545 | RNA replication | - |

### Qβ Bacteriophage
| Protein | UniProt | Length | Function | PDB |
|---------|---------|--------|----------|-----|
| Coat protein | P03615 | 132 | Capsid | 1QBE |
| A1 protein | P03616 | 329 | Minor capsid | - |
| Replicase β | P03634 | 589 | RdRp subunit | 3MMP |
| Maturation A2 | P03617 | 420 | Entry/lysis | 5MNT |

### FASTA retrieval
```bash
# Download from UniProt
curl -s "https://rest.uniprot.org/uniprotkb/P03612.fasta" >> rna_phage_proteins.fasta
curl -s "https://rest.uniprot.org/uniprotkb/P03610.fasta" >> rna_phage_proteins.fasta
curl -s "https://rest.uniprot.org/uniprotkb/P03615.fasta" >> rna_phage_proteins.fasta
curl -s "https://rest.uniprot.org/uniprotkb/P03634.fasta" >> rna_phage_proteins.fasta
```

## Priority 2: Fungal dsRNA Viruses (Totiviridae)

**Rationale**: dsRNA viruses infecting fungi/protozoa. Very distant from
animal viruses. Simple genomes with characterized proteins.

### Saccharomyces cerevisiae virus L-A
| Protein | UniProt | Length | Function | PDB |
|---------|---------|--------|----------|-----|
| Gag (capsid) | P32503 | 680 | Coat + decapping | 1M1C |
| Gag-Pol | P21525 | 1427 | RdRp fusion | - |

### Helminthosporium victoriae virus 190S
| Protein | UniProt | Length | Function |
|---------|---------|--------|----------|
| Capsid | Q9YMT4 | 778 | Structural |
| RdRp | Q9YMT3 | 900 | Replication |

## Priority 3: Plant Virus Movement Proteins

**Rationale**: Unique to plant viruses - no homologs in animal/bacterial systems.
Critical for cell-to-cell movement through plasmodesmata.

### 30K Superfamily (TMV-like)
| Protein | Virus | UniProt | Length | Function |
|---------|-------|---------|--------|----------|
| 30K MP | TMV | P03582 | 268 | Plasmodesmata transport |
| P30 | TSWV | P36291 | 301 | Movement |

### Triple Gene Block
| Protein | Virus | UniProt | Length | Function |
|---------|-------|---------|--------|----------|
| TGB1 | PVX | P10410 | 231 | Helicase-like |
| TGB2 | PVX | P10411 | 122 | ER association |
| TGB3 | PVX | P10412 | 67 | Membrane protein |

## Priority 4: Archaeal Viruses

**Rationale**: Most divergent viral lineage. Infect extremophiles.
Some have unique morphologies not seen elsewhere.

### Sulfolobus turreted icosahedral virus (STIV)
| Protein | UniProt | Length | Function | Notes |
|---------|---------|--------|----------|-------|
| B345 (MCP) | Q6GZL8 | 345 | Major capsid | Double-barrel fold |
| A197 | Q6GZM5 | 197 | Turret protein | |
| C381 | Q6GZL0 | 381 | ATPase | Packaging |

### Sulfolobus spindle-shaped virus 1 (SSV1)
| Protein | UniProt | Length | Function |
|---------|---------|--------|----------|
| VP1 | P22525 | 73 | Capsid |
| VP3 | P22526 | 92 | Capsid |
| Integrase | P22527 | 335 | Integration |

## Priority 5: Divergent Polymerases

**Rationale**: RdRps have conserved function but divergent sequences.
Good for testing functional annotation transfer at low identity.

### Narnaviridae (simplest RNA viruses)
- Just an RdRp, no capsid
- Found in fungi
- Example: Saccharomyces 20S RNA narnavirus (P22348)

### Hypoviridae (fungal hypovirulence)
- CHV1 RdRp: Well-characterized, divergent
- UniProt: P22535

## Ground Truth Sources

For each protein, ground truth should include:
1. **Pfam domains** - From InterPro/Pfam
2. **GO terms** - From UniProt-GOA
3. **Functional category** - Manual curation
4. **Structure** - PDB ID if available (for validation)

## Suggested Test Set (20 proteins)

| # | Protein | Source | Category | Expected Challenge |
|---|---------|--------|----------|-------------------|
| 1 | MS2 coat | P03612 | structural | RNA phage |
| 2 | MS2 maturation | P03610 | structural | RNA phage |
| 3 | MS2 replicase | P00585 | replication | RNA phage |
| 4 | Qβ coat | P03615 | structural | RNA phage |
| 5 | Qβ replicase | P03634 | replication | RNA phage |
| 6 | L-A Gag | P32503 | structural | Fungal dsRNA |
| 7 | L-A RdRp | P21525 | replication | Fungal dsRNA |
| 8 | TMV 30K | P03582 | movement | Plant virus |
| 9 | PVX TGB1 | P10410 | movement | Plant virus |
| 10 | STIV MCP | Q6GZL8 | structural | Archaeal |
| 11 | SSV1 integrase | P22527 | nuclease | Archaeal |
| 12 | Narnavirus RdRp | P22348 | replication | Minimal virus |
| 13 | CHV1 RdRp | P22535 | replication | Fungal |
| 14 | TSWV MP | P36291 | movement | Plant virus |
| 15 | Qβ A2 | P03617 | lysis | RNA phage |
| 16 | HvV190S capsid | Q9YMT4 | structural | Fungal dsRNA |
| 17 | PVX TGB2 | P10411 | movement | Plant virus |
| 18 | STIV ATPase | Q6GZL0 | packaging | Archaeal |
| 19 | MS2 lysis | P03609 | lysis | RNA phage |
| 20 | SSV1 VP1 | P22525 | structural | Archaeal |

## Quick Start

```bash
# Create test set FASTA
cd case_studies/remote_homology

# Download proteins (requires internet)
for acc in P03612 P03610 P00585 P03615 P03634 P32503 P21525 P03582 P10410 Q6GZL8; do
    curl -s "https://rest.uniprot.org/uniprotkb/${acc}.fasta" >> divergent_proteins.fasta
done

# Run vHold
vhold run -i divergent_proteins.fasta -o results/ --device cuda -t 4

# Analyze identity distribution
python analyze_identity.py results/
```
