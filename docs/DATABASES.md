# vHold Database Documentation

## Overview

vHold searches two complementary viral structure databases:

| Database | Source | Structures | Annotations | Coverage |
|----------|--------|------------|-------------|----------|
| **BFVD** | AlphaFold2 + TrEMBL | 351,242 | UniProt metadata | Comprehensive |
| **Viro3D** | Experimental + ColabFold | 85,162 | Curated | High quality |

## BFVD (Big Fantastic Virus Database)

### Source
- AlphaFold2 structure predictions
- Proteins from **TrEMBL** (unreviewed UniProt entries)
- Automatically predicted from viral genome sequences

### Key Characteristics

**Strengths**:
- Comprehensive coverage of viral protein diversity
- Consistent structure prediction methodology
- Large search space for finding homologs

**Limitations**:
- Based on TrEMBL, not Swiss-Prot (reviewed entries)
- Many entries are fragments or partial sequences
- Annotations are computationally derived, not manually curated

### Implication for Users

When querying **Swiss-Prot reviewed proteins** (e.g., `sp|P00585|RDRP_BPMS2`), you may see:
- Hits at 97-99% identity to TrEMBL entries of the same protein
- NOT 100% identity because Swiss-Prot entries aren't in BFVD directly
- This is a database composition artifact, not a pipeline limitation

**Example**:
```
Query: P00585 (MS2 RdRp, Swiss-Prot, 545 aa)
Hit:   D0U1E6 (MS2 RdRp, TrEMBL, 429 aa fragment)
Identity: 97.9%
```

Both are MS2 RdRp - the 2.1% difference reflects the fragment nature of the TrEMBL entry.

## Viro3D

### Source
- Experimental structures from PDB
- High-quality ColabFold predictions
- Curated from 4,400+ virus species

### Key Characteristics

**Strengths**:
- Higher annotation quality (curated)
- Experimental structures where available
- Rich metadata (Gene3D domains, structure quality scores)

**Limitations**:
- Smaller database (85K vs 351K)
- May not cover all viral diversity

### Complementary Value

Viro3D often finds hits at **lower identity** than BFVD because it contains structures from different viral species. These are true remote homologs:

```
Query: Phi6 RdRp (Cystoviridae)
BFVD hit: PhiYY RdRp @ 48% identity (different phage, same family)
Viro3D hit: Reovirus RdRp @ 11% identity (different virus order!)
```

## Novelty Classification

vHold classifies hits by "novelty" to help users understand the value of each annotation:

| Identity | Classification | Meaning | BLAST |
|----------|----------------|---------|-------|
| >95% | `database_match` | Same protein, different DB entry | Works |
| 70-95% | `close_homolog` | Related strain/variant | Works |
| 30-70% | `remote_homolog` | Functional transfer via structure | Marginal |
| <30% | `twilight_zone` | Novel structural similarity | **Fails** |

**Interpretation**:
- `database_match`: Confirms the protein is known (validation)
- `close_homolog`: Finds related proteins (expected)
- `remote_homolog`: **vHold adds value** - structure-based annotation
- `twilight_zone`: **vHold unique** - BLAST cannot find these

## Database Statistics

Identity distribution across Foldseek hits varies by query dataset. In general, the vast majority of structural hits are in the "twilight zone" (<30% identity) where BLAST fails, particularly for Viro3D which contains structures from diverse virus species. This is where vHold provides unique value.

## Recommendations

### For Validation Studies
- Expect ~97-99% identity hits for well-characterized proteins
- Use `novelty: database_match` to identify these cases
- True validation requires checking if annotation is correct, not just high identity

### For Discovery Studies
- Focus on `remote_homolog` and `twilight_zone` hits
- These represent annotations BLAST cannot provide
- Cross-database agreement at low identity is strong evidence

### For Novel Proteins (Metagenomics)
- Proteins NOT in databases will only find remote/twilight hits
- This is where vHold provides unique value
- Functional transfer at <30% identity enables annotation of "dark matter"
