# Future Case Study Options

This document tracks potential case studies for future development.

## Completed Case Studies

| # | Name | Purpose | Status |
|---|------|---------|--------|
| 1 | SARS-CoV-2 | Pipeline validation | Complete |
| 2 | Remote Homology | Demonstrate low-identity annotation | Complete |
| 3 | Metagenomic Dark Matter | Novel protein discovery (palmdb) | Complete |

## Planned Future Case Studies

### crAss-like Phage ORFans

**Purpose**: Annotate unknown proteins from the most abundant phages in the human gut

**Source**: crAssphage and related genomes
- Dutilh et al., Nature Communications 2014 (original crAssphage)
- Guerin et al., Nature Communications 2018 (crAss-like phages)
- Yutin et al., Nature Microbiology 2018 (expanded crAss-like diversity)

**Why interesting**:
- crAssphage is the most abundant virus in the human gut
- ~50% of ORFs have no detectable homologs
- Biologically important for understanding gut microbiome
- Many proteins remain completely uncharacterized

**Proteins to test**:
- ORF1-ORF100 from crAssphage genomes
- Focus on ORFs with no Pfam/InterPro matches
- Compare vHold predictions to any recent characterizations

**Validation approach**:
- Some proteins have been characterized since discovery
- Can compare vHold predictions to recent literature
- Structural similarity to known phage proteins = plausible annotation

---

### Giant Virus ORFans (Nucleocytoviricota)

**Purpose**: Annotate unknown proteins from giant viruses with massive genomes

**Source**: NCLDV/Nucleocytoviricota genomes
- Mimivirus, Pandoravirus, Pithovirus, etc.
- IMG/VR giant virus genomes

**Why interesting**:
- Giant viruses have 100-2500 genes
- 50-90% of genes have no detectable homologs
- Some proteins may have eukaryotic origins
- Potential for discovering novel protein folds

**Proteins to test**:
- ORFans from well-characterized giant viruses
- Proteins with no Pfam/InterPro/CDD matches
- Focus on medium-sized proteins (100-500 aa)

**Challenges**:
- Large proteins = slow ProstT5 prediction
- Very divergent sequences
- May find eukaryotic rather than viral homologs

**Validation approach**:
- Check if predictions match any characterized homologs
- Structural similarity to known viral proteins
- Compare to experimental studies of specific ORFs

---

### Recent Phage Discoveries (2024-2025)

**Purpose**: Annotate proteins from phages discovered after BFVD was built

**Source**:
- INPHARED database (updated monthly)
- Recent publications in Nature, Cell, ISME, etc.
- GenBank phage submissions from 2024+

**Why interesting**:
- Guaranteed to not be in BFVD (too recent)
- Can compare vHold predictions to paper characterizations
- Tests real-world discovery scenario

**Proteins to test**:
- Structural proteins from novel phage families
- Hypothetical proteins with no annotation
- Proteins the authors couldn't annotate

**Validation approach**:
- Compare to authors' annotations
- Check if vHold finds what authors missed
- Quantify improvement over BLAST-only approach

---

### Archaeal Virus Proteins

**Purpose**: Annotate proteins from understudied archaeal viruses

**Source**:
- Archaeal virus genomes from NCBI/IMG
- Recent discoveries from extreme environments

**Why interesting**:
- Archaeal viruses are poorly characterized
- Unique morphologies (spindle, bottle, etc.)
- May have novel protein folds
- Distant from bacterial phages

**Proteins to test**:
- Capsid/structural proteins
- Unknown ORFs from spindle-shaped viruses
- Proteins from hyperthermophilic virus isolates

**Challenges**:
- Limited reference data for validation
- May be too divergent for any database

---

### Marine Virus Diversity

**Purpose**: Annotate proteins from marine viral metagenomes

**Source**:
- GOV 2.0 (Global Ocean Virome)
- Tara Oceans viral contigs
- Deep-sea viral metagenomes

**Why interesting**:
- Marine viruses are incredibly diverse
- Many novel families with no cultured representatives
- Important for understanding ocean ecosystems

**Proteins to test**:
- Major capsid proteins from novel viral families
- Auxiliary metabolic genes (AMGs)
- Proteins from giant phages (jumbo phages)

**Validation approach**:
- Structural match to known viral folds
- AMG predictions validated by metabolic context
- Compare to GOV 2.0 annotations

---

## Selection Criteria for Future Studies

When selecting proteins for case studies, prioritize:

1. **Novelty**: Proteins NOT in BFVD or Viro3D
2. **Validateability**: Some ground truth or validation method
3. **Biological interest**: Relevant to active research areas
4. **Diversity**: Cover different viral types/hosts
5. **Practicality**: Reasonable protein sizes, sufficient quality

## Data Sources

| Source | URL | Content |
|--------|-----|---------|
| palmdb | https://github.com/ababaian/palmdb | Novel RdRps |
| INPHARED | https://github.com/RyanCook94/inphared | Updated phage DB |
| IMG/VR | https://img.jgi.doe.gov/vr/ | Metagenome viruses |
| GOV 2.0 | https://datacommons.cyverse.org/browse/iplant/home/shared/iVirus/GOV2.0 | Ocean viromes |
| crAssphage | NCBI NC_024711 | Gut phage |
| PHROGs | https://phrogs.lmge.uca.fr/ | Phage protein families |
