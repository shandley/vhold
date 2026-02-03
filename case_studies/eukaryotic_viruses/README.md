# Case Study 5: Eukaryotic Virus Proteins

Demonstrates vHold's ability to annotate proteins from **mammalian viruses** - the key capability that distinguishes vHold from pHold (which targets phages).

## Background

While pHold excels at phage protein annotation, many important viruses infect eukaryotic hosts:
- Emerging zoonotic viruses (Nipah, Hendra, Ebola)
- Vaccine-preventable diseases (Measles, Rabies)
- Pandemic threats (Coronaviruses, Flaviviruses)

This case study tests vHold on proteins from multiple mammalian virus families.

## Virus Families Included

| Family | Viruses | Genome | Proteins |
|--------|---------|--------|----------|
| **Paramyxoviridae** | Nipah, Hendra, Measles | (-) ssRNA | 11 |
| **Filoviridae** | Ebola | (-) ssRNA | 6 |
| **Rhabdoviridae** | Rabies, VSV | (-) ssRNA | 5 |
| **Hantaviridae** | Hantaan | (-) ssRNA | 2 |
| **Togaviridae** | Chikungunya | (+) ssRNA | 1 |
| **Coronaviridae** | MERS-CoV | (+) ssRNA | 3 |
| **Flaviviridae** | Dengue | (+) ssRNA | 1 |

## Test Dataset

| Metric | Value |
|--------|-------|
| Total proteins | 29 |
| Virus families | 7 |
| Length range | 166-7078 aa |
| Structural proteins | 18 |
| Regulatory/accessory | 8 |
| ORFans (NCBI hypothetical) | 2 |

## Key Proteins

### High-Priority Targets (Public Health)

| Protein | Virus | Function | Why Important |
|---------|-------|----------|---------------|
| NP_112026.1 | Nipah | Fusion protein | BSL-4 pathogen, vaccine target |
| NP_066246.1 | Ebola | Spike glycoprotein | Therapeutic target |
| NP_056796.1 | Rabies | Glycoprotein | Vaccine antigen |

### Accessory Proteins (Often Poorly Annotated)

| Protein | Virus | Function | Challenge |
|---------|-------|----------|-----------|
| NP_112023.1 | Nipah | V protein | Interferon antagonist |
| NP_112024.1 | Nipah | C protein | Accessory, less characterized |
| NP_066249.1 | Ebola | VP30 | Transcription factor |
| NP_066250.1 | Ebola | VP24 | Immune evasion |

### True ORFans (NCBI "hypothetical")

| Protein | Virus | Actual Function |
|---------|-------|-----------------|
| NP_941977.1 | Hantaan | Nucleocapsid (labeled hypothetical) |
| NP_941978.1 | Hantaan | Glycoprotein precursor (labeled hypothetical) |

## Success Criteria

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Annotation rate | >70% | vHold finds hits for most proteins |
| Category accuracy | >60% | Structural proteins identified correctly |
| Accessory proteins | >50% | V/C proteins get regulatory category |
| Hantavirus ORFans | 2/2 | True ORFans correctly annotated |

## Running the Case Study

```bash
cd case_studies/eukaryotic_viruses

# Fetch proteins (already done)
python fetch_proteins.py

# Prepare test set (already done)
python prepare_test_set.py

# Run vHold
vhold run -i test_proteins.fasta -o results/ -t 4 --device cpu

# Analyze results
python analyze_results.py
```

## Expected Results

### If vHold Works Well

- Structural proteins (F, G, N, M, VP40) correctly identified
- Accessory proteins (V, C, VP24, VP30) get regulatory/unknown category
- Hantavirus "hypothetical" proteins identified as structural
- Cross-family homologs found (e.g., Nipah F similar to Hendra F)

### Challenging Cases

- **Polyproteins** (Dengue, Chikungunya): Require processing, may get ambiguous annotations
- **L proteins** (polymerases): Very large, may be slow
- **Accessory proteins**: May not have structural homologs in database

## References

1. Wang LF, et al. (2000). The exceptionally large genome of Hendra virus. Virology 278:587-600.
2. Chua KB, et al. (2000). Nipah virus: a recently emergent deadly paramyxovirus. Science 288:1432-1435.
3. Feldmann H, et al. (2020). Ebola virus disease. Nat Rev Dis Primers 6:13.
4. Rupprecht CE, et al. (2002). Rabies re-examined. Lancet Infect Dis 2:327-343.

## Files

| File | Description |
|------|-------------|
| `all_eukaryotic_proteins.fasta` | Raw proteins from 19 virus genomes |
| `test_proteins.fasta` | 29 curated test proteins |
| `ground_truth.json` | Expected annotations from literature |
| `fetch_proteins.py` | NCBI protein fetcher |
| `prepare_test_set.py` | Test set curation script |
| `analyze_results.py` | Results analysis script |
