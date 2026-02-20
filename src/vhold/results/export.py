"""Export vHold results to metagenomic tool formats.

Supports: anvi'o functions-txt, vConTACT2 gene2genome, vConTACT3,
DRAM-v supplementary annotations, and GFF3.
"""

import re
from pathlib import Path

import pandas as pd

from vhold.results.consensus import ConsensusResult
from vhold.utils.logging import get_logger

logger = get_logger(__name__)

# Supported export format names
EXPORT_FORMATS = ["anvio", "vcontact2", "vcontact3", "dramv", "gff3"]

# Prodigal convention: contig_N where N is a number
_PRODIGAL_SUFFIX_RE = re.compile(r"^(.+)_(\d+)$")

# geNomad provirus format: seq|provirus_start_end_N
_GENOMAD_PROVIRUS_RE = re.compile(r"^(.+\|provirus_\d+_\d+)_(\d+)$")


def contig_id_from_protein_id(
    protein_id: str,
    contig: str | None = None,
) -> str:
    """Derive contig/genome ID from a protein identifier.

    Priority:
    1. Explicit contig (from GenBank/GFF ProteinRecord or TSV column)
    2. geNomad provirus format: strip trailing _N after provirus coordinates
    3. Prodigal convention: strip trailing _N suffix
    4. Return protein_id as-is if no pattern matches

    Args:
        protein_id: Protein identifier string
        contig: Explicit contig ID if available

    Returns:
        Derived contig/genome identifier
    """
    if contig:
        return contig

    # Try geNomad provirus format first (more specific)
    m = _GENOMAD_PROVIRUS_RE.match(protein_id)
    if m:
        return m.group(1)

    # Try Prodigal convention (general _N suffix)
    m = _PRODIGAL_SUFFIX_RE.match(protein_id)
    if m:
        return m.group(1)

    return protein_id


def _to_dataframe(
    results: dict[str, ConsensusResult] | pd.DataFrame,
) -> pd.DataFrame:
    """Convert ConsensusResult dict to DataFrame, or pass through."""
    if isinstance(results, pd.DataFrame):
        return results
    rows = [result.to_dict() for result in results.values()]
    return pd.DataFrame(rows)


def load_results_tsv(tsv_path: Path | str) -> pd.DataFrame:
    """Load a vHold results TSV for export."""
    return pd.read_csv(tsv_path, sep="\t")


def _safe_str(val) -> str:
    """Convert a value to string, treating NaN/None as empty."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val)


# ============================================================
# anvi'o functions-txt
# ============================================================


def export_anvio(
    results: dict[str, ConsensusResult] | pd.DataFrame,
    output_path: Path | str,
    source_prefix: str = "vHold",
) -> Path:
    """Export annotations in anvi'o functions-txt format.

    Generates one row per annotation source per gene:
    - {prefix}_structural: primary structural homology hit
    - {prefix}_Pfam: Pfam domain if present
    - {prefix}_GO_BP: GO biological process
    - {prefix}_GO_MF: GO molecular function
    - {prefix}_category: functional category (always present)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    df = _to_dataframe(results)

    for idx, (_, row) in enumerate(df.iterrows()):
        gene_id = idx

        # Primary structural hit
        desc = _safe_str(row.get("description", ""))
        if desc and desc != "hypothetical protein":
            evalue = row.get("primary_evalue", "")
            rows.append({
                "gene_callers_id": gene_id,
                "source": f"{source_prefix}_structural",
                "accession": _safe_str(row.get("primary_target", "")),
                "function": desc,
                "e_value": evalue if _safe_str(evalue) else 0,
            })

        # Pfam
        pfam = _safe_str(row.get("pfam", ""))
        if pfam:
            first_pfam = pfam.split(";")[0].strip()
            rows.append({
                "gene_callers_id": gene_id,
                "source": f"{source_prefix}_Pfam",
                "accession": first_pfam,
                "function": pfam,
                "e_value": 0,
            })

        # GO BP
        go_bp = _safe_str(row.get("go_bp", ""))
        go_bp_ids = _safe_str(row.get("go_bp_ids", ""))
        if go_bp:
            rows.append({
                "gene_callers_id": gene_id,
                "source": f"{source_prefix}_GO_BP",
                "accession": go_bp_ids,
                "function": go_bp,
                "e_value": 0,
            })

        # GO MF
        go_mf = _safe_str(row.get("go_mf", ""))
        go_mf_ids = _safe_str(row.get("go_mf_ids", ""))
        if go_mf:
            rows.append({
                "gene_callers_id": gene_id,
                "source": f"{source_prefix}_GO_MF",
                "accession": go_mf_ids,
                "function": go_mf,
                "e_value": 0,
            })

        # Functional category (always present)
        category = _safe_str(row.get("functional_category", "unknown"))
        rows.append({
            "gene_callers_id": gene_id,
            "source": f"{source_prefix}_category",
            "accession": category,
            "function": category,
            "e_value": 0,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Wrote anvi'o functions-txt: {output_path} ({len(out_df)} rows)")
    return output_path


# ============================================================
# vConTACT2 gene2genome.csv
# ============================================================


def export_vcontact2(
    results: dict[str, ConsensusResult] | pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Export gene-to-genome mapping for vConTACT2.

    Columns: protein_id, contig_id, keywords (CSV format).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    df = _to_dataframe(results)

    for _, row in df.iterrows():
        protein_id = row["query_id"]
        contig = row.get("contig", None)
        if isinstance(contig, float) and pd.isna(contig):
            contig = None
        contig_id = contig_id_from_protein_id(str(protein_id), contig)

        desc = _safe_str(row.get("description", "hypothetical protein"))
        category = _safe_str(row.get("functional_category", "unknown"))
        if desc and desc != "hypothetical protein":
            keywords = f"{desc}; {category}"
        else:
            keywords = category

        rows.append({
            "protein_id": protein_id,
            "contig_id": contig_id,
            "keywords": keywords,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)  # CSV, not TSV
    logger.info(f"Wrote vConTACT2 gene2genome: {output_path} ({len(out_df)} rows)")
    return output_path


# ============================================================
# vConTACT3 gene2genome.tsv + genome_lengths.tsv
# ============================================================


def export_vcontact3(
    results: dict[str, ConsensusResult] | pd.DataFrame,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Export gene-to-genome and genome lengths for vConTACT3.

    Produces two files:
    - gene2genome.tsv: protein_id, genome_id
    - genome_lengths.tsv: genome_id, length
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _to_dataframe(results)
    g2g_rows = []
    genome_lengths: dict[str, int] = {}

    for _, row in df.iterrows():
        protein_id = row["query_id"]
        contig = row.get("contig", None)
        if isinstance(contig, float) and pd.isna(contig):
            contig = None
        genome_id = contig_id_from_protein_id(str(protein_id), contig)

        g2g_rows.append({
            "protein_id": protein_id,
            "genome_id": genome_id,
        })

        prot_length = int(row.get("query_length", 0))
        genome_lengths[genome_id] = genome_lengths.get(genome_id, 0) + prot_length

    g2g_path = output_dir / "gene2genome.tsv"
    pd.DataFrame(g2g_rows).to_csv(g2g_path, sep="\t", index=False)

    gl_path = output_dir / "genome_lengths.tsv"
    gl_rows = [{"genome_id": gid, "length": length}
               for gid, length in genome_lengths.items()]
    pd.DataFrame(gl_rows).to_csv(gl_path, sep="\t", index=False)

    logger.info(f"Wrote vConTACT3 files: {g2g_path}, {gl_path}")
    return {"gene2genome": g2g_path, "genome_lengths": gl_path}


# ============================================================
# DRAM-v supplementary annotations
# ============================================================


def export_dramv(
    results: dict[str, ConsensusResult] | pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Export DRAM-v-compatible supplementary annotations.

    Keyed by scaffold + gene_position, designed to be mergeable
    with DRAM-v annotations.tsv.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    df = _to_dataframe(results)
    scaffold_gene_counts: dict[str, int] = {}

    for _, row in df.iterrows():
        protein_id = row["query_id"]
        contig = row.get("contig", None)
        if isinstance(contig, float) and pd.isna(contig):
            contig = None
        scaffold = contig_id_from_protein_id(str(protein_id), contig)

        scaffold_gene_counts[scaffold] = scaffold_gene_counts.get(scaffold, 0) + 1
        gene_position = scaffold_gene_counts[scaffold]

        start = row.get("start", "")
        end = row.get("end", "")
        strand = row.get("strand", "")

        if strand == 1 or strand == "1":
            strandedness = "+"
        elif strand == -1 or strand == "-1":
            strandedness = "-"
        else:
            strandedness = ""

        rows.append({
            "scaffold": scaffold,
            "gene_position": gene_position,
            "start_position": _safe_str(start),
            "end_position": _safe_str(end),
            "strandedness": strandedness,
            "vhold_description": _safe_str(row.get("description", "")),
            "vhold_category": _safe_str(row.get("functional_category", "unknown")),
            "vhold_confidence": _safe_str(row.get("confidence_level", "none")),
            "vhold_pfam": _safe_str(row.get("pfam", "")),
            "vhold_go_bp": _safe_str(row.get("go_bp", "")),
            "vhold_go_mf": _safe_str(row.get("go_mf", "")),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Wrote DRAM-v supplementary: {output_path} ({len(out_df)} rows)")
    return output_path


# ============================================================
# GFF3 output
# ============================================================


def _gff3_escape(text: str) -> str:
    """Escape special characters for GFF3 attribute values."""
    return text.replace(";", "%3B").replace("=", "%3D").replace(",", "%2C")


def export_gff3(
    results: dict[str, ConsensusResult] | pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Export annotations as GFF3 features.

    Only proteins with genomic coordinates (contig, start, end, strand)
    are included.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = _to_dataframe(results)
    count = 0

    with open(output_path, "w") as f:
        f.write("##gff-version 3\n")

        for _, row in df.iterrows():
            contig = row.get("contig", None)
            start = row.get("start", None)
            end = row.get("end", None)
            strand_val = row.get("strand", None)

            if contig is None or (isinstance(contig, float) and pd.isna(contig)):
                continue
            if start is None or (isinstance(start, float) and pd.isna(start)):
                continue
            if end is None or (isinstance(end, float) and pd.isna(end)):
                continue

            if strand_val == 1 or strand_val == "1":
                strand = "+"
            elif strand_val == -1 or strand_val == "-1":
                strand = "-"
            else:
                strand = "."

            protein_id = row["query_id"]
            desc = _safe_str(row.get("description", "hypothetical protein"))
            category = _safe_str(row.get("functional_category", "unknown"))
            confidence = _safe_str(row.get("confidence_level", "none"))
            evalue = row.get("primary_evalue", ".")

            attrs = [
                f"ID={protein_id}",
                f"Name={_gff3_escape(desc)}",
                f"vhold_category={category}",
                f"vhold_confidence={confidence}",
            ]

            pfam = _safe_str(row.get("pfam", ""))
            if pfam:
                attrs.append(f"Dbxref=Pfam:{_gff3_escape(pfam)}")

            go_ids = []
            go_bp_ids = _safe_str(row.get("go_bp_ids", ""))
            go_mf_ids = _safe_str(row.get("go_mf_ids", ""))
            if go_bp_ids:
                go_ids.extend(go_bp_ids.replace("; ", ",").split(","))
            if go_mf_ids:
                go_ids.extend(go_mf_ids.replace("; ", ",").split(","))
            if go_ids:
                attrs.append(f"Ontology_term={','.join(go_ids)}")

            score = evalue if _safe_str(evalue) and evalue != "." else "."
            feat_type = _safe_str(row.get("feature_type", "CDS")) or "CDS"

            f.write(
                f"{contig}\tvHold\t{feat_type}\t{int(start)}\t{int(end)}\t{score}\t"
                f"{strand}\t.\t{';'.join(attrs)}\n"
            )
            count += 1

    logger.info(f"Wrote GFF3: {output_path} ({count} features)")
    return output_path


# ============================================================
# Dispatcher
# ============================================================


def export_all_formats(
    results: dict[str, ConsensusResult] | pd.DataFrame,
    output_dir: Path | str,
    formats: list[str] | None = None,
    prefix: str = "vhold",
) -> dict[str, Path | dict[str, Path]]:
    """Export results in one or more formats.

    Args:
        results: ConsensusResult dict or DataFrame
        output_dir: Output directory
        formats: List of format names, or None/"all" for all
        prefix: Output file prefix

    Returns:
        Dict mapping format name to output path(s)
    """
    if formats is None or "all" in formats:
        formats = list(EXPORT_FORMATS)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, Path | dict[str, Path]] = {}

    for fmt in formats:
        if fmt == "anvio":
            path = export_anvio(results, output_dir / f"{prefix}_anvio_functions.txt")
            output_files["anvio"] = path
        elif fmt == "vcontact2":
            path = export_vcontact2(results, output_dir / f"{prefix}_vcontact2_gene2genome.csv")
            output_files["vcontact2"] = path
        elif fmt == "vcontact3":
            paths = export_vcontact3(results, output_dir / "vcontact3")
            output_files["vcontact3"] = paths
        elif fmt == "dramv":
            path = export_dramv(results, output_dir / f"{prefix}_dramv_annotations.tsv")
            output_files["dramv"] = path
        elif fmt == "gff3":
            path = export_gff3(results, output_dir / f"{prefix}_annotations.gff3")
            output_files["gff3"] = path
        else:
            logger.warning(f"Unknown export format: {fmt}")

    return output_files
