"""Functional category classification for viral proteins.

This module provides enhanced functional classification using:
1. Pfam domain annotations
2. GO biological process and molecular function terms
3. SUPERFAMILY structural classifications
4. Keyword-based fallback matching
"""

from dataclasses import dataclass
from typing import Optional

# =============================================================================
# KEYWORD-BASED CLASSIFICATION (original approach)
# =============================================================================

FUNCTIONAL_CATEGORIES = {
    "structural": [
        "capsid",
        "coat",
        "envelope",
        "spike",
        "matrix",
        "tail",
        "fiber",
        "portal",
        "shell",
        "head",
        "virion",
        "glycoprotein",
        "membrane protein",
        "nucleocapsid",
        "tegument",
        "baseplate",
        "assembly",  # Assembly proteins (e.g., MS2 maturation protein)
        "maturation",  # Maturation proteins
    ],
    "replication": [
        "polymerase",
        "replicase",
        "helicase",
        "primase",
        "rdrp",
        "reverse transcriptase",
        "rep protein",
        "nsp12",
        "nsp13",
    ],
    "protease": [
        "protease",
        "peptidase",
        "maturase",
        "3cl",
        "mpro",
        "nsp5",
    ],
    "nuclease": [
        "nuclease",
        "endonuclease",
        "exonuclease",
        "integrase",
        "recombinase",
        "ligase",
        "rnase",
        "dnase",
    ],
    "packaging": [
        "terminase",
        "packaging",
        "scaffold",
        "portal protein",
    ],
    "regulatory": [
        "transcription",
        "repressor",
        "activator",
        "regulator",
        "anti-repressor",
        "antirepressor",
        "translation enhancer",  # e.g., Rotavirus NSP3
        "translation effector",  # e.g., Rotavirus NSP3
        "effector protein",  # General effector proteins
    ],
    "movement": [
        "movement",
        "cell-to-cell",
        "transport protein",
        "movement protein",
        "plasmodesmata",  # Plant cell connections
        "tubule-forming",  # Movement protein characteristic
        " mp",  # Gene name for movement protein (with space to avoid false matches)
    ],
    "lysis": [
        "lysin",
        "holin",
        "endolysin",
        "spanin",
        "lysis",
    ],
}

# Terms that indicate unknown/hypothetical proteins
UNKNOWN_TERMS = [
    "hypothetical",
    "uncharacterized",
    "unknown",
    "duf",
    "unipr",
]

# =============================================================================
# PFAM DOMAIN TO CATEGORY MAPPING
# =============================================================================

PFAM_TO_CATEGORY = {
    # Structural proteins
    "structural": [
        "capsid",
        "coat protein",
        "matrix protein",
        "virion",
        "envelope",
        "glycoprotein",
        "spike",
        "nucleocapsid",
        "vp1", "vp2", "vp3", "vp4", "vp5", "vp6", "vp7",
        "major capsid",
        "minor capsid",
        "picornavirus capsid",
        "circovirus capsid",
        "polyomavirus coat",
        "parvovirus coat",
        "vesiculovirus matrix",
        "rhabdovirus spike",
        "herpes virus major capsid",
        "herpesvirus vp23",
        "herpesvirus glycoprotein",
        "adenovirus hexon",
        "adenovirus penton",
        "fiber protein",
        "tail protein",
        "baseplate",
        "portal protein",
        "head protein",
        "tegument",
        "membrane protein",
        "structural envelope",
        "assembly protein",  # e.g., MS2 maturation/assembly
        "maturation protein",
    ],
    # Replication machinery
    "replication": [
        "polymerase",
        "rdrp",
        "rna-dependent rna polymerase",
        "dna polymerase",
        "reverse transcriptase",
        "helicase",
        "rna helicase",
        "dna helicase",
        "primase",
        "dna primase",
        "replicase",
        "rep protein",
        "parvovirus non-structural",
        "ns1",
        "nsp12", "nsp13",
        "viral helicase",
        "superfamily 1 rna helicase",
        "superfamily 2 helicase",
        # Nucleotide metabolism (required for replication)
        "thymidine kinase",
        "thymidylate",
        "ribonucleotide reductase",
        "dutpase",  # Also in nuclease, but fits replication too
        "uracil dna glycosylase",
    ],
    # Proteases
    "protease": [
        "protease",
        "peptidase",
        "proteinase",
        "maturase",
        "assemblin",
        "3cl protease",
        "mpro",
        "nsp5",
        "serine endopeptidase",
        "cysteine protease",
        "cysteine proteinase",
        "cysteine-type peptidase",
        "metalloprotease",
        "aspartic protease",
        "otu",
        "ovarian tumor",
    ],
    # Nucleases
    "nuclease": [
        "nuclease",
        "endonuclease",
        "exonuclease",
        "rnase",
        "dnase",
        "integrase",
        "recombinase",
        "ligase",
        "dna ligase",
        "rna ligase",
        "dutpase",
        "uracil dna glycosylase",
        "dna/rna non-specific endonuclease",
        "pd-(d/e)xk",
        "giy-yig",
        "hnh",
        "restriction enzyme",
    ],
    # Packaging
    "packaging": [
        "terminase",
        "packaging",
        "scaffold",
        "dna packaging",
        "genome packaging",
        "portal",
        "ul6",
    ],
    # Regulatory
    "regulatory": [
        "transcription",
        "transcriptional regulator",
        "repressor",
        "activator",
        "regulator",
        "anti-repressor",
        "trans-activator",
        "immediate-early",
        "protein kinase",
        "kinase domain",
        "translation enhancer",
        "translation effector",
        "translation regulation",
    ],
    # Movement (plant viruses)
    "movement": [
        "movement protein",
        "cell-to-cell",
        "transport protein",
        "tubule-forming",
    ],
    # Lysis (bacteriophages)
    "lysis": [
        "lysin",
        "holin",
        "endolysin",
        "spanin",
        "lysis",
        "lysozyme",
        "muramidase",
        "amidase",
        "peptidoglycan",
    ],
}

# =============================================================================
# GO BIOLOGICAL PROCESS TO CATEGORY MAPPING
# =============================================================================

GO_BP_TO_CATEGORY = {
    "structural": [
        "viral capsid assembly",
        "virion assembly",
        "viral particle assembly",
        "capsid assembly",
    ],
    "replication": [
        "dna replication",
        "viral genome replication",
        "viral dna genome replication",
        "viral rna genome replication",
        "rna-templated transcription",
        "dna-templated transcription",
        "telomere maintenance",
        # Nucleotide metabolism (supports replication)
        "tmp biosynthetic process",
        "thymidine biosynthetic process",
        "nucleotide biosynthetic process",
        "deoxyribonucleotide biosynthetic process",
    ],
    "protease": [
        "proteolysis",
        "viral protein processing",
        "protein processing",
    ],
    "nuclease": [
        "dna repair",
        "dna recombination",
        "dna integration",
    ],
    "packaging": [
        "viral dna genome packaging",
        "dna packaging",
        "genome packaging",
        "chromosome organization",
    ],
    "regulatory": [
        "regulation of dna-templated transcription",
        "positive regulation of dna-templated transcription",
        "negative regulation of transcription",
        "viral transcription",
        "protein phosphorylation",
        "regulation of dna replication",
    ],
    "host_interaction": [
        "virus-mediated perturbation of host defense response",
        "modulation by virus of host process",
        "immune response",
        "immune evasion",
    ],
    "entry": [
        "fusion of virus membrane with host plasma membrane",
        "virion attachment to host cell",
        "viral entry",
        "membrane fusion",
    ],
    "lysis": [
        "lysis",
        "cell lysis",
        "host cell lysis",
    ],
}

# =============================================================================
# GO MOLECULAR FUNCTION TO CATEGORY MAPPING
# =============================================================================

GO_MF_TO_CATEGORY = {
    "structural": [
        "structural molecule activity",
        "structural constituent of virion",
    ],
    "replication": [
        "dna-directed 5'-3' rna polymerase activity",
        "rna-dependent rna polymerase activity",
        "dna helicase activity",
        "rna helicase activity",
        "dna primase activity",
        "dna binding",
        "nucleotide binding",
        "atp binding",
        "atp hydrolysis activity",
    ],
    "protease": [
        "serine-type endopeptidase activity",
        "cysteine-type peptidase activity",
        "metalloendopeptidase activity",
        "aspartic-type endopeptidase activity",
        "peptidase activity",
    ],
    "nuclease": [
        "nuclease activity",
        "endonuclease activity",
        "exonuclease activity",
        "dna ligase activity",
        "rna ligase activity",
        "endodeoxyribonuclease activity",
    ],
    "regulatory": [
        "protein kinase activity",
        "methyltransferase activity",
        "transcription factor activity",
    ],
}

# =============================================================================
# SUPERFAMILY TO CATEGORY MAPPING
# =============================================================================

SUPERFAMILY_TO_CATEGORY = {
    "structural": [
        "viral glycoprotein",
        "positive stranded ssrna viruses",
        "vsv matrix protein",
        "ev matrix protein",
        "group i dsdna viruses",
        "viral glycoprotein ectodomain",
        "ndv fusion glycoprotein",
        "wssv envelope",
        "p40 nucleoprotein",
        "immunoglobulin",  # Many viral receptors/attachment proteins
    ],
    "replication": [
        "dna/rna polymerases",
        "p-loop containing nucleoside triphosphate hydrolases",
        "ribonuclease h-like",
        "beta and beta-prime subunits of dna dependent rna-polymerase",
        "dna ligase/mrna capping enzyme",
        "dna clamp",
    ],
    "protease": [
        "cysteine proteinases",
        "metalloproteases",
        "zincins",
    ],
    "nuclease": [
        "dnase i-like",
        "dutpase-like",
        "pin domain-like",
    ],
    "regulatory": [
        "s-adenosyl-l-methionine-dependent methyltransferases",
        "protein kinase-like",
        "ankyrin repeat",
        "ring/u-box",
        "tata-box binding protein-like",
        "dna-binding domain of mlu1-box",
        "fkbp12-rapamycin-binding domain",
    ],
    "host_interaction": [
        "bcl-2 inhibitors of programmed cell death",
    ],
}

# =============================================================================
# ENHANCED ANNOTATION DATA CLASS
# =============================================================================


@dataclass
class AnnotationEvidence:
    """Evidence for functional classification from various sources."""

    pfam: Optional[str] = None
    pfam_confidence: float = 0.0
    go_bp: Optional[str] = None
    go_bp_confidence: float = 0.0
    go_mf: Optional[str] = None
    go_mf_confidence: float = 0.0
    superfamily: Optional[str] = None
    superfamily_confidence: float = 0.0

    def has_evidence(self) -> bool:
        """Check if any annotation evidence exists."""
        return any([self.pfam, self.go_bp, self.go_mf, self.superfamily])


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================


def _match_terms(text: str, term_list: list[str]) -> bool:
    """Check if any term from the list is found in the text."""
    text_lower = text.lower()
    for term in term_list:
        if term.lower() in text_lower:
            return True
    return False


def classify_by_pfam(pfam_annotation: str) -> Optional[str]:
    """Classify protein by Pfam domain annotation.

    Args:
        pfam_annotation: Pfam domain name/description

    Returns:
        Functional category or None if no match
    """
    if not pfam_annotation:
        return None

    for category, terms in PFAM_TO_CATEGORY.items():
        if _match_terms(pfam_annotation, terms):
            return category

    return None


def classify_by_go_bp(go_bp: str) -> Optional[str]:
    """Classify protein by GO biological process.

    Args:
        go_bp: GO biological process term

    Returns:
        Functional category or None if no match
    """
    if not go_bp:
        return None

    for category, terms in GO_BP_TO_CATEGORY.items():
        if _match_terms(go_bp, terms):
            return category

    return None


def classify_by_go_mf(go_mf: str) -> Optional[str]:
    """Classify protein by GO molecular function.

    Args:
        go_mf: GO molecular function term

    Returns:
        Functional category or None if no match
    """
    if not go_mf:
        return None

    for category, terms in GO_MF_TO_CATEGORY.items():
        if _match_terms(go_mf, terms):
            return category

    return None


def classify_by_superfamily(superfamily: str) -> Optional[str]:
    """Classify protein by SUPERFAMILY annotation.

    Args:
        superfamily: SUPERFAMILY structural classification

    Returns:
        Functional category or None if no match
    """
    if not superfamily:
        return None

    for category, terms in SUPERFAMILY_TO_CATEGORY.items():
        if _match_terms(superfamily, terms):
            return category

    return None


def classify_by_keywords(description: str, gene: str | None = None) -> str:
    """Classify protein by keyword matching (original approach).

    Args:
        description: Protein description/annotation text
        gene: Optional gene name

    Returns:
        Functional category string
    """
    text = (description + " " + (gene or "")).lower()

    for category, keywords in FUNCTIONAL_CATEGORIES.items():
        for keyword in keywords:
            if keyword in text:
                return category

    for term in UNKNOWN_TERMS:
        if term in text:
            return "unknown"

    return "unknown"


def classify_protein(
    description: str,
    gene: str | None = None,
    evidence: AnnotationEvidence | None = None,
) -> str:
    """Classify protein into functional category using all available evidence.

    Uses a hierarchical approach:
    1. Domain-based classification (Pfam, SUPERFAMILY) - most reliable
    2. GO term-based classification (biological process, molecular function)
    3. Keyword-based fallback (original approach)

    Args:
        description: Protein description/annotation text
        gene: Optional gene name
        evidence: Optional AnnotationEvidence with Pfam/GO/SUPERFAMILY data

    Returns:
        Functional category string (e.g., "structural", "replication", "unknown")
    """
    # If we have rich annotation evidence, use hierarchical classification
    if evidence and evidence.has_evidence():
        # Priority 1: Pfam domain classification (most specific)
        if evidence.pfam:
            category = classify_by_pfam(evidence.pfam)
            if category:
                return category

        # Priority 2: SUPERFAMILY structural classification
        if evidence.superfamily:
            category = classify_by_superfamily(evidence.superfamily)
            if category:
                return category

        # Priority 3: GO biological process
        if evidence.go_bp:
            category = classify_by_go_bp(evidence.go_bp)
            if category:
                return category

        # Priority 4: GO molecular function
        if evidence.go_mf:
            category = classify_by_go_mf(evidence.go_mf)
            if category:
                return category

    # Fallback: keyword-based classification
    return classify_by_keywords(description, gene)


def get_classification_source(
    description: str,
    gene: str | None = None,
    evidence: AnnotationEvidence | None = None,
) -> tuple[str, str]:
    """Get classification and its source for detailed reporting.

    Args:
        description: Protein description/annotation text
        gene: Optional gene name
        evidence: Optional AnnotationEvidence with Pfam/GO/SUPERFAMILY data

    Returns:
        Tuple of (category, source) where source indicates what matched
    """
    if evidence and evidence.has_evidence():
        if evidence.pfam:
            category = classify_by_pfam(evidence.pfam)
            if category:
                return category, f"pfam:{evidence.pfam}"

        if evidence.superfamily:
            category = classify_by_superfamily(evidence.superfamily)
            if category:
                return category, f"superfamily:{evidence.superfamily}"

        if evidence.go_bp:
            category = classify_by_go_bp(evidence.go_bp)
            if category:
                return category, f"go_bp:{evidence.go_bp}"

        if evidence.go_mf:
            category = classify_by_go_mf(evidence.go_mf)
            if category:
                return category, f"go_mf:{evidence.go_mf}"

    category = classify_by_keywords(description, gene)
    return category, "keywords"


# =============================================================================
# CATEGORY STATISTICS
# =============================================================================

# All valid functional categories
ALL_CATEGORIES = [
    "structural",
    "replication",
    "protease",
    "nuclease",
    "packaging",
    "regulatory",
    "movement",
    "lysis",
    "host_interaction",
    "entry",
    "unknown",
]

# Category descriptions for reporting
CATEGORY_DESCRIPTIONS = {
    "structural": "Virion structural proteins (capsid, envelope, matrix)",
    "replication": "Genome replication machinery (polymerase, helicase, primase)",
    "protease": "Proteolytic enzymes (protease, peptidase)",
    "nuclease": "Nucleic acid processing (nuclease, integrase, ligase)",
    "packaging": "Genome packaging (terminase, scaffold)",
    "regulatory": "Gene regulation (transcription factors, kinases)",
    "movement": "Cell-to-cell movement (plant viruses)",
    "lysis": "Host cell lysis (bacteriophages)",
    "host_interaction": "Host interaction and immune evasion",
    "entry": "Host cell entry and fusion",
    "unknown": "Unknown or uncharacterized function",
}
