"""Functional category classification for viral proteins."""

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
    ],
    "movement": [
        "movement",
        "cell-to-cell",
        "transport protein",
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


def classify_protein(description: str, gene: str | None = None) -> str:
    """Classify protein into functional category based on description and gene name.

    Args:
        description: Protein description/annotation text
        gene: Optional gene name

    Returns:
        Functional category string (e.g., "structural", "replication", "unknown")
    """
    # Combine description and gene for matching
    text = (description + " " + (gene or "")).lower()

    # Check each category in priority order
    for category, keywords in FUNCTIONAL_CATEGORIES.items():
        for keyword in keywords:
            if keyword in text:
                return category

    # Check for unknown indicators
    for term in UNKNOWN_TERMS:
        if term in text:
            return "unknown"

    return "unknown"
