"""Command-line interface for vhold."""

import click

from vhold import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="vhold")
def main():
    """vhold - Viral protein annotation using structural homology.

    Annotate viral proteins by comparing predicted 3D structures against
    BFVD and Viro3D reference databases using ProstT5 and Foldseek.
    """
    pass


@main.command()
@click.option(
    "-d", "--database",
    type=click.Path(),
    default=None,
    help="Database directory (default: ~/.vhold/databases)",
)
@click.option(
    "--bfvd/--no-bfvd",
    default=True,
    help="Download BFVD database (default: yes)",
)
@click.option(
    "--viro3d/--no-viro3d",
    default=True,
    help="Download Viro3D database (default: yes)",
)
@click.option(
    "-f", "--force",
    is_flag=True,
    default=False,
    help="Force re-download even if files exist",
)
@click.option(
    "--embeddings/--no-embeddings",
    default=False,
    help="Download embedding database for triage mode",
)
@click.option(
    "--classifier/--no-classifier",
    default=False,
    help="Download MLP classifier model",
)
@click.option(
    "--disorder-classifier/--no-disorder-classifier",
    "disorder_classifier",
    default=False,
    help="Download disorder-aware classifier model",
)
@click.option(
    "--starling-db/--no-starling-db",
    "starling_db",
    default=False,
    help="Download STARLING search database (~18.6GB)",
)
@click.option(
    "--swissprot/--no-swissprot",
    "swissprot",
    default=False,
    help="Download Swiss-Prot viral reference DB for GO transfer",
)
@click.option(
    "--swissprot-all",
    is_flag=True,
    default=False,
    help="Download full Swiss-Prot reference DB (570K proteins, ~1.1GB) for cross-domain discovery",
)
@click.option(
    "--all-models",
    is_flag=True,
    default=False,
    help="Download all optional models (embeddings + classifier + disorder classifier). Does not include STARLING DB; use --starling-db separately.",
)
def install(database, bfvd, viro3d, force, embeddings, classifier, disorder_classifier, starling_db,
            swissprot, swissprot_all, all_models):
    """Download and install reference databases.

    Downloads BFVD and Viro3D Foldseek databases and metadata files
    required for annotation.
    """
    if all_models:
        embeddings = True
        classifier = True
        disorder_classifier = True

    from vhold.databases.install import install_databases
    install_databases(
        database_dir=database,
        install_bfvd=bfvd,
        install_viro3d=viro3d,
        install_embeddings=embeddings,
        install_classifier=classifier,
        install_disorder_classifier=disorder_classifier,
        install_starling_db=starling_db,
        force=force,
    )

    # Swiss-Prot reference DB (separate from main install flow)
    if swissprot or swissprot_all:
        from pathlib import Path as _P
        from vhold.databases.swissprot import install_swissprot_db
        from vhold.utils.constants import get_db_dir

        db_dir = _P(database) if database else get_db_dir()
        scope = "all" if swissprot_all else "viral"
        install_swissprot_db(db_dir, scope=scope, force=force)


@main.command()
@click.option(
    "-i", "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input FASTA file with protein sequences",
)
@click.option(
    "-o", "--output",
    required=True,
    type=click.Path(),
    help="Output directory for 3Di predictions",
)
@click.option(
    "-t", "--threads",
    default=1,
    type=int,
    help="Number of threads (default: 1)",
)
@click.option(
    "--batch-size",
    default=1,
    type=int,
    help="Batch size for ProstT5 inference (default: 1)",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
    help="Device for inference (default: auto)",
)
@click.option(
    "--confidence-threshold",
    default=0.7,
    type=float,
    help="Minimum confidence for 3Di residues (default: 0.7)",
)
@click.option(
    "--model-dir",
    type=click.Path(),
    default=None,
    help="Directory for ProstT5 model cache",
)
@click.option(
    "--fast",
    is_flag=True,
    default=False,
    help="Use greedy decoding (~3x faster, may reduce sensitivity for remote homologs)",
)
@click.option(
    "--backend",
    type=click.Choice(["torch", "onnx"]),
    default="torch",
    help="Inference backend (default: torch). ONNX requires: pip install vhold[onnx] && vhold export-onnx",
)
def predict(input_file, output, threads, batch_size, device, confidence_threshold, model_dir, fast,
            backend):
    """Predict 3Di structural sequences using ProstT5.

    Takes protein sequences and predicts 3Di structural alphabet
    representations using the ProstT5 model.
    """
    from vhold.subcommands.predict import run_predict
    run_predict(
        input_file=input_file,
        output_dir=output,
        threads=threads,
        batch_size=batch_size,
        device=device,
        confidence_threshold=confidence_threshold,
        model_dir=model_dir,
        fast=fast,
        backend=backend,
    )


@main.command()
@click.option(
    "-p", "--predictions-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing 3Di predictions",
)
@click.option(
    "-o", "--output",
    required=True,
    type=click.Path(),
    help="Output directory for Foldseek results",
)
@click.option(
    "-d", "--database",
    type=click.Path(),
    default=None,
    help="Database directory (default: ~/.vhold/databases)",
)
@click.option(
    "--databases",
    type=click.Choice(["all", "bfvd", "viro3d"]),
    default="all",
    help="Which databases to search (default: all)",
)
@click.option(
    "-t", "--threads",
    default=4,
    type=int,
    help="Number of threads for Foldseek (default: 4)",
)
@click.option(
    "-e", "--evalue",
    default=1e-3,
    type=float,
    help="E-value threshold (default: 1e-3)",
)
@click.option(
    "--sensitivity",
    default=9.5,
    type=float,
    help="Foldseek sensitivity (default: 9.5)",
)
@click.option(
    "--max-seqs",
    default=1000,
    type=int,
    help="Maximum sequences to report per query (default: 1000)",
)
def compare(predictions_dir, output, database, databases, threads, evalue, sensitivity, max_seqs):
    """Search 3Di predictions against reference databases.

    Uses Foldseek to compare predicted 3Di sequences against BFVD
    and/or Viro3D databases.
    """
    from vhold.subcommands.compare import run_compare
    run_compare(
        predictions_dir=predictions_dir,
        output_dir=output,
        database_dir=database,
        databases=databases,
        threads=threads,
        evalue=evalue,
        sensitivity=sensitivity,
        max_seqs=max_seqs,
    )


@main.command()
@click.option(
    "-i", "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input FASTA file with protein sequences",
)
@click.option(
    "-o", "--output",
    required=True,
    type=click.Path(),
    help="Output directory for all results",
)
@click.option(
    "-d", "--database",
    type=click.Path(),
    default=None,
    help="Database directory (default: ~/.vhold/databases)",
)
@click.option(
    "--databases",
    type=click.Choice(["all", "bfvd", "viro3d"]),
    default="all",
    help="Which databases to search (default: all)",
)
@click.option(
    "-t", "--threads",
    default=4,
    type=int,
    help="Number of threads (default: 4)",
)
@click.option(
    "--batch-size",
    default=1,
    type=int,
    help="Batch size for ProstT5 inference (default: 1)",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
    help="Device for ProstT5 inference (default: auto)",
)
@click.option(
    "-e", "--evalue",
    default=1e-3,
    type=float,
    help="E-value threshold for Foldseek (default: 1e-3)",
)
@click.option(
    "--sensitivity",
    default=9.5,
    type=float,
    help="Foldseek sensitivity (default: 9.5)",
)
@click.option(
    "--confidence-threshold",
    default=0.7,
    type=float,
    help="Minimum confidence for 3Di residues (default: 0.7)",
)
@click.option(
    "--model-dir",
    type=click.Path(),
    default=None,
    help="Directory for ProstT5 model cache",
)
@click.option(
    "--prefix",
    default="vhold",
    help="Prefix for output files (default: vhold)",
)
@click.option(
    "--fast",
    is_flag=True,
    default=False,
    help="Use greedy decoding (~3x faster, may reduce sensitivity for remote homologs)",
)
@click.option(
    "--llm/--no-llm",
    "llm_classify",
    default=None,
    help="Use LLM to classify unknown proteins (auto-enabled when ANTHROPIC_API_KEY is set)",
)
@click.option(
    "--llm-model",
    default="claude-haiku-4-5-20251001",
    help="Model for LLM classification (default: claude-haiku-4-5-20251001)",
)
@click.option(
    "--triage/--no-triage",
    default=False,
    help="Use embedding triage to skip decoder for known proteins (requires embedding DB)",
)
@click.option(
    "--triage-threshold",
    default=0.90,
    type=float,
    help="Cosine similarity threshold for embedding triage (default: 0.90)",
)
@click.option(
    "--classify/--no-classify",
    default=False,
    help="Use MLP classifier for unknown proteins (default: disabled, use --classify to enable)",
)
@click.option(
    "--classifier-confidence",
    default=0.5,
    type=float,
    help="Minimum confidence for MLP classifier predictions (default: 0.5)",
)
@click.option(
    "--classifier-entropy",
    default=0.6,
    type=float,
    help="Max normalized entropy for MLP classifier (0=certain, 1=uniform, default: 0.6)",
)
@click.option(
    "--backend",
    type=click.Choice(["torch", "onnx"]),
    default="torch",
    help="Inference backend (default: torch). ONNX requires: pip install vhold[onnx] && vhold export-onnx",
)
@click.option(
    "--lora/--no-lora",
    default=True,
    help="Auto-load contrastive LoRA adapter if installed (default: yes)",
)
@click.option(
    "--disorder/--no-disorder",
    "disorder",
    default=None,
    help="Predict intrinsic disorder (auto-enabled when metapredict installed)",
)
@click.option(
    "--disorder-threshold",
    default=0.5,
    type=float,
    help="Per-residue disorder threshold (default: 0.5)",
)
@click.option(
    "--disorder-classifier-confidence",
    default=0.5,
    type=float,
    help="Min confidence for disorder classifier (default: 0.5)",
)
@click.option(
    "--starling-search/--no-starling-search",
    "use_starling_search",
    default=None,
    help="Search STARLING IDR database for disordered unknowns (auto-enabled when available)",
)
@click.option(
    "--starling-similarity-threshold",
    default=0.7,
    type=float,
    help="Min similarity for STARLING search hits (default: 0.7)",
)
@click.option(
    "--input-format",
    type=click.Choice(["auto", "fasta", "genbank", "gff"]),
    default="auto",
    help="Input file format (default: auto-detect by extension)",
)
@click.option(
    "--genome-fasta",
    type=click.Path(exists=True),
    default=None,
    help="Genome FASTA for GFF input (if not embedded in GFF)",
)
@click.option(
    "--mat-peptide/--no-mat-peptide",
    "include_mat_peptide",
    default=True,
    help="Extract mat_peptide features from GenBank (default: yes)",
)
@click.option(
    "--export-format",
    "export_formats",
    multiple=True,
    type=click.Choice(["anvio", "vcontact2", "vcontact3", "dramv", "gff3", "all"]),
    default=None,
    help="Auto-export to metagenomic format(s) after pipeline completes",
)
@click.option(
    "--go-transfer/--no-go-transfer",
    "go_transfer",
    default=None,
    help="GO term transfer from Swiss-Prot (auto-enabled when DB installed)",
)
@click.option(
    "--go-transfer-k",
    default=3,
    type=int,
    help="Number of nearest neighbors for GO transfer (default: 3)",
)
@click.option(
    "--go-transfer-threshold",
    default=0.5,
    type=float,
    help="Min cosine similarity for GO transfer (default: 0.5)",
)
@click.option(
    "--go-reliability-threshold",
    default=0.3,
    type=float,
    help="Min Reliability Index to keep transferred GO term (default: 0.3)",
)
@click.option(
    "--cross-domain/--no-cross-domain",
    default=True,
    help="Include non-viral Swiss-Prot matches for cross-domain discovery (default: yes)",
)
@click.option(
    "--cross-domain-threshold",
    default=0.95,
    type=float,
    help="Min cosine similarity for non-viral donor GO transfer (default: 0.95)",
)
@click.option(
    "--gene-caller",
    type=click.Choice(["auto", "pyrodigal", "none"]),
    default="none",
    help="Gene calling for nucleotide input: auto (Pyrodigal + miniprot), "
         "pyrodigal (Pyrodigal only), none (expect protein input, default)",
)
@click.option(
    "--translation-table",
    default=11,
    type=int,
    help="Genetic code for gene calling (11=bacterial/viral, 1=eukaryotic, default: 11)",
)
@click.option(
    "--closed/--no-closed",
    default=False,
    help="Only call complete genes with start and stop codons (default: no)",
)
@click.option(
    "--min-gene",
    default=90,
    type=int,
    help="Minimum gene length in nucleotides (default: 90)",
)
def run(
    input_file, output, database, databases, threads, batch_size, device,
    evalue, sensitivity, confidence_threshold, model_dir, prefix,
    fast, llm_classify, llm_model, triage, triage_threshold,
    classify, classifier_confidence, classifier_entropy, backend, lora,
    disorder, disorder_threshold, disorder_classifier_confidence,
    use_starling_search, starling_similarity_threshold,
    input_format, genome_fasta, include_mat_peptide, export_formats,
    go_transfer, go_transfer_k, go_transfer_threshold,
    go_reliability_threshold, cross_domain, cross_domain_threshold,
    gene_caller, translation_table, closed, min_gene,
):
    """Run the full vhold annotation pipeline.

    This command runs the complete pipeline:
    0. (Optional) Call genes from nucleotide sequences (--gene-caller)
    1. Predict 3Di sequences using ProstT5
    2. Search against BFVD/Viro3D databases using Foldseek
    3. Transfer annotations and generate output files

    Accepts FASTA (.fasta/.fa/.faa), GenBank (.gb/.gbk), GFF3 (.gff/.gff3),
    or nucleotide FASTA (.fna/.ffn) with --gene-caller.

    Examples:
        vhold run -i proteins.fasta -o results/ -t 4

        vhold run -i genome.gb -o results/ -t 4

        vhold run -i contigs.fna -o results/ --gene-caller auto -t 4

        vhold run -i genes.gff3 --genome-fasta genome.fna -o results/
    """
    import os
    from pathlib import Path as _Path

    # Auto-detect nucleotide input when gene_caller is "none"
    if gene_caller == "none":
        input_suffix = _Path(input_file).suffix.lower()
        if input_suffix in {".fna", ".ffn"}:
            click.echo(
                f"Nucleotide FASTA detected ({input_suffix}). Use --gene-caller auto "
                "to enable gene calling, or provide protein sequences."
            )
            raise SystemExit(1)
        if input_suffix in {".fasta", ".fa"} and input_format in (None, "auto"):
            from vhold.io.nucleotide import is_nucleotide_fasta
            if is_nucleotide_fasta(input_file):
                click.echo(
                    "Input appears to contain nucleotide sequences. "
                    "Use --gene-caller auto to enable gene calling, "
                    "or provide protein sequences."
                )
                raise SystemExit(1)

    # Auto-detect LLM availability when user didn't specify --llm/--no-llm
    if llm_classify is None:
        llm_classify = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if llm_classify:
            click.echo(
                "LLM classification auto-enabled (ANTHROPIC_API_KEY detected). "
                "Use --no-llm to disable."
            )

    # Auto-detect disorder prediction availability
    if disorder is None:
        from vhold.features.disorder import check_metapredict_available
        disorder = check_metapredict_available()
        if disorder:
            click.echo(
                "Disorder prediction auto-enabled (metapredict detected). "
                "Use --no-disorder to disable."
            )

    # Auto-detect STARLING search availability (requires disorder to be enabled)
    if use_starling_search is None and disorder:
        from vhold.features.starling_search import check_starling_search_available
        use_starling_search = check_starling_search_available()
        if use_starling_search:
            click.echo(
                "STARLING similarity search auto-enabled (SearchEngine detected). "
                "Use --no-starling-search to disable."
            )

    # Auto-detect GO transfer availability (Swiss-Prot DB installed)
    if go_transfer is None:
        from vhold.databases.swissprot import get_available_scope
        go_transfer = get_available_scope(
            _Path(database) if database else None
        ) is not None
        if go_transfer:
            click.echo(
                "GO term transfer auto-enabled (Swiss-Prot DB detected). "
                "Use --no-go-transfer to disable."
            )

    from vhold.subcommands.run import run_pipeline
    run_pipeline(
        input_file=input_file,
        output_dir=output,
        database_dir=database,
        databases=databases,
        threads=threads,
        batch_size=batch_size,
        device=device,
        evalue=evalue,
        sensitivity=sensitivity,
        confidence_threshold=confidence_threshold,
        model_dir=model_dir,
        prefix=prefix,
        fast=fast,
        llm_classify=llm_classify,
        llm_model=llm_model,
        triage=triage,
        triage_threshold=triage_threshold,
        classify=classify,
        classifier_confidence=classifier_confidence,
        classifier_entropy=classifier_entropy,
        backend=backend,
        use_lora=lora,
        disorder=disorder,
        disorder_threshold=disorder_threshold,
        disorder_classifier_confidence=disorder_classifier_confidence,
        use_starling_search=bool(use_starling_search),
        starling_similarity_threshold=starling_similarity_threshold,
        input_format=input_format,
        genome_fasta=genome_fasta,
        include_mat_peptide=include_mat_peptide,
        export_formats=list(export_formats) if export_formats else None,
        gene_caller=gene_caller,
        translation_table=translation_table,
        closed=closed,
        min_gene=min_gene,
        go_transfer=bool(go_transfer),
        go_transfer_k=go_transfer_k,
        go_transfer_threshold=go_transfer_threshold,
        go_reliability_threshold=go_reliability_threshold,
        cross_domain=cross_domain,
        cross_domain_threshold=cross_domain_threshold,
    )


@main.command()
@click.option(
    "-i", "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input FASTA file with protein sequences",
)
@click.option(
    "-o", "--output",
    required=True,
    type=click.Path(),
    help="Output directory for alignment results",
)
@click.option(
    "-t", "--threads",
    default=4,
    type=int,
    help="Number of threads for FoldMason (default: 4)",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
    help="Device for ProstT5 inference (default: auto)",
)
@click.option(
    "--fast",
    is_flag=True,
    default=False,
    help="Use greedy decoding for ProstT5 (~3x faster)",
)
@click.option(
    "--confidence-threshold",
    default=0.7,
    type=float,
    help="Minimum confidence for 3Di residues (default: 0.7)",
)
@click.option(
    "--model-dir",
    type=click.Path(),
    default=None,
    help="Directory for ProstT5 model cache",
)
@click.option(
    "--refine-iters",
    default=0,
    type=int,
    help="Number of FoldMason refinement iterations (default: 0, skip)",
)
@click.option(
    "--predictions-dir",
    type=click.Path(exists=True),
    default=None,
    help="Pre-computed 3Di predictions directory (skip ProstT5 step)",
)
@click.option(
    "--backend",
    type=click.Choice(["torch", "onnx"]),
    default="torch",
    help="Inference backend (default: torch). ONNX requires: pip install vhold[onnx] && vhold export-onnx",
)
def align(input_file, output, threads, device, fast, confidence_threshold,
          model_dir, refine_iters, predictions_dir, backend):
    """Multiple structural alignment of viral protein families.

    Predicts 3Di structural sequences using ProstT5, then runs FoldMason
    to produce a multiple structural alignment. Works without 3D coordinates
    by using FoldMason's fastMode (3Di+AA string alignment).

    Requires FoldMason: conda install -c bioconda foldmason

    Examples:

        vhold align -i proteins.fasta -o alignment/ --fast

        vhold align -i proteins.fasta --predictions-dir predictions/ -o alignment/
    """
    from vhold.subcommands.align import run_align
    run_align(
        input_file=input_file,
        output_dir=output,
        threads=threads,
        device=device,
        fast=fast,
        confidence_threshold=confidence_threshold,
        model_dir=model_dir,
        refine_iters=refine_iters,
        predictions_dir=predictions_dir,
        backend=backend,
    )


@main.command()
@click.option(
    "-i", "--input",
    "input_tsv",
    required=True,
    type=click.Path(exists=True),
    help="Input vHold results TSV file (*_results.tsv)",
)
@click.option(
    "-o", "--output",
    required=True,
    type=click.Path(),
    help="Output directory for exported files",
)
@click.option(
    "-f", "--format",
    "formats",
    multiple=True,
    type=click.Choice(["anvio", "vcontact2", "vcontact3", "dramv", "gff3", "all"]),
    default=["all"],
    help="Export format(s) (default: all). Can specify multiple: -f anvio -f dramv",
)
@click.option(
    "--prefix",
    default="vhold",
    help="Prefix for output files (default: vhold)",
)
def export(input_tsv, output, formats, prefix):
    """Export vHold results to metagenomic tool formats.

    Converts an existing vHold results TSV to formats compatible with
    anvi'o, vConTACT2/3, DRAM-v, and standard GFF3.

    Examples:

        vhold export -i results/vhold_results.tsv -o exports/

        vhold export -i results.tsv -o exports/ -f anvio -f dramv

        vhold export -i results.tsv -o exports/ -f vcontact3
    """
    from vhold.subcommands.export import run_export
    run_export(
        input_tsv=input_tsv,
        output_dir=output,
        formats=list(formats),
        prefix=prefix,
    )


@main.command("export-onnx")
@click.option(
    "--model-dir",
    type=click.Path(),
    default=None,
    help="ProstT5 model cache directory (default: ~/.vhold/models)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output for ONNX models (default: ~/.vhold/models/onnx_int8)",
)
@click.option(
    "--quantization",
    type=click.Choice(["avx512_vnni", "avx512", "avx2", "arm64"]),
    default=None,
    help="Quantization target (auto-detected if not specified)",
)
def export_onnx(model_dir, output_dir, quantization):
    """Export ProstT5 to ONNX INT8 for faster CPU inference.

    This is a one-time setup step. After export, use --backend onnx
    with run/predict/align commands for 3-6x CPU speedup.

    Requires: pip install vhold[onnx]

    Examples:

        vhold export-onnx

        vhold export-onnx --quantization arm64
    """
    from vhold.features.onnx_export import export_and_quantize
    export_and_quantize(
        model_dir=model_dir,
        output_dir=output_dir,
        quantization_target=quantization,
    )


if __name__ == "__main__":
    main()
