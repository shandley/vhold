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
    "--all-models",
    is_flag=True,
    default=False,
    help="Download all optional models (embeddings + classifier)",
)
def install(database, bfvd, viro3d, force, embeddings, classifier, all_models):
    """Download and install reference databases.

    Downloads BFVD and Viro3D Foldseek databases and metadata files
    required for annotation.
    """
    if all_models:
        embeddings = True
        classifier = True

    from vhold.databases.install import install_databases
    install_databases(
        database_dir=database,
        install_bfvd=bfvd,
        install_viro3d=viro3d,
        install_embeddings=embeddings,
        install_classifier=classifier,
        force=force,
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
    default=True,
    help="Use MLP classifier for unknown proteins (requires trained model)",
)
@click.option(
    "--classifier-confidence",
    default=0.5,
    type=float,
    help="Minimum confidence for MLP classifier predictions (default: 0.5)",
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
def run(
    input_file, output, database, databases, threads, batch_size, device,
    evalue, sensitivity, confidence_threshold, model_dir, prefix,
    fast, llm_classify, llm_model, triage, triage_threshold,
    classify, classifier_confidence, backend, lora,
    input_format, genome_fasta,
):
    """Run the full vhold annotation pipeline.

    This command runs the complete pipeline:
    1. Predict 3Di sequences using ProstT5
    2. Search against BFVD/Viro3D databases using Foldseek
    3. Transfer annotations and generate output files

    Accepts FASTA (.fasta/.fa/.faa), GenBank (.gb/.gbk), or GFF3 (.gff/.gff3).
    GenBank/GFF input preserves genomic coordinates for neighborhood voting.

    Examples:
        vhold run -i proteins.fasta -o results/ -t 4

        vhold run -i genome.gb -o results/ -t 4

        vhold run -i genes.gff3 --genome-fasta genome.fna -o results/
    """
    import os

    # Auto-detect LLM availability when user didn't specify --llm/--no-llm
    if llm_classify is None:
        llm_classify = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if llm_classify:
            click.echo(
                "LLM classification auto-enabled (ANTHROPIC_API_KEY detected). "
                "Use --no-llm to disable."
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
        backend=backend,
        use_lora=lora,
        input_format=input_format,
        genome_fasta=genome_fasta,
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
