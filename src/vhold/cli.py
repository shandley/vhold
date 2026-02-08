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
def install(database, bfvd, viro3d, force, embeddings):
    """Download and install reference databases.

    Downloads BFVD and Viro3D Foldseek databases and metadata files
    required for annotation.
    """
    from vhold.databases.install import install_databases
    install_databases(
        database_dir=database,
        install_bfvd=bfvd,
        install_viro3d=viro3d,
        install_embeddings=embeddings,
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
def predict(input_file, output, threads, batch_size, device, confidence_threshold, model_dir, fast):
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
    "--llm-classify",
    is_flag=True,
    default=False,
    help="Use LLM to classify unknown proteins (requires: pip install vhold[llm])",
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
    default=0.95,
    type=float,
    help="Cosine similarity threshold for embedding triage (default: 0.95)",
)
def run(
    input_file, output, database, databases, threads, batch_size, device,
    evalue, sensitivity, confidence_threshold, model_dir, prefix,
    fast, llm_classify, llm_model, triage, triage_threshold,
):
    """Run the full vhold annotation pipeline.

    This command runs the complete pipeline:
    1. Predict 3Di sequences using ProstT5
    2. Search against BFVD/Viro3D databases using Foldseek
    3. Transfer annotations and generate output files

    Example:
        vhold run -i proteins.fasta -o results/ -t 4
    """
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
    )


if __name__ == "__main__":
    main()
