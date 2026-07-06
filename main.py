# CLI entrypoint for RAGS
import argparse
import json
from pathlib import Path

from src.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="RAGS - Retrieval Augmentation System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF corpus")
    ingest_parser.add_argument("--pdf", default="data/pdfs", help="PDF directory")
    ingest_parser.add_argument("--rebuild", action="store_true", help="Rebuild index")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--provider", choices=["nim", "openrouter", "ollama"], help="LLM provider")
    query_parser.add_argument("--model", help="Specific model name")

    # Ablation flags (for future phases, but define now)
    query_parser.add_argument("--no-hyde", action="store_true", help="Disable HyDE")
    query_parser.add_argument("--no-rerank", action="store_true", help="Disable reranker")
    query_parser.add_argument("--no-crag", action="store_true", help="Disable CRAG")
    query_parser.add_argument("--no-rewrite", action="store_true", help="Disable query rewriting")

    # Stats command
    subparsers.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize pipeline with flags
    pipeline = RAGPipeline(
        use_hyde=not args.no_hyde if hasattr(args, 'no_hyde') else False,
        use_reranker=not args.no_rerank if hasattr(args, 'no_rerank') else False,
        use_crag=not args.no_crag if hasattr(args, 'no_crag') else False,
        use_query_rewrite=not args.no_rewrite if hasattr(args, 'no_rewrite') else False,
    )

    if args.command == "ingest":
        print(f"Ingesting PDFs from {args.pdf}...")
        pipeline.ingest(pdf_dir=args.pdf, rebuild_index=args.rebuild)
        stats = pipeline.stats()
        print(json.dumps(stats, indent=2))

    elif args.command == "query":
        if args.provider:
            pipeline.switch_llm(args.provider, args.model)

        print(f"Querying: {args.question}")
        response = pipeline.query(args.question)
        print("\n" + response.format_with_sources())

    elif args.command == "stats":
        stats = pipeline.stats()
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()