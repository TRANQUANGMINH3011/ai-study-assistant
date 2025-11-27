import argparse, sys

from .processor import podcast_processor

def main():
    parser = argparse.ArgumentParser(description="Generate a podcast from a PDF document")
    
    # Required arguments
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file")
    
    # Optional arguments
    parser.add_argument("--config", type=str, help="Path to a custom config file", default="./Local-NotebookLM/example_config.json")
    parser.add_argument("--format", type=str, choices=["podcast", "interview", "panel-discussion", "debate", "summary", "narration", "storytelling", "explainer", "lecture", "tutorial", "q-and-a", "news-report", "executive-brief", "meeting-minutes", "analysis"], default="summary", help="Output format type")
    parser.add_argument("--length", type=str, choices=["short", "medium", "long", "very-long"], default="medium", help="Content length")
    parser.add_argument("--style", type=str, choices=["normal", "friendly", "professional", "academic", "casual", "technical", "gen-z", "funny"], default="normal", help="Content style")
    parser.add_argument("--preference", type=str, help="Additional preferences or instructions")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to store output files")
    parser.add_argument("--skip-to", type=int, choices=[1, 2, 3, 4], help="Skip to a specific step (1-4)")
    parser.add_argument("--language", type=str, help="Additional preferences or instructions")
    
    args = parser.parse_args()
    
    # Call the main process function with parsed arguments
    success, result = podcast_processor(
        pdf_path=args.pdf,
        config_path=args.config,
        format_type=args.format,
        length=args.length,
        style=args.style,
        preference=args.preference,
        output_dir=args.output_dir,
        skip_to=args.skip_to,
        language=args.language
    )

    if success:
        print(f"✅ Podcast generated at: {result}")
    else:
        print(f"❌ Failed: {result}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

if __name__ == "__main__":
    exit(main())