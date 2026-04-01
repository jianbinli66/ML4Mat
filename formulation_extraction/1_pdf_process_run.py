#!/usr/bin/env python3
# pdf_process_run.py
import json
import argparse
import os
from pathlib import Path

from processors.pdf_processor import process_files_in_batches


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Extract data from PDFs using Mineru API')
    parser.add_argument('--dataset', type=str, default=None, help='dataset')
    parser.add_argument('--input_dir', type=str, default='data/DAC', help='Input directory containing PDF files')
    parser.add_argument('--output_dir', type=str, default='data/extracted_pdf2json',
                        help='Output directory for extracted results')
    parser.add_argument('--num_papers', type=int, default=None, help='Number of papers to process (default: all)')
    parser.add_argument('--batch_size', type=int, default=3, help='Maximum number of files to process per batch')

    args = parser.parse_args()

    # Create directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        exit(1)

    # read config.json
    config = json.load(open(os.path.join("config.json")))
    dataset = config["dataset"] if args.dataset is None else args.dataset
    config["dataset"] = dataset
    
    # Dynamically discover PDF files in the input directory
    pdf_files = list(input_dir.glob("*.pdf"))
    paper_lis = sorted([f.stem for f in pdf_files])  # Get filenames without extension
    
    if not paper_lis:
        print(f"Error: No PDF files found in {input_dir}")
        exit(1)
    
    print(f"Found {len(paper_lis)} PDF files in {input_dir}")
    
    if args.num_papers is not None:
        paper_lis = paper_lis[:args.num_papers]
        print(f"Processing {len(paper_lis)} papers")

    # Process PDFs using Mineru API
    process_files_in_batches(
        input_dir=input_dir,
        output_dir=output_dir,
        papers=paper_lis
    )


if __name__ == "__main__":
    main()