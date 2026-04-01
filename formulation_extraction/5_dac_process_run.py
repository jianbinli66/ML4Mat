#!/usr/bin/env python3
# dac_process_run.py
"""
DAC Data Processing Script

This script processes extracted DAC (Direct Air Capture) data by applying various transformations,
cleaning operations, and calculations to standardize the data format and units.
"""

import argparse
import sys
import os
from utils.dac_extracted_data_process import process_dac_data


def main():
    """Main function with command line argument support for DAC data processing"""
    parser = argparse.ArgumentParser(
        description='Process DAC extracted data with standardized transformations and cleaning operations.'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/DAC_sample_papers_extract/dac_extracted_data.csv',
        help='Input CSV file path containing raw extracted DAC data (default: data/DAC_sample_papers_extract/dac_extracted_data.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/DAC_sample_papers_extract/dac_processed_data_by_LLM.csv',
        help='Output CSV file path for processed DAC data (default: data/DAC_sample_papers_extract/dac_processed_data_by_LLM.csv)'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Process the data
        processed_df = process_dac_data(input_file=args.input, output_file=args.output)
        print(f"Data processing completed successfully. Output saved to {args.output}")

        # Print summary statistics
        print(f"Processed {len(processed_df)} records with {len(processed_df.columns)} columns.")

    except Exception as e:
        print(f"Error during data processing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()