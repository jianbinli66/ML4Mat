import os
import json
import argparse
from pathlib import Path

import pandas as pd

from processors.article_processor import ArticleExtractor


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Comprehensive PDF data extraction using LLM (text) and VLM (images)')
    parser.add_argument('--dataset', type=str, default='DAC_sample_papers', required=True,
                        help='Dataset name (optionally with .class suffix, e.g., DAC_sample_papers-Physical Impregnation Method)')
    parser.add_argument('--llm_model', type=str, required=True, help='LLM model for text extraction')
    parser.add_argument('--vlm_model', type=str, default='gpt-4-vision-preview', help='VLM model for image analysis')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to use for LLM extraction')
    parser.add_argument('--num_papers', type=int, default=None, help='Number of papers to process')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    parser.add_argument('--target_params', nargs='+', default=['adsorption'],
                        help='Target parameters for VLM extraction')
    parser.add_argument('--exp_class', type=str, default=None,
                        help='Filter data by experiment class (can be multiple space-separated values)')
    parser.add_argument('--exp_class_file_path', type=str, default='data/DAC_extract/classification.csv',
                        help='data classes')
    parser.add_argument('--base_dir', type=str, default='data/pdfs2textimgs',
                        help='Base directory containing PDF directories')
    parser.add_argument('--use_vlm', action='store_true', default=True,
                        help='Enable VLM model for image analysis (default: True)')
    parser.add_argument('--no_vlm', action='store_false', dest='use_vlm',
                        help='Disable VLM model for image analysis')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Parse dataset argument for optional class suffix
    # ------------------------------------------------------------------
    base_dataset = args.dataset
    class_suffix = None
    if '.' in args.dataset:
        base_dataset, class_suffix = args.dataset.split('-', 1)
        # If the class suffix is empty, ignore it
        if not class_suffix:
            class_suffix = None
        else:
            # Override any --exp_class argument with the suffix from dataset
            if args.exp_class is not None:
                print(f"Warning: Both --dataset with class suffix and --exp_class provided. "
                      f"Using class from dataset: '{class_suffix}'")
            args.exp_class = [class_suffix]  # set as a list for filtering

    # Update dataset name for config lookup
    args.dataset = base_dataset

    # ------------------------------------------------------------------
    # 2. Load configuration and dynamically discover paper directories
    # ------------------------------------------------------------------
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory {args.base_dir} does not exist")
        exit(1)

    config = json.load(open(os.path.join(args.config)))
    config["dataset"] = args.dataset
    
    # Dynamically discover paper directories in the dataset directory
    dataset_dir = os.path.join(args.base_dir, args.dataset)
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} does not exist")
        exit(1)
    
    # Get all subdirectories (each representing a paper)
    paper_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    paper_lis = sorted(paper_dirs)
    
    if not paper_lis:
        print(f"No paper directories found in {dataset_dir}")
        exit(1)
    
    if args.num_papers is not None:
        paper_lis = paper_lis[:args.num_papers]
    
    print("Config:", config)
    print(f"Found {len(paper_lis)} paper directories")

    # ------------------------------------------------------------------
    # 3. Apply class filtering if requested (from --exp_class or from dataset suffix)
    # ------------------------------------------------------------------
    if args.exp_class is not None and os.path.exists(args.exp_class_file_path):
        class_df = pd.read_csv(args.exp_class_file_path,dtype=str)
        # Ensure args.exp_class is a list (it might be a string if set from suffix, but we set it as list above)
        if isinstance(args.exp_class, str):
            args.exp_class = [args.exp_class]
        filtered_papers = class_df[class_df['Sub-Class 3'].isin(args.exp_class)]['paper_id'].unique().tolist()
        if not filtered_papers:
            print(f"Warning: No papers found for class(es) {args.exp_class}. Exiting.")
            exit(0)
        paper_lis = filtered_papers
        if args.num_papers is not None:
            paper_lis = paper_lis[:args.num_papers]

        # Create a clean suffix for the prompt (replace spaces with underscores)
        class_suffix_str = '_'.join(str(c).replace(' ', '_') for c in args.exp_class)
        args.prompt = f"{args.prompt}_{class_suffix_str}"

    # ------------------------------------------------------------------
    # 4. Initialize extractor and run extraction
    # ------------------------------------------------------------------
    extractor = ArticleExtractor(config=config)

    # Create output directory name based on whether VLM is used
    vlm_suffix = "_with_vlm" if args.use_vlm else "_no_vlm"
    OUTPUT_DIR = f"./json_results/{extractor.dataset_name}/{args.prompt}_{args.llm_model}{vlm_suffix}"
    print("Output directory:", OUTPUT_DIR)

    base_dir = os.path.join(args.base_dir, extractor.dataset_name)

    extractor.process_all_papers(
        base_dir=base_dir,
        llm_model_choice=args.llm_model,
        prompt_choice=args.prompt,
        vlm_model_choice=args.vlm_model,
        target_parameters=args.target_params,
        output_dir=OUTPUT_DIR,
        paper_lis=paper_lis,
        use_vlm=args.use_vlm
    )

if __name__ == "__main__":
    main()