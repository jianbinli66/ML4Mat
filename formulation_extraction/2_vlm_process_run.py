#!/usr/bin/env python3
# vlm_process_run.py
import argparse
import json
import os

import pandas as pd

from processors.vlm_img_processor import VLMImageProcessor, ParallelVLMProcessor


def main():
    """Standalone VLM processing for images with parallel processing and class filtering"""
    parser = argparse.ArgumentParser(description='Standalone VLM image processing with parallel and batch processing')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (optionally with .class suffix, e.g., Geopolymer-Physical Impregnation Method)')
    parser.add_argument('--vlm_model', type=str, default='gpt-4-vision-preview', help='VLM model for image analysis')
    parser.add_argument('--num_papers', type=int, default=None, help='Number of papers to process')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    parser.add_argument('--target_params', nargs='+', default=['28d_compressive_strength'],
                        help='Target parameters for VLM extraction')
    parser.add_argument('--base_dir', type=str, default='data/pdfs2textimgs',
                        help='Base directory containing PDF directories')
    parser.add_argument('--output_dir', type=str, default='vlm_results',
                        help='Output directory for VLM results')
    parser.add_argument('--max_workers', type=int, default=5,
                        help='Maximum number of parallel workers for real-time (default: 5)')
    parser.add_argument('--serial', action='store_true', default=False,
                        help='Use serial processing instead of parallel (default: False)')
    parser.add_argument('--replace', action='store_true', default=False,
                        help='Replace existing analysis files (default: False)')
    parser.add_argument('--batch', action='store_true', default=False,
                        help='Use batch processing if available (default: False)')
    parser.add_argument('--force_realtime', action='store_true', default=False,
                        help='Force real-time processing instead of batch (default: False)')
    # New arguments for class filtering
    parser.add_argument('--class_file', type=str, default='data/DAC_extract/classification.csv',
                        help='CSV file with paper classification (must contain paper_id and Class columns)')
    parser.add_argument('--class', dest='class_filter', type=str, default=None,
                        help='Filter papers by class (alternative to dataset suffix)')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Parse dataset argument for optional class suffix
    # ------------------------------------------------------------------
    base_dataset = args.dataset
    class_suffix = None
    if '.' in args.dataset:
        base_dataset, class_suffix = args.dataset.split('-', 1)
        if not class_suffix:
            class_suffix = None
        else:
            print(f"Class suffix detected from dataset: '{class_suffix}'")
            # Override any explicit --class argument with the suffix
            if args.class_filter is not None:
                print(f"Warning: Both dataset class suffix and --class provided. Using class from dataset: '{class_suffix}'")
            args.class_filter = class_suffix

    # ------------------------------------------------------------------
    # 2. Load configuration and dynamically discover paper directories
    # ------------------------------------------------------------------
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory {args.base_dir} does not exist")
        exit(1)

    config = json.load(open(args.config))
    config["dataset"] = base_dataset

    # Dynamically discover paper directories in the dataset directory
    dataset_dir = os.path.join(args.base_dir, base_dataset)
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} does not exist")
        exit(1)
    
    # Get all subdirectories (each representing a paper)
    paper_dirs = [d.replace(".pdf-id",'') for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    paper_lis = sorted(paper_dirs)
    
    if not paper_lis:
        print(f"No paper directories found in {dataset_dir}")
        exit(1)

    print(f"Base dataset: {base_dataset}, total papers found: {len(paper_lis)}")

    # ------------------------------------------------------------------
    # 3. Apply class filtering if requested
    # ------------------------------------------------------------------
    if args.class_filter is not None:
        if not os.path.exists(args.class_file):
            print(f"Error: Classification file {args.class_file} not found. Cannot filter by class.")
            exit(1)

        class_df = pd.read_csv(args.class_file,dtype=str)
        # Ensure required columns exist
        if 'paper_id' not in class_df.columns or 'Class' not in class_df.columns:
            print("Error: Classification CSV must contain 'paper_id' and 'Class' columns.")
            exit(1)

        # Filter papers that belong to the specified class
        filtered = class_df[class_df['Sub-Class 3'] == args.class_filter]['paper_id'].unique().tolist()
        if not filtered:
            print(f"No papers found for class '{args.class_filter}' in {args.class_file}")
            exit(0)
        print("Got filter lis:", filtered)
        # Intersect with the discovered paper directories
        paper_lis = [pid for pid in paper_lis if pid in filtered]
        print(paper_lis)

        if not paper_lis:
            print(f"No papers from discovered directories belong to class '{args.class_filter}'")
            exit(0)

        print(f"Filtered to {len(paper_lis)} papers for class '{args.class_filter}'")

        # Prepare a clean class name for output subfolder (spaces -> underscores)
        clean_class = args.class_filter.replace(' ', '_')
    else:
        clean_class = None

    # Apply --num_papers limit after filtering
    if args.num_papers is not None:
        paper_lis = paper_lis[:args.num_papers]
        print(f"Limited to first {len(paper_lis)} papers")

    # ------------------------------------------------------------------
    # 4. Determine output directory (include class subfolder if filtering)
    # ------------------------------------------------------------------
    if clean_class:
        dataset_output_dir = os.path.join(args.output_dir, base_dataset, clean_class)
    else:
        dataset_output_dir = os.path.join(args.output_dir, base_dataset)

    os.makedirs(dataset_output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 5. Initialize VLM processor and run
    # ------------------------------------------------------------------
    vlm_processor = VLMImageProcessor(config)

    print(f"\nStarting VLM processing:")
    print(f"  Dataset: {base_dataset}" + (f" (class: {args.class_filter})" if args.class_filter else ""))
    print(f"  Papers to process: {len(paper_lis)}")
    print(f"  Target parameters: {args.target_params}")
    print(f"  Output directory: {dataset_output_dir}")

    if args.serial:
        print(f"  Processing mode: Serial")
    else:
        use_batch = args.batch and not args.force_realtime
        print(f"  Processing mode: {'Batch' if use_batch else 'Parallel real-time'}")
        if not use_batch:
            print(f"  Max workers: {args.max_workers}")

    base_dir = os.path.join(args.base_dir, base_dataset)

    if args.serial:
        results = vlm_processor.batch_process_papers(
            base_dir=base_dir,
            vlm_model_choice=args.vlm_model,
            target_parameters=args.target_params,
            paper_list=paper_lis,
            output_dir=dataset_output_dir
        )
    else:
        parallel_processor = ParallelVLMProcessor(vlm_processor, replace=args.replace)
        use_batch = args.batch and not args.force_realtime
        results = parallel_processor.process_papers_in_parallel(
            base_dir=base_dir,
            vlm_model_choice=args.vlm_model,
            target_parameters=args.target_params,
            paper_list=paper_lis,
            output_dir=dataset_output_dir,
            max_workers=args.max_workers,
            use_batch=use_batch
        )

    print(f"\n=== VLM PROCESSING COMPLETE ===")
    print(f"Total VLM findings: {len(results) if results else 0}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()