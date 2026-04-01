#!/usr/bin/env python3
# 2_dac_article_classifying.py
# Updated to include Sub-Class 3 for solid amine adsorbents.

import os
import csv
import json
import argparse
import threading
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
import time
from datetime import datetime
import concurrent.futures
from tqdm import tqdm

from processors.llm import LLMHandler


class ParallelArticleClassifier:
    """
    Classifies DAC articles into a detailed nine‑category taxonomy using an LLM in parallel.
    Results are saved incrementally to a CSV file.
    """

    # Full classification taxonomy (in English) – used for prompt construction
    TAXONOMY = {
        "Review and Perspective": {
            "Systematic Review of DAC Technologies": [],
            "Topical Review (Materials, Processes, Electrochemistry, Economics, etc.)": [],
            "Perspective/Commentary/Editorial": [],
            "Bibliometric Analysis / Knowledge Mapping": [],
            "Patent Analysis / Technology Landscape": [],
            "Standards/White Papers/Industry Reports": []
        },
        "Core Capture Materials": {
            "Liquid Absorbents": [
                "Aqueous Amine Solutions (MEA, DEA, MDEA, hindered amines, blends)",
                "Inorganic Alkali Solutions (NaOH, KOH, Ca(OH)₂)",
                "Ionic Liquids (ILs)",
                "Deep Eutectic Solvents (DES)",
                "Phase-change Absorbents (liquid-solid/liquid-liquid)"
            ],
            "Solid Adsorbents": [
                "Solid Amine Adsorbents (supported amines, polymer-grafted amines)",
                "Physical Adsorbents (porous carbons, zeolites, porous oxides)",
                "Crystalline Porous Materials (MOFs, COFs, POPs)",
                "Mineral-based/Natural Materials (calcium-based, magnesium-based, industrial waste/minerals)",
                "Novel Solid Adsorbents (LDHs, POCs, multicomponent composites)"
            ],
            "Membrane Materials": [
                "Inorganic Membranes (ceramic, zeolite)",
                "Organic Membranes (polyimide, polysulfone)",
                "Mixed Matrix Membranes (MMMs)",
                "Facilitated Transport Membranes (fixed/mobile carrier CO₂ transport)"
            ]
        },
        "Electrochemical/Photoelectrochemical DAC": {
            "Electrochemically Mediated Adsorption/Desorption (ESA)": [],
            "Electrocatalytic CO₂ Reduction Coupled with DAC (eDAC+CO₂RR)": [],
            "Photoelectrochemical DAC": [],
            "Membrane Electrolysis for Alkali Regeneration": [],
            "Integrated Electrochemical Capture-Mineralization": []
        },
        "Processes, Reactors, and Engineering": {
            "Regeneration Processes (TSA, VSA, PSA, HSA)": [],
            "Reactor Design (fixed bed, fluidized bed, moving bed, rotating bed)": [],
            "Pilot/Demonstration Studies": [],
            "Scale-up and Engineering Challenges": []
        },
        "Material Mechanisms and Performance Evaluation": {
            "Capture Mechanisms (physisorption, chemisorption, interfacial coordination)": [],
            "Thermodynamics/Kinetics/Mass Transfer": [],
            "Material Characterization (XRD, XPS, FTIR, in situ)": [],
            "Stability and Degradation Mechanisms (moisture resistance, oxidation, cycling deactivation)": [],
            "Selectivity and Interference Resistance (O₂, H₂O, impurities)": []
        },
        "Economic, Environmental, and Policy Assessment": {
            "Techno-economic Analysis (TEA), Cost Estimation": [],
            "Life Cycle Assessment (LCA), Carbon Footprint": [],
            "Energy Consumption and Renewable Energy Integration": [],
            "Carbon Trading, Policy, and Carbon Neutrality Pathways": [],
            "Carbon Dioxide Removal (CDR) Benefit Analysis": []
        },
        "System Integration and Coupling Applications": {
            "DAC-Renewable Energy Coupling (solar, wind, geothermal)": [],
            "DAC+CCU (fuels, chemicals, materials)": [],
            "DAC+CCS/Mineral Carbonation (geologic storage, mineralization)": [],
            "Negative Emissions Integration (DAC+BECCS, etc.)": []
        },
        "Computational Simulation and Data-driven Approaches": {
            "Quantum Chemistry/Molecular Simulation (DFT, MD, GCMC)": [],
            "Process Simulation (Aspen, gPROMS)": [],
            "Machine Learning-assisted Materials Design": [],
            "Multiscale Modeling (atom → reactor → system)": []
        },
        "Experimental Methods and Testing Standards": {
            "DAC Performance Testing Methods (dynamic breakthrough, static adsorption)": [],
            "Standardized Material Testing Protocols": [],
            "In-situ/Operando Characterization Techniques": [],
            "Standardized Test Platforms": []
        }
    }

    # Solid amine preparation methods – definitions for Sub-Class 3
    SOLID_AMINE_METHODS = {
        'Physical Impregnation Method': """
            A basic and commonly used method for solid amine preparation, in which amine reagents 
            (small-molecule polyamines, polymeric amines, etc.) are dissolved in polar solvents 
            such as water/alcohols, and dry porous supports are immersed in the amine solution. 
            Amine molecules enter the support pores and adsorb on the surface through physical 
            interactions such as capillary action, hydrogen bonding, and van der Waals forces, 
            and then the solvent is removed by evaporation, vacuum drying and other methods to 
            realize amine loading.

            Common Implementation Methods: Impregnation-evaporation method, vacuum impregnation 
            method, incipient wetness impregnation method.
        """,
        'Covalent Grafting Method': """
            A preparation method that permanently anchors amino groups on the support surface 
            by forming stable covalent bonds (e.g., Si-O-Si, C-N, Al-O-Si) through condensation 
            reaction, ring-opening reaction, amidation reaction, etc., between active functional 
            groups on the surface of porous supports (e.g., ≡Si-OH of silica-based supports, 
            -Al-OH of alumina-based supports, -COOH/-OH of oxidized carbon-based supports) and 
            amine reagents (mainly amine-functional silane coupling agents, or reactive amino 
            groups introduced by modification).

            Common Implementation Methods: Wet grafting, dry grafting, stepwise grafting.
        """,
        'In-situ Polymerization Method': """
            A preparation method in which amine monomers/prepolymers (e.g., ethyleneimine, 
            allylamine) are introduced into the pores of porous supports together with 
            initiators/catalysts, and thermal initiation, photoinitiation, chemical initiator 
            initiation and other methods are used to trigger in-situ polymerization of amine 
            monomers in and on the surface of support pores. The generated polyamine molecules 
            are fixed on the support due to steric hindrance effect to realize amino group loading.

            Common Implementation Methods: Thermally initiated in-situ polymerization, 
            UV-initiated in-situ polymerization, support-grafted initiator method.
        """,
        'Impregnation-Grafting Composite Method': """
            A composite preparation method that combines the advantages of high loading capacity 
            of physical impregnation method and high stability of covalent grafting method. 
            A small amount of amino groups are first grafted on the surface of porous supports 
            as chemical anchor points by covalent grafting method, and then amine reagents are 
            loaded into the support pores by physical impregnation method. The anchor points and 
            impregnated amines form a synergistic combination through hydrogen bonding and 
            amino-amino interactions to realize high amine loading and high-stability binding.

            Common Implementation Methods: Grafting first then impregnation, impregnation first 
            then crosslinking grafting.
        """
    }

    def __init__(self, config_path: str = "config.json", replace: bool = False):
        """Initialize the classifier with configuration"""
        self.load_config(config_path)
        self.llm_handler = LLMHandler(self.config)
        self.replace = replace
        self.output_csv = None
        self._csv_lock = threading.Lock()
        self._header_written = False

    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError(f"Config file {config_path} not found")

    def setup_model_config(self, model_choice: str = None):
        """Setup model configuration for LLM"""
        available_models = self.config.get("available_models", {})

        model_name = model_choice or self.config.get("extraction", {}).get("default_model")

        if model_name not in available_models:
            raise ValueError(f"Model {model_name} not found in models config")

        model_config = available_models[model_name].copy()
        model_config['name'] = model_name

        key_env = model_config.get('key_env')
        if not key_env:
            raise ValueError(f"key_env not specified for model {model_name}")

        api_key = os.getenv(key_env)
        if not api_key:
            raise ValueError(f"Environment variable {key_env} not set for model {model_name}")

        model_config['key'] = api_key
        return model_config

    def get_first_n_words(self, text: str, n: int = 500) -> str:
        """Get the first n words from text"""
        if not text:
            return ""
        words = text.split()
        if len(words) <= n:
            return text
        return ' '.join(words[:n])

    def prepare_article_tasks(self, pdf_dirs: List[Path], model_config: Dict[str, Any],
                              num_words: int = 500) -> List[Dict[str, Any]]:
        """Prepare all article processing tasks"""
        tasks = []
        for pdf_dir in pdf_dirs:
            full_md_path = pdf_dir / "full.md"
            pdf_filename = f"{pdf_dir.stem}.pdf"
            if not full_md_path.exists():
                print(f"Warning: full.md not found in {pdf_dir}")
                continue
            tasks.append({
                'full_md_path': str(full_md_path),
                'pdf_filename': pdf_filename,
                'paper_id': pdf_dir.stem,
                'num_words': num_words,
                'model_config': model_config.copy()
            })
        return tasks

    def process_single_article_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single article task and return a dictionary with all required fields."""
        full_md_path = task['full_md_path']
        pdf_filename = task['pdf_filename']
        paper_id = task['paper_id']
        num_words = task['num_words']
        model_config = task['model_config']

        try:
            with open(full_md_path, 'r', encoding='utf-8') as f:
                content = f.read()

            content_sample = self.get_first_n_words(content, num_words)
            classification_prompt = self._create_classification_prompt(content_sample)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a materials science and Direct Air Capture (DAC) expert. "
                        "Your task is to classify research articles according to a detailed nine‑category taxonomy. "
                        "Always respond with a valid JSON object containing the requested fields. "
                        "Do not include any special symbols that would break JSON parsing."
                    )
                },
                {"role": "user", "content": classification_prompt}
            ]

            response = self.llm_handler.call_llm_api(
                url=model_config['url'],
                key=model_config['key'],
                model=model_config['name'],
                messages=messages,
                temperature=0.1
            )

            if response and hasattr(response, 'choices') and response.choices:
                extracted_text = response.choices[0].message.content

                # Extract JSON from response
                try:
                    start_idx = extracted_text.find('{')
                    end_idx = extracted_text.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = extracted_text[start_idx:end_idx]
                        parsed_data = json.loads(json_str)

                        # Map to our fieldnames
                        result = {
                            'Title': parsed_data.get('Title', ''),
                            'Class': parsed_data.get('Main_Category', ''),
                            'Sub-Class 1': parsed_data.get('Sub_Category_1', ''),
                            'Sub-Class 2': parsed_data.get('Sub_Category_2', ''),
                            'Sub-Class 3': parsed_data.get('Sub_Category_3', ''),
                            'Secondary_Classification': parsed_data.get('Secondary_Classification', ''),
                            'Abstract': parsed_data.get('Abstract', ''),
                            'Keywords': parsed_data.get('Keywords', ''),
                            'DOI': parsed_data.get('DOI', ''),
                            'PDF_file_name': pdf_filename,
                            'paper_id': paper_id,
                            'Classification_Confidence': parsed_data.get('Classification_Confidence', 0),
                            'Classification_Reasoning': parsed_data.get('Classification_Reasoning', '')
                        }
                        return result
                except json.JSONDecodeError as e:
                    pass  # fall through to error handling

            # If we reach here, something went wrong
            return {
                'Title': '',
                'Class': 'TBC',
                'Sub-Class 1': '',
                'Sub-Class 2': '',
                'Sub-Class 3': '',
                'Secondary_Classification': '',
                'Abstract': '',
                'Keywords': '',
                'DOI': '',
                'PDF_file_name': pdf_filename,
                'paper_id': paper_id,
                'Classification_Confidence': 0,
                'Classification_Reasoning': 'LLM response could not be parsed or was empty.'
            }

        except Exception as e:
            return {
                'Title': '',
                'Class': 'TBC',
                'Sub-Class 1': '',
                'Sub-Class 2': '',
                'Sub-Class 3': '',
                'Secondary_Classification': '',
                'Abstract': '',
                'Keywords': '',
                'DOI': '',
                'PDF_file_name': pdf_filename,
                'paper_id': paper_id,
                'Classification_Confidence': 0,
                'Classification_Reasoning': f'Processing error: {str(e)}'
            }

    def _create_classification_prompt(self, content_sample: str) -> str:
        """Create a detailed prompt incorporating the full DAC taxonomy and solid amine sub‑class 3."""
        # Build taxonomy string for the prompt
        taxonomy_lines = []
        for main_cat, subcats in self.TAXONOMY.items():
            taxonomy_lines.append(f"\n{main_cat}:")
            for subcat, subsubcats in subcats.items():
                if subsubcats:
                    taxonomy_lines.append(f"  - {subcat}")
                    for subsub in subsubcats:
                        taxonomy_lines.append(f"    * {subsub}")
                else:
                    taxonomy_lines.append(f"  - {subcat}")

        taxonomy_str = "\n".join(taxonomy_lines)

        # Build solid amine methods string (definitions)
        methods_str = ""
        for name, definition in self.SOLID_AMINE_METHODS.items():
            methods_str += f"\n{name}:\n{definition}\n{'-'*50}\n"

        prompt = f"""
Return a valid JSON object only.

You are an expert in materials science and Direct Air Capture (DAC). Analyze the following research article content and classify it according to the provided nine‑category taxonomy. Follow the classification rules strictly.

ARTICLE CONTENT (First 500 words):
{content_sample}

CLASSIFICATION TAXONOMY (use only these categories, no others):
{taxonomy_str}

SPECIAL INSTRUCTION FOR SOLID AMINE ADSORBENTS:
If the Sub_Category_2 is "Solid Amine Adsorbents", you MUST further classify the preparation method into Sub_Category_3 using one of the four definitions below. If Sub_Category_2 is NOT "Solid Amine Adsorbents", set Sub_Category_3 to an empty string.

DEFINITIONS OF SOLID AMINE PREPARATION METHODS (use these to determine Sub_Category_3):
{methods_str}

CLASSIFICATION RULES:
1. Assign the most specific leaf category possible. If the paper fits a sub‑subcategory, use it.
2. Output the main category, sub‑category 1, and sub‑category 2 (the most specific). If a level does not apply (e.g., only two levels), leave sub‑category 2 empty.
3. For Sub_Category_3: follow the special instruction above.
4. If the paper has a secondary focus (e.g., covers two distinct main areas), provide a secondary classification as a free‑text string in the format "Main - Sub1 - Sub2". Otherwise leave it empty.
5. If the paper cannot be classified (e.g., abstract too vague), set Main_Category to "TBC" and explain why in the reasoning field.
6. Confidence should be an integer 0–100.
7. Extract the title, abstract, keywords, and DOI if present in the sample. Abstract should not be empty if present.
8. All text must be in English, no special symbols (LaTeX, etc.).

RESPONSE FORMAT (JSON):
{{
    "Title": "Full article title",
    "Main_Category": "top‑level category",
    "Sub_Category_1": "second‑level category (empty if none)",
    "Sub_Category_2": "most specific category (empty if none)",
    "Sub_Category_3": "empty or one of the four solid amine preparation methods (if applicable)",
    "Secondary_Classification": "if applicable, otherwise empty string",
    "Abstract": "full abstract text",
    "Keywords": "comma‑separated keywords",
    "DOI": "DOI number",
    "Classification_Confidence": 0-100,
    "Classification_Reasoning": "brief explanation of the choice"
}}

Return ONLY the JSON object, no other text.
"""
        return prompt

    def _append_result_to_csv(self, result: Dict[str, Any]):
        """Thread‑safe append of a single result to the CSV file."""
        if not self.output_csv:
            return

        fieldnames = [
            'Title', 'Class', 'Sub-Class 1', 'Sub-Class 2', 'Sub-Class 3', 'Secondary_Classification',
            'Abstract', 'Keywords', 'DOI', 'PDF_file_name',
            'Classification_Confidence', 'Classification_Reasoning', 'paper_id'
        ]

        with self._csv_lock:
            file_exists = os.path.isfile(self.output_csv)
            write_header = not file_exists or os.path.getsize(self.output_csv) == 0

            with open(self.output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)
                csvfile.flush()

    def _clean_csv_of_unknown_abstracts(self, csv_path: str):
        """Remove rows where Abstract equals "Unknown"."""
        if not os.path.exists(csv_path):
            return 0
        try:
            df = pd.read_csv(csv_path, dtype=str)
            initial_count = len(df)
            df_cleaned = df[df['Abstract'].notna()]
            removed = initial_count - len(df_cleaned)
            if removed > 0:
                df_cleaned.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"Cleaned CSV: removed {removed} row(s) with 'Unknown' Abstract.")
            return removed
        except Exception as e:
            print(f"Warning: could not clean CSV {csv_path}: {e}")
            return 0

    def process_with_parallel(self, tasks: List[Dict[str, Any]], max_workers: int = 5) -> List[Dict[str, Any]]:
        """Process tasks in parallel using ThreadPoolExecutor and save results incrementally."""
        print(f"Processing {len(tasks)} articles in parallel with {max_workers} workers...")
        results = []
        failed_tasks = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(self.process_single_article_task, task): task for task in tasks}

            with tqdm(total=len(tasks), desc="Classifying articles") as pbar:
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    print("processing: ", task['paper_id'])
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            self._append_result_to_csv(result)
                        else:
                            failed_tasks.append(task)
                    except Exception as e:
                        print(f"\nTask failed: {task['paper_id']} - Error: {e}")
                        failed_tasks.append(task)
                    finally:
                        pbar.update(1)
                        time.sleep(3)

        print(f"\nCompleted: {len(results)} successful, {len(failed_tasks)} failed")
        return results

    def print_summary(self, results: List[Dict]):
        """Print summary statistics of classification (based on Main Category = Class)."""
        total = len(results)
        if total == 0:
            print("No results to summarize")
            return

        class_counts = {}
        confidence_sum = 0
        confidence_count = 0

        for result in results:
            cls = result.get('Class', 'TBC')
            class_counts[cls] = class_counts.get(cls, 0) + 1
            confidence = result.get('Classification_Confidence', 0)
            if isinstance(confidence, (int, float)) and confidence > 0:
                confidence_sum += confidence
                confidence_count += 1

        print("\n" + "=" * 60)
        print("CLASSIFICATION SUMMARY")
        print("=" * 60)
        print(f"Total articles processed: {total}")
        print("\nClass Distribution (Main Category):")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            print(f"  {cls}: {count} ({percentage:.1f}%)")

        if confidence_count > 0:
            avg_confidence = confidence_sum / confidence_count
            print(f"\nAverage classification confidence: {avg_confidence:.1f}%")
        print("=" * 60)

    def process_all_articles(self, base_dir: str, output_csv: str, paper_lis: List[str] = None,
                             model_choice: str = None, num_words: int = 500,
                             max_workers: int = 5, resume: bool = False) -> List[Dict]:
        """
        Process all articles in the base directory with parallel processing.
        Results are saved incrementally to the CSV file.
        """
        base_path = Path(base_dir)
        print("Base directory: ", base_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Base directory {base_dir} not found")

        self.output_csv = output_csv

        # Fresh start if not resuming
        if not resume and os.path.exists(output_csv):
            os.remove(output_csv)
            self._header_written = False
            print("Resume=False: removed existing CSV to start fresh.")

        # Clean CSV if resuming
        processed_papers = set()
        print("Processing all articles...", output_csv)
        if resume and os.path.exists(output_csv):
            self._clean_csv_of_unknown_abstracts(output_csv)
            try:
                df = pd.read_csv(output_csv, dtype=str)
                if 'paper_id' in df.columns:
                    processed_papers = set(df['paper_id'].tolist())
                print(f"Found {len(processed_papers)} already processed articles (after cleaning). Resuming...")
                if 'PDF_file_name' in df.columns:
                    processed_papers.update(set(df['PDF_file_name'].tolist()))
            except Exception as e:
                print(f"Could not read existing CSV file: {e}. Starting fresh.")
                processed_papers = set()

        # Setup model configuration
        model_config = self.setup_model_config(model_choice)
        print(f"Using model: {model_config['name']}")

        # Find all PDF-id directories
        pdf_dirs = []
        if paper_lis:
            for paper_id in paper_lis:
                dir_path = base_path / f"{paper_id}"
                # print(dir_path)
                if dir_path.exists():
                    pdf_dirs.append(dir_path)
        else:
            pdf_dirs = [d for d in base_path.glob("*.pdf-id") if d.is_dir()]

        print(f"Found {len(pdf_dirs)} article directories")

        # Filter out already processed articles
        pdf_dirs_to_process = []
        for pdf_dir in pdf_dirs:
            paper_id = pdf_dir.stem + '.pdf'
            if paper_id not in processed_papers or self.replace:
                pdf_dirs_to_process.append(pdf_dir)

        print(f"Articles to process: {len(pdf_dirs_to_process)}")

        if not pdf_dirs_to_process:
            print("All articles already processed!")
            if os.path.exists(output_csv):
                df = pd.read_csv(output_csv)
                results = df.to_dict('records')
                self.print_summary(results)
                return results

        # Prepare tasks
        print("Preparing article processing tasks...")
        tasks = self.prepare_article_tasks(pdf_dirs_to_process, model_config, num_words)
        print(f"Total articles to process: {len(tasks)}")

        if not tasks:
            print("No articles found to process")
            return []

        # Process in parallel (results saved incrementally)
        results = self.process_with_parallel(tasks, max_workers)

        # Print summary
        self.print_summary(results)

        return results


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(
        description='Classify DAC articles into a nine‑category taxonomy using LLM in parallel')
    parser.add_argument('--dataset', type=str, default='DAC', help='Dataset name')
    parser.add_argument('--base_dir', type=str, default='data/pdfs2textimgs',
                        help='Base directory containing PDF-id directories with full.md files')
    parser.add_argument('--output', type=str, default='article_classification.csv',
                        help='Output CSV file path')
    parser.add_argument('--num_papers', type=int, default=None,
                        help='Number of papers to process (default: all)')
    parser.add_argument('--model', type=str, default=None,
                        help='LLM model to use for classification (default: from config)')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Config file path')
    parser.add_argument('--num_words', type=int, default=500,
                        help='Number of words to use from the beginning of each article (default: 500)')
    parser.add_argument('--max_workers', type=int, default=5,
                        help='Maximum number of parallel workers (default: 5)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from existing CSV file, skipping already processed articles')
    parser.add_argument('--replace', action='store_true', default=False,
                        help='Replace existing results (reprocess all)')

    args = parser.parse_args()

    # Load config
    config = json.load(open(args.config))
    dataset = args.dataset
    
    # Dynamically discover paper directories
    dataset_dir = os.path.join(args.base_dir, dataset)
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

    print(f"Dataset: {dataset}")
    print(f"Number of papers to process: {len(paper_lis)}")
    print(f"Number of words per article: {args.num_words}")
    print(f"Max parallel workers: {args.max_workers}")
    print(f"Resume mode: {args.resume}")
    print(f"Replace mode: {args.replace}")

    # Initialize classifier
    classifier = ParallelArticleClassifier(args.config, replace=args.replace)

    # Process articles
    base_dir = os.path.join(args.base_dir, dataset)
    results = classifier.process_all_articles(
        base_dir=base_dir,
        output_csv=args.output,
        paper_lis=paper_lis,
        model_choice=args.model,
        num_words=args.num_words,
        max_workers=args.max_workers,
        resume=args.resume
    )

    print(f"\nProcessing complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main()