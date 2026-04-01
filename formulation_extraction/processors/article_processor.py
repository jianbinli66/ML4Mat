import os
import json
import glob
from pathlib import Path
import pandas as pd
import time

from processors.vlm_img_processor import VLMImageProcessor
from processors.llm import LLMHandler


class ArticleExtractor:
    def __init__(self, config_path="config.json", config=None):
        """Initialize the extractor with configuration"""
        if config is None:
            self.load_config(config_path)
        else:
            self.config = config
            self.dataset_name = self.config.get("dataset", "default_dataset")
            self.extraction_config = self.config.get("extraction", {})

        # Initialize VLM processor
        self.vlm_processor = VLMImageProcessor(self.config)
        # Initialize LLM handler
        self.llm_handler = LLMHandler(self.config)

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                self.dataset_name = self.config.get("dataset", "default_dataset")
                self.extraction_config = self.config.get("extraction", {})
        else:
            raise FileNotFoundError(f"Config file {config_path} not found")

    def setup_model_config(self, model_choice=None, prompt_choice=None, model_type="llm"):
        """Setup model configuration based on choices or config"""
        if model_type == "vlm":
            # Delegate VLM setup to VLM processor
            return self.vlm_processor.setup_vlm_config(model_choice), None

        available_models = self.config.get("available_models", {})

        model_name = model_choice or self.extraction_config.get("default_model")
        prompt_name = prompt_choice or self.extraction_config.get("default_prompt")

        if model_name not in available_models:
            raise ValueError(f"Model {model_name} not found in {model_type} models config")

        # Check if prompt file exists
        prompt_path = f"prompts/{prompt_name}.txt"
        dataset_prompt_path = f"prompts/{self.dataset_name}_{prompt_name}.txt"
        
        if not os.path.exists(prompt_path) and not os.path.exists(dataset_prompt_path):
            raise FileNotFoundError(f"Prompt file {prompt_name}.txt not found in prompts/ directory")

        model_config = available_models[model_name].copy()
        model_config['name'] = model_name

        key_env = model_config.get('key_env')
        if not key_env:
            raise ValueError(f"key_env not specified for model {model_name}")

        api_key = os.getenv(key_env)
        if not api_key:
            raise ValueError(f"Environment variable {key_env} not set for model {model_name}")

        model_config['key'] = api_key
        return model_config, prompt_name

    def extract_text_from_elements(self, pdf_id_dir):
        """Extract text from all available text elements in the PDF directory"""
        extracted_texts = {
            'full_md': '',
            'layout_data': {},
            'content_list': {}
        }

        full_md_path = os.path.join(pdf_id_dir, "full.md")
        if os.path.exists(full_md_path):
            try:
                with open(full_md_path, 'r', encoding='utf-8') as f:
                    extracted_texts['full_md'] = f.read()
            except Exception as e:
                print(f"Error reading full.md: {e}")
        return extracted_texts

    def extract_data_with_llm(self, text_elements, model_config, prompt_name, vlm_analysis_texts=None):
        """Extract data using LLM from text content including VLM analysis"""
        if vlm_analysis_texts is None:
            vlm_analysis_texts = []

        # Combine all text sources for LLM processing
        combined_text = f"""
        MAIN TEXT CONTENT:
        {text_elements['full_md']}
        """

        # Add VLM analysis to the combined text
        if vlm_analysis_texts:
            vlm_analysis_section = "VLM IMAGE ANALYSIS:\n"
            for i, analysis in enumerate(vlm_analysis_texts, 1):
                vlm_analysis_section += f"\n--- Image {i}: {analysis['image_file']} ---\n"
                vlm_analysis_section += f"{analysis['analysis_text']}\n"

            combined_text += f"\n{vlm_analysis_section}"

        if not combined_text.strip():
            return None

        # Use the LLM handler to process the request
        return self.llm_handler.process_extraction_request(
            combined_text, model_config, prompt_name
        )

    def parse_llm_response(self, llm_data):
        """Parse LLM response and extract structured data"""
        if not llm_data:
            return {'metadata': {}, 'experimental_records': [], 'parse_error': 'No LLM data'}

        try:
            # Try to find JSON in the response
            extracted_text = llm_data['extracted_data']
            start_idx = extracted_text.find('{')
            end_idx = extracted_text.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = extracted_text[start_idx:end_idx]
                parsed_data = json.loads(json_str)

                # Ensure structure
                if 'metadata' not in parsed_data:
                    parsed_data['metadata'] = {}
                if 'experimental_records' not in parsed_data:
                    parsed_data['experimental_records'] = []

                return parsed_data
            else:
                return {
                    'metadata': {'raw_text_sample': extracted_text[:200]},
                    'experimental_records': [],
                    'parse_error': 'No JSON structure found in LLM response'
                }

        except json.JSONDecodeError as e:
            return {
                'metadata': {'raw_text_sample': llm_data['extracted_data'][:200]},
                'experimental_records': [],
                'parse_error': f'JSON parsing error: {str(e)}'
            }

    def comprehensive_extraction(self, pdf_id_dir, llm_config, prompt_name, vlm_config=None, target_parameters=None,
                                 use_vlm=True):
        """Perform comprehensive extraction using LLM (text) and optionally VLM (images)"""
        if target_parameters is None:
            target_parameters = ["28d_compressive_strength"]

        pdf_name = os.path.basename(pdf_id_dir)

        print(f"Processing {pdf_id_dir}...")
        print(f"  VLM enabled: {use_vlm}")

        # Step 1: Extract text elements
        text_elements = self.extract_text_from_elements(pdf_id_dir)
        print(f"  Text extracted: {len(text_elements['full_md'])} characters from full.md")

        # Step 2: VLM extraction from images (if enabled)
        vlm_results = []
        vlm_analysis_texts = []
        if use_vlm and vlm_config:
            print("  Running VLM image analysis...")
            vlm_results = self.vlm_processor.process_images_for_paper(
                pdf_id_dir, vlm_config, target_parameters
            )
            # Load VLM analysis texts for LLM processing
            vlm_analysis_texts = self.vlm_processor.load_vlm_analysis(pdf_id_dir)
            print(f"  Loaded VLM analysis texts: {len(vlm_analysis_texts)}")
        else:
            print("  VLM analysis skipped")

        # Step 3: LLM extraction from text including VLM analysis (if available)
        llm_result = None
        if text_elements['full_md'] or vlm_analysis_texts:
            print("  Running LLM text extraction...")
            llm_result = self.extract_data_with_llm(text_elements, llm_config, prompt_name, vlm_analysis_texts)

        # Step 4: Parse LLM results
        parsed_llm_data = self.parse_llm_response(llm_result) if llm_result else {
            'metadata': {},
            'experimental_records': [],
            'parse_error': 'No text content available for LLM processing'
        }

        # Combine all results
        comprehensive_result = {
            'pdf_id': pdf_name,
            'extraction_summary': {
                'text_source': 'full.md',
                'text_length': len(text_elements['full_md']),
                'vlm_enabled': use_vlm,
                'vlm_analysis_texts_used': len(vlm_analysis_texts),
                'llm_records_extracted': len(parsed_llm_data.get('experimental_records', [])),
                'vlm_findings': len(vlm_results),
                'target_parameters': target_parameters,
                'total_timestamp': pd.Timestamp.now().isoformat()
            },
            'llm_extraction': {
                'model_used': llm_result['model_name'] if llm_result else None,
                'prompt_used': llm_result['prompt_name'] if llm_result else None,
                'api_usage': llm_result.get('usage', {}) if llm_result else {},
                'vlm_analysis_included': llm_result.get('vlm_analysis_included', False) if llm_result else False,
                'parsed_data': parsed_llm_data,
                'raw_response': llm_result['extracted_data'] if llm_result else None
            },
            'source_files': {
                'full_md_available': bool(text_elements['full_md']),
                'vlm_analysis_available': len(vlm_analysis_texts) > 0
            }
        }

        return comprehensive_result

    def save_comprehensive_results(self, result, output_dir):
        """Save comprehensive LLM and VLM results to JSON file"""
        os.makedirs(output_dir, exist_ok=True)

        pdf_id = result['pdf_id']
        output_path = os.path.join(output_dir, f"{pdf_id}_comprehensive_extraction.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  Saved comprehensive results: {output_path}")
        print(f"    LLM records: {result['extraction_summary']['llm_records_extracted']}")
        print(f"    VLM findings: {result['extraction_summary']['vlm_findings']}")

        return output_path

    def process_all_papers(self, base_dir, llm_model_choice=None, prompt_choice=None,
                           vlm_model_choice=None, target_parameters=None, output_dir="comprehensive_results",
                           paper_lis=None, use_vlm=True):
        """Process all papers with comprehensive LLM + VLM extraction"""

        if target_parameters is None:
            target_parameters = ["28d_compressive_strength"]

        try:
            # Setup LLM configuration
            llm_config, prompt_name = self.setup_model_config(llm_model_choice, prompt_choice, "llm")

            # Setup VLM configuration only if VLM is enabled
            vlm_config = None
            if use_vlm:
                vlm_config, _ = self.setup_model_config(vlm_model_choice, None, "vlm")

        except ValueError as e:
            print(f"Configuration error: {e}")
            return 0, 0, 0

        print(f"Starting comprehensive extraction:")
        print(f"  LLM Model: {llm_config['name']}")
        print(f"  LLM Prompt: {prompt_name}")
        print(f"  VLM Model: {vlm_config['name'] if vlm_config else 'Disabled'}")
        print(f"  VLM Enabled: {use_vlm}")
        print(f"  Target parameters: {target_parameters}")

        pdf_dirs = []

        if paper_lis:
            for paper_id in paper_lis:
                file_dir = Path(os.path.join(base_dir, str(paper_id) + '.pdf-id'))
                pdf_dirs.append(file_dir)

        if not pdf_dirs:
            print(f"No PDF directories found in {base_dir}")
            return 0, 0, 0

        # skip processed ones
        pdf_dirs = [d for d in pdf_dirs if not os.path.exists(
            os.path.join(output_dir, f"{os.path.basename(d)}_comprehensive_extraction.json"))]
        print(f"Found {len(pdf_dirs)} PDF directories to process")

        successful_processing = 0
        total_llm_records = 0
        total_vlm_findings = 0

        for i, pdf_dir in enumerate(pdf_dirs):
            print(f"\n[{i + 1}/{len(pdf_dirs)}] Processing: {os.path.basename(pdf_dir)}")

            try:
                result = self.comprehensive_extraction(
                    pdf_dir, llm_config, prompt_name, vlm_config, target_parameters, use_vlm
                )

                if result:
                    self.save_comprehensive_results(result, output_dir)
                    successful_processing += 1
                    total_llm_records += result['extraction_summary']['llm_records_extracted']
                    total_vlm_findings += result['extraction_summary']['vlm_findings']

            except Exception as e:
                print(f"Error processing {pdf_dir}: {e}")
                continue

            time.sleep(1)  # Rate limiting between papers

        print(f"\n=== COMPREHENSIVE EXTRACTION COMPLETE ===")
        print(f"Successfully processed: {successful_processing}/{len(pdf_dirs)}")
        print(f"Total LLM experimental records: {total_llm_records}")
        print(f"Total VLM findings: {total_vlm_findings}")
        print(f"VLM enabled: {use_vlm}")
        print(f"Output directory: {output_dir}")

        return successful_processing, total_llm_records, total_vlm_findings