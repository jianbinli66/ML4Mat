# vlm_img_processor.py
import os
import glob
import json
import base64
import time
import re
import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import io
import pandas as pd
from openai import OpenAI


class VLMImageProcessor:
    def __init__(self, config=None):
        """Initialize VLM processor with configuration"""
        self.config = config or {}
        self.dataset_name = self.config.get("dataset", "default_dataset")
        self.client = None
        self.batch_mode = False

    def extract_figure_captions_from_md(self, pdf_id_dir):
        """Extract figure captions from full.md in the paper directory"""
        full_md_path = os.path.join(pdf_id_dir, "full.md")

        if not os.path.exists(full_md_path):
            print(f"  full.md not found at: {full_md_path}")
            return ""

        with open(full_md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = r'^(?:Figure|Fig\.?)\s*\d+[\.:]?\s*(.*?)$'

        captions = []
        for line in content.split('\n'):
            line = line.strip()
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                caption = match.string
                if caption:
                    captions.append(caption)

        return "\n".join(captions) if captions else ""

    def setup_vlm_config(self, model_choice=None):
        """Setup VLM model configuration"""
        available_models = self.config.get("vlm_models", {})

        model_name = model_choice or self.config.get("extraction", {}).get("default_vlm_model", "gpt-4-vision-preview")

        if model_name not in available_models:
            raise ValueError(f"VLM model {model_name} not found in config")

        model_config = available_models[model_name].copy()
        model_config['name'] = model_name

        key_env = model_config.get('key_env', 'DASHSCOPE_API_KEY')
        api_key = os.getenv(key_env)

        if not api_key:
            raise ValueError(f"Environment variable {key_env} not set for VLM model {model_name}")

        model_config['key'] = api_key

        # Set AliYun base URL if not specified
        if 'url' not in model_config and 'aliyun' in model_name.lower():
            model_config['url'] = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

        return model_config

    def init_aliyun_client(self, model_choice=None):
        """Initialize AliYun Bailian client for batch processing"""
        try:
            model_config = self.setup_vlm_config(model_choice)

            # Check if model is supported by AliYun Batch
            supported_models = self.get_aliyun_supported_models()
            model_name = model_config['name']

            if model_name not in supported_models:
                print(f"Warning: {model_name} not in AliYun Batch supported models, falling back to real-time")
                return False, model_config

            # Initialize AliYun client
            api_key = os.getenv("DASHSCOPE_API_KEY") or model_config.get('key')
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY environment variable not set")

            base_url = model_config.get('url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')

            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self.batch_mode = True
            print(f"Initialized AliYun Batch client for {model_name}")
            return True, model_config

        except Exception as e:
            print(f"Failed to initialize AliYun client: {e}, falling back to real-time")
            return False, None

    def get_aliyun_supported_models(self) -> List[str]:
        """Get list of models supported by AliYun Batch interface"""
        return [
            # Qwen text generation models
            "qwen3-max", "qwen-max", "qwen-max-latest",
            "qwen-plus", "qwen-plus-latest",
            "qwen-flash",
            "qwen-turbo", "qwen-turbo-latest",
            "qwen-long", "qwen-long-latest",
            "qwq-plus", "qwq-32b-preview",
            # Third-party models
            "deepseek-r1", "deepseek-v3",
            # Multi-modal models
            "qwen3-vl-plus", "qwen3-vl-flash",
            "qwen-vl-max", "qwen-vl-max-latest",
            "qwen-vl-plus", "qwen-vl-plus-latest",
            "qwen-vl-ocr", "qwen-vl-ocr-latest",
            "qwen-omni-turbo"
        ]

    def get_images_list(self, pdf_id_dir):
        """Get list of all images in the PDF directory"""
        images_dir = os.path.join(pdf_id_dir, "images")
        if not os.path.exists(images_dir):
            return []

        image_files = glob.glob(os.path.join(images_dir, "*.png")) + \
                      glob.glob(os.path.join(images_dir, "*.jpg")) + \
                      glob.glob(os.path.join(images_dir, "*.jpeg"))

        return image_files

    def encode_image(self, image_path, resize_factor=0.5, quality=85):
        """Encode image to base64 with optional resizing"""
        if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"  Skipping unsupported image format: {os.path.basename(image_path)}")
            return None

        try:
            with Image.open(image_path) as img:
                # Convert PNG to JPEG if needed
                if img.format == 'PNG':
                    img = img.convert('RGB')

                # Resize if needed
                if resize_factor < 1.0:
                    width, height = img.size
                    new_width = int(width * resize_factor)
                    new_height = int(height * resize_factor)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Save to bytes buffer
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality)

                # Encode to base64
                img_bytes = buffer.getvalue()
                return base64.b64encode(img_bytes).decode('utf-8')

        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def load_vlm_prompt(self, target_parameter):
        """Load specialized prompt for VLM extraction"""
        prompt_path = f"prompts/{self.dataset_name}_VLM_prompt_4_{target_parameter}.txt"

        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()

        # Try generic prompt
        generic_prompt_path = f"prompts/generic_VLM_prompt.txt"
        if os.path.exists(generic_prompt_path):
            with open(generic_prompt_path, "r", encoding="utf-8") as f:
                return f.read()

        # Try dataset-specific generic prompt
        dataset_prompt_path = f"prompts/{self.dataset_name}_VLM_prompt.txt"
        if os.path.exists(dataset_prompt_path):
            with open(dataset_prompt_path, "r", encoding="utf-8") as f:
                return f.read()

        # Return default prompt
        return f"Analyze this scientific figure for {target_parameter} data. Extract any relevant experimental data, values, units, and conditions shown in the figure. Return your analysis in JSON format."

    def call_vlm_api(self, image_path, prompt, model_config, max_retries=3):
        """Call VLM API for image analysis with figure captions from full.md"""
        # Extract figure captions from full.md if pdf_id_dir is provided
        figure_captions_info = ""
        pdf_id_dir = Path(image_path).parent.parent

        if pdf_id_dir:
            figure_captions = self.extract_figure_captions_from_md(pdf_id_dir)
            if figure_captions:
                figure_captions_info = "\n\nPotential figure captions from the article:\n" + figure_captions

        # Add figure captions to the prompt
        enhanced_prompt = prompt + figure_captions_info

        for attempt in range(max_retries):
            try:
                # Encode image
                base64_image = self.encode_image(image_path)
                if not base64_image:
                    return None

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",
                             "text": f"(Add the file_name to the output) {os.path.basename(image_path)}, {enhanced_prompt}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ]
                client = OpenAI(
                    api_key=model_config['key'],
                    base_url=model_config.get('url', 'https://api.openai.com/v1'),
                )

                response = client.chat.completions.create(
                    model=model_config['name'],
                    messages=messages,
                    temperature=0.3,
                )

                return response.choices[0].message.content

            except Exception as e:
                print(f"VLM API attempt {attempt + 1} failed for {image_path}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None

    def save_vlm_analysis(self, images_dir, image_filename, analysis_text):
        """Save VLM analysis text to the image directory"""
        analysis_filename = os.path.splitext(image_filename)[0] + "_vlm_analysis.txt"
        analysis_path = os.path.join(images_dir, analysis_filename)

        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(analysis_text)

        print(f"  Saved VLM analysis: {analysis_filename}")
        return analysis_path

    def load_vlm_analysis(self, pdf_id_dir):
        """Load all VLM analysis texts from the image directory"""
        images_dir = os.path.join(pdf_id_dir, "images")
        if not os.path.exists(images_dir):
            return []

        analysis_files = glob.glob(os.path.join(images_dir, "*_vlm_analysis.txt"))
        analysis_texts = []

        for analysis_file in analysis_files:
            try:
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    analysis_text = f.read()
                    analysis_texts.append({
                        'image_file': os.path.basename(analysis_file).replace('_vlm_analysis.txt', ''),
                        'analysis_text': analysis_text
                    })
            except Exception as e:
                print(f"Error reading VLM analysis file {analysis_file}: {e}")

        return analysis_texts

    def parse_vlm_response(self, response):
        """Parse VLM response and extract structured data"""
        try:
            # If already a dict, return as-is
            if isinstance(response, dict):
                return response

            response_str = str(response)

            # Try to parse directly as JSON
            try:
                parsed_data = json.loads(response_str)
                return parsed_data
            except json.JSONDecodeError:
                pass

            # Try to find JSON within the response
            start_idx = response_str.find('{')
            end_idx = response_str.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response_str[start_idx:end_idx]

                # Clean the JSON string
                json_str = self.clean_and_repair_json(json_str)

                try:
                    parsed_data = json.loads(json_str)
                    return parsed_data
                except json.JSONDecodeError as e:
                    return {
                        'analysis_text': response_str,
                        'parse_error': f'JSON parsing error: {str(e)}',
                        'json_attempt': json_str[:200]
                    }
            else:
                return {
                    'analysis_text': response_str,
                    'parse_error': 'No JSON structure found in VLM response'
                }

        except Exception as e:
            print(f"Error in parse_vlm_response: {e}")
            return {
                'analysis_text': str(response) if response else "Empty response",
                'parse_error': f'Unexpected error: {str(e)}'
            }

    def clean_and_repair_json(self, json_str):
        """Clean and repair JSON string"""
        if not json_str:
            return json_str

        # Remove markdown code blocks
        json_str = json_str.replace('```json', '').replace('```', '').strip()

        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Fix missing quotes around property names
        json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)

        # Fix escaped quotes
        json_str = json_str.replace('\\"', '"')

        return json_str

    def process_images_for_paper(self, pdf_id_dir, vlm_config, target_parameters):
        """Process all images in a paper directory with VLM"""
        images_dir = os.path.join(pdf_id_dir, "images")
        print(f"  VLM analyzing images in {pdf_id_dir}")

        image_files = self.get_images_list(pdf_id_dir)
        vlm_results = []

        for target_param in target_parameters:
            prompt = self.load_vlm_prompt(target_param)

            for img_path in image_files:
                img_filename = os.path.basename(img_path)
                analysis_text_file = os.path.splitext(img_filename)[0] + "_vlm_analysis.txt"
                analysis_file_path = os.path.join(images_dir, analysis_text_file)

                # Check if analysis already exists
                if os.path.exists(analysis_file_path):
                    with open(analysis_file_path, 'r', encoding='utf-8') as f:
                        response = f.read()
                        print(f"  Using existing analysis file: {analysis_file_path}")
                else:
                    response = self.call_vlm_api(img_path, prompt, vlm_config)
                    if response:
                        self.save_vlm_analysis(images_dir, img_filename, response)
                    time.sleep(1)  # Rate limiting between VLM calls

                if response:
                    parsed_data = self.parse_vlm_response(response)
                    parsed_data['image_file'] = img_filename
                    parsed_data['parameter_target'] = target_param
                    parsed_data['extraction_timestamp'] = datetime.now().isoformat()
                    vlm_results.append(parsed_data)

        return vlm_results

    def batch_process_papers(self, base_dir, vlm_model_choice=None, target_parameters=None,
                             paper_list=None, output_dir="vlm_results"):
        """Batch process multiple papers with VLM"""
        if target_parameters is None:
            target_parameters = ["28d_compressive_strength"]

        try:
            vlm_config = self.setup_vlm_config(vlm_model_choice)
        except ValueError as e:
            print(f"VLM configuration error: {e}")
            return []

        print(f"Starting VLM batch processing:")
        print(f"  VLM Model: {vlm_config['name']}")
        print(f"  Target parameters: {target_parameters}")

        # Collect paper directories
        pdf_dirs = []
        if paper_list:
            for paper_id in paper_list:
                paper_dir = Path(os.path.join(base_dir, str(paper_id) + '.pdf-id'))
                if os.path.exists(paper_dir):
                    pdf_dirs.append(paper_dir)

        if not pdf_dirs:
            print(f"No PDF directories found in {base_dir}")
            return []

        all_results = []

        for i, pdf_dir in enumerate(pdf_dirs):
            print(f"\n[{i + 1}/{len(pdf_dirs)}] VLM Processing: {os.path.basename(pdf_dir)}")

            try:
                paper_results = self.process_images_for_paper(
                    pdf_dir, vlm_config, target_parameters
                )

                if paper_results:
                    all_results.extend(paper_results)

            except Exception as e:
                print(f"Error VLM processing {pdf_dir}: {e}")
                continue

            time.sleep(1)  # Rate limiting between papers

        print(f"\n=== VLM BATCH PROCESSING COMPLETE ===")
        print(f"Total papers processed: {len(pdf_dirs)}")
        print(f"Total VLM findings: {len(all_results)}")

        return all_results


class ParallelVLMProcessor:
    def __init__(self, vlm_processor: VLMImageProcessor, replace=False):
        """Initialize parallel processor with VLM processor"""
        self.vlm_processor = vlm_processor
        self.replace = replace
        self.batch_mode = False

    def prepare_image_tasks(self, pdf_dirs: List[str], target_parameters: List[str],
                            vlm_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare all image processing tasks"""
        tasks = []

        for pdf_dir in pdf_dirs:
            images_dir = os.path.join(pdf_dir, "images")
            if not os.path.exists(images_dir):
                continue

            image_files = self.vlm_processor.get_images_list(pdf_dir)
            if self.replace:
                # remove all existed figures
                [os.remove(f) for f in glob.glob(os.path.join(images_dir, "*.txt"))]
            for target_param in target_parameters:
                try:
                    prompt = self.vlm_processor.load_vlm_prompt(target_param)
                except FileNotFoundError:
                    print(f"Warning: Prompt for {target_param} not found, skipping")
                    continue

                for img_path in image_files:
                    tasks.append({
                        'image_path': img_path,
                        'target_param': target_param,
                        'prompt': prompt,
                        'vlm_config': vlm_config,
                        'pdf_dir': pdf_dir,
                        'paper_id': os.path.basename(pdf_dir)
                    })

        return tasks

    def prepare_batch_input_file(self, tasks: List[Dict[str, Any]], output_path: str) -> str:
        """Prepare JSONL input file for AliYun Batch processing"""
        jsonl_lines = []

        for i, task in enumerate(tasks):
            # Encode image
            base64_image = self.vlm_processor.encode_image(task['image_path'])
            if not base64_image:
                continue

            # Construct the request body
            request_body = {
                "model": task['vlm_config']['name'],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"(Add the file_name to the output) {os.path.basename(task['image_path'])}, {task['prompt']}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.3
            }

            # Add additional parameters
            if 'max_tokens' in task['vlm_config']:
                request_body['max_tokens'] = task['vlm_config']['max_tokens']

            # Create JSONL entry
            jsonl_entry = {
                "custom_id": f"{task['paper_id']}_{os.path.basename(task['image_path']).replace('.', '_')}_{task['target_param']}_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body
            }

            jsonl_lines.append(json.dumps(jsonl_entry, ensure_ascii=False))

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(jsonl_lines))

        print(f"Prepared batch input file with {len(jsonl_lines)} requests: {output_path}")
        return output_path

    def process_with_batch_api(self, tasks: List[Dict[str, Any]],
                               output_dir: str = "batch_results") -> List[Dict[str, Any]]:
        """Process tasks using AliYun Batch API"""
        print(f"Processing {len(tasks)} images using AliYun Batch API...")

        if not tasks:
            print("No tasks to process")
            return []

        # Prepare model config for batch processing
        model_config = tasks[0]['vlm_config'] if tasks else {}

        # Initialize client if not already done
        if not self.vlm_processor.client:
            self.vlm_processor.init_aliyun_client(model_config['name'])

        if not self.vlm_processor.client:
            print("Batch API not available, falling back to real-time")
            return self.process_with_fallback(tasks, max_workers=5)

        # Prepare batch input file
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        input_file_path = os.path.join(output_dir, f"batch_input_{timestamp}.jsonl")

        self.prepare_batch_input_file(tasks, input_file_path)

        # Check if file was created and has content
        if not os.path.exists(input_file_path) or os.path.getsize(input_file_path) == 0:
            print("No valid images to process in batch")
            return []

        try:
            # Upload file
            print(f"Uploading input file: {input_file_path}")
            file_object = self.vlm_processor.client.files.create(
                file=Path(input_file_path),
                purpose="batch"
            )
            file_id = file_object.id
            print(f"File uploaded successfully. File ID: {file_id}")

            # Create batch job
            print(f"Creating batch job")
            batch = self.vlm_processor.client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            batch_id = batch.id

            print(f"Batch job created successfully. Batch ID: {batch_id}")

            # Wait for completion
            status = ""
            while status not in ["completed", "failed", "expired", "cancelled"]:
                try:
                    batch_info = self.vlm_processor.client.batches.retrieve(batch_id=batch_id)
                    status = batch_info.status
                    print(f"Batch status: {status}")

                    if status in ["completed", "failed", "expired", "cancelled"]:
                        break

                    print(f"Waiting for batch completion... next check in 30 seconds")
                    time.sleep(30)

                except Exception as e:
                    print(f"Error checking batch status: {e}")
                    time.sleep(30)

            if status != "completed":
                print(f"Batch job failed with status: {status}")
                if hasattr(batch_info, 'errors') and batch_info.errors:
                    print(f"Errors: {batch_info.errors}")
                return []

            # Download results
            results = []
            if batch_info.output_file_id:
                output_path = os.path.join(output_dir, f"batch_output_{timestamp}.jsonl")

                content = self.vlm_processor.client.files.content(batch_info.output_file_id)
                content.write_to_file(output_path)

                # Parse results
                with open(output_path, 'r', encoding='utf-8') as f:
                    batch_results = [json.loads(line.strip()) for line in f if line.strip()]

                for result in batch_results:
                    try:
                        # Extract response content
                        response_content = None
                        if 'response' in result and 'body' in result['response']:
                            body = result['response']['body']
                            if 'choices' in body and body['choices']:
                                response_content = body['choices'][0].get('message', {}).get('content', '')

                        if response_content:
                            parsed_data = self.vlm_processor.parse_vlm_response(response_content)

                            # Extract metadata from custom_id
                            custom_id = result.get('custom_id', '')
                            custom_parts = custom_id.split('_')

                            if len(custom_parts) >= 4:
                                parsed_data['paper_id'] = custom_parts[0]
                                parsed_data['image_file'] = custom_parts[1]
                                parsed_data['parameter_target'] = custom_parts[2]

                            parsed_data['extraction_timestamp'] = datetime.now().isoformat()
                            parsed_data['batch_id'] = batch_id
                            results.append(parsed_data)

                    except Exception as e:
                        print(f"Error processing batch result: {e}")
                        continue

            print(f"Batch processing complete. Processed {len(results)} images successfully.")
            return results

        except Exception as e:
            print(f"Failed to create batch job: {e}")
            return []

    def process_single_image_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single image task"""
        img_path = task['image_path']
        target_param = task['target_param']
        prompt = task['prompt']
        vlm_config = task['vlm_config']
        pdf_dir = task['pdf_dir']

        img_filename = os.path.basename(img_path)
        images_dir = os.path.join(pdf_dir, "images")

        # Check if analysis already exists
        analysis_text_file = os.path.splitext(img_filename)[0] + "_vlm_analysis.txt"
        analysis_file_path = os.path.join(images_dir, analysis_text_file)

        if os.path.exists(analysis_file_path) and not self.replace:
            try:
                with open(analysis_file_path, 'r', encoding='utf-8') as f:
                    response = f.read()
                # Parse the existing response and return it
                parsed_data = self.vlm_processor.parse_vlm_response(response)
                if parsed_data:
                    parsed_data['image_file'] = img_filename
                    parsed_data['parameter_target'] = target_param
                    parsed_data['extraction_timestamp'] = datetime.now().isoformat()
                    parsed_data['paper_id'] = os.path.basename(pdf_dir)
                    return parsed_data
            except Exception as e:
                print(f"Error reading existing analysis {analysis_file_path}: {e}")
                # If there's an error reading, reprocess it

        # Process with API
        response = self.vlm_processor.call_vlm_api(img_path, prompt, vlm_config)
        if response:
            self.vlm_processor.save_vlm_analysis(images_dir, img_filename, response)
            time.sleep(0.5)  # Small delay to avoid rate limiting

        if response:
            parsed_data = self.vlm_processor.parse_vlm_response(response)
            if parsed_data:
                parsed_data['image_file'] = img_filename
                parsed_data['parameter_target'] = target_param
                parsed_data['extraction_timestamp'] = datetime.now().isoformat()
                parsed_data['paper_id'] = os.path.basename(pdf_dir)
                return parsed_data

        return None

    def process_with_fallback(self, tasks: List[Dict[str, Any]], max_workers: int = 5) -> List[Dict[str, Any]]:
        """Fallback to real-time processing if batch API is not available"""
        print(f"Falling back to real-time processing for {len(tasks)} images...")

        results = []
        failed_tasks = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            for task in tasks:
                future = executor.submit(self.process_single_image_task, task)
                future_to_task[future] = task

            # Process with progress bar
            with tqdm(total=len(tasks), desc="Processing images") as pbar:
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        print(f"Task failed: {task['image_path']} - Error: {e}")
                        failed_tasks.append(task)
                    finally:
                        pbar.update(1)

        print(f"\nCompleted: {len(results)} successful, {len(failed_tasks)} failed")
        return results

    def process_papers_in_parallel(self, base_dir: str, vlm_model_choice: str = None,
                                   target_parameters: List[str] = None, paper_list: List[str] = None,
                                   output_dir: str = "vlm_results", max_workers: int = 5,
                                   use_batch: bool = True) -> List[Dict[str, Any]]:
        """Batch process multiple papers with VLM in parallel"""
        if target_parameters is None:
            target_parameters = ["28d_compressive_strength"]

        try:
            # Initialize AliYun client for batch mode
            if use_batch:
                batch_supported, vlm_config = self.vlm_processor.init_aliyun_client(vlm_model_choice)
                self.batch_mode = batch_supported
            else:
                vlm_config = self.vlm_processor.setup_vlm_config(vlm_model_choice)
                self.batch_mode = False

        except ValueError as e:
            print(f"VLM configuration error: {e}")
            return []

        print(f"Starting VLM processing:")
        print(f"  VLM Model: {vlm_config['name']}")
        print(f"  Target parameters: {target_parameters}")
        print(f"  Processing mode: {'Batch' if self.batch_mode else 'Real-time'}")
        if not self.batch_mode:
            print(f"  Max workers: {max_workers}")

        # Collect paper directories
        pdf_dirs = []
        if paper_list:
            for paper_id in paper_list:
                paper_dir = Path(os.path.join(base_dir, str(paper_id) + '.pdf-id'))
                if os.path.exists(paper_dir):
                    pdf_dirs.append(str(paper_dir))

        if not pdf_dirs:
            print(f"No PDF directories found in {base_dir}")
            return []

        print(f"Found {len(pdf_dirs)} papers to process")

        # Prepare all tasks
        print("Preparing image processing tasks...")
        tasks = self.prepare_image_tasks(pdf_dirs, target_parameters, vlm_config)
        print(f"Total images to process: {len(tasks)}")

        if not tasks:
            print("No images found to process")
            return []

        # Process based on mode
        if self.batch_mode and use_batch:
            print("Using AliYun Batch API for processing...")
            all_results = self.process_with_batch_api(tasks, output_dir)
        else:
            print(f"Using real-time processing with {max_workers} workers...")
            all_results = self.process_with_fallback(tasks, max_workers)

        # Save results
        self.save_results(all_results, output_dir, vlm_config, target_parameters)

        return all_results

    def save_results(self, all_results, output_dir, vlm_config, target_parameters):
        """Save results to organized files"""
        os.makedirs(output_dir, exist_ok=True)

        # Group results by paper
        results_by_paper = {}
        for result in all_results:
            paper_id = result.get('paper_id', 'unknown')
            if paper_id not in results_by_paper:
                results_by_paper[paper_id] = []
            results_by_paper[paper_id].append(result)

        # Save individual paper results
        for paper_id, paper_results in results_by_paper.items():
            result_data = {
                'paper_id': paper_id,
                'vlm_model': vlm_config['name'],
                'target_parameters': target_parameters,
                'total_images_processed': len(paper_results),
                'results': paper_results,
                'processing_timestamp': datetime.now().isoformat(),
                'processing_mode': 'batch' if self.batch_mode else 'parallel',
                'max_workers': 0 if self.batch_mode else 5
            }

            output_path = os.path.join(output_dir, f"{paper_id}_vlm_results.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            print(f"  ✓ Saved VLM results: {output_path} ({len(paper_results)} findings)")

        # Save summary
        summary_data = {
            'total_papers_processed': len(results_by_paper),
            'total_images_processed': len(all_results),
            'vlm_model': vlm_config['name'],
            'target_parameters': target_parameters,
            'processing_mode': 'batch' if self.batch_mode else 'parallel',
            'processing_timestamp': datetime.now().isoformat(),
            'results_by_paper': {k: len(v) for k, v in results_by_paper.items()}
        }

        summary_path = os.path.join(output_dir, "processing_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"\n=== VLM PROCESSING COMPLETE ===")
        print(f"Total papers processed: {len(results_by_paper)}")
        print(f"Total VLM findings: {len(all_results)}")
        print(f"Processing mode: {'Batch' if self.batch_mode else 'Real-time'}")
        print(f"Cost savings: {'50%' if self.batch_mode else '0%'}")
        print(f"Summary saved to: {summary_path}")



if __name__ == "__main__":
    main()