import os
import json
import time
import pandas as pd
from openai import OpenAI


class LLMHandler:
    def __init__(self, config):
        """Initialize the LLM handler with configuration"""
        self.config = config

    def call_llm_api(self, url, key, model, messages, temperature=0.1, max_retries=3):
        """Call LLM API with structured prompt and retry logic"""
        for attempt in range(max_retries):
            try:
                client = OpenAI(
                    api_key=key,
                    base_url=url,
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    seed=self.config.get("seed", 42),
                )
                return response
            except Exception as e:
                print(f"LLM API call attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"All LLM API call attempts failed for model {model}")
                    return None

    def load_prompt(self, prompt_name):
        """Load specific prompt file for LLM"""
        prompt_path = f"prompts/{prompt_name}.txt"
        print(prompt_path)

        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()

        raise FileNotFoundError(f"Prompt file {prompt_name} not found")

    def process_extraction_request(self, combined_text, model_config, prompt_name):
        """Process extraction request with LLM"""
        combined_prompt = self.load_prompt(prompt_name)

        messages = [
            {
                "role": "system",
                "content": "You are a scientific data extraction assistant. Extract paper metadata and experimental records in structured JSON. Use both the main text content and VLM image analysis results."
            },
            {
                "role": "user",
                "content": f"{combined_prompt}\n\nDocument Content:\n{combined_text}"
            }
        ]

        response = self.call_llm_api(
            model_config['url'],
            model_config['key'],
            model_config['name'],
            messages,
            model_config.get('temperature', 0.3)
        )

        if response and response.choices:
            extracted_data = response.choices[0].message.content

            usage_dict = {}
            if hasattr(response, 'usage') and response.usage:
                usage_dict = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0)
                }

            return {
                'extracted_data': extracted_data,
                'model_name': model_config['name'],
                'prompt_name': prompt_name,
                'usage': usage_dict,
                'timestamp': pd.Timestamp.now().isoformat(),
                'vlm_analysis_included': combined_text.count("VLM IMAGE ANALYSIS:") > 0
            }

        return None