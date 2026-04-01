# LLM-based Data Extraction for Direct Air Capture (DAC) Research

A comprehensive pipeline for extracting structured experimental data from scientific PDFs using Large Language Models (LLMs) and Vision-Language Models (VLMs) for Direct Air Capture (DAC) research.

## Overview

This system automates the extraction of experimental data from scientific literature on CO₂ capture materials. It processes PDFs, classifies articles, extracts structured data using LLMs, and validates/processes the extracted data into standardized formats.

## Key Features

- **PDF Processing**: Extract text and images from scientific PDFs
- **Article Classification**: Classify papers by experimental method
- **Multi-modal Extraction**: Combine LLM (text) and VLM (image) analysis
- **Dynamic File Discovery**: Automatically discovers papers and prompts without hardcoded configs
- **Batch Processing**: Process large datasets efficiently with parallel workers
- **Structured Output**: Generate standardized CSV/JSON outputs for analysis

## Project Structure

```
llm_extract/
├── data/                          # Data directories
│   ├── {dataset_name}/           # Raw PDFs (e.g., data/DAC/001.pdf)
│   └── pdfs2textimgs/            # Processed PDFs with text/images
│       └── {dataset_name}/
│           └── {paper_id}/       # Individual paper directories
├── prompts/                       # Prompt templates (.txt files)
├── json_results/                  # LLM extraction results
├── processors/                    # Core processing modules
├── old_format/                    # Legacy code
├── config.json                    # Configuration (models, API settings)
├── .env                          # API keys (not in repo)
└── *.py                          # Main execution scripts
```

## Installation

### 1. Clone and Setup Environment

```bash
# Create and activate conda environment
conda create -n llm_extract python=3.9
conda activate llm_extract

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file with your API keys:

```bash
# OpenAI API (for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Dashscope API (for Qwen models)
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# Mineru API (for PDF processing)
MINERU_API_KEY=your_mineru_api_key_here
```

### 3. Prepare Data

Place PDF files in the appropriate directory structure:

```bash
# For a new dataset called "my_dataset"
mkdir -p data/my_dataset
cp *.pdf data/my_dataset/

# Files should be named as paper IDs (e.g., 001.pdf, 002.pdf, etc.)
```

## Configuration

The `config.json` file contains model configurations and defaults:

```json
{
  "dataset": "DAC",                    # Default dataset name
  "seed": 8888,                        # Random seed for reproducibility
  "extraction": {
    "default_model": "qwen3-max",      # Default LLM model
    "default_prompt": "detailed_extraction"  # Default prompt
  },
  "vlm_models": { ... },               # VLM model configurations
  "available_models": { ... }          # LLM model configurations
}
```

**Note**: The system now uses dynamic file discovery. No hardcoded paper lists or prompt lists are needed in the config.

## Usage

### Pipeline Overview

The complete data extraction pipeline consists of 5 main steps:

1. **PDF Processing** → 2. **Article Classification** → 3. **VLM Processing** → 4. **LLM Extraction** → 5. **Data Processing**

### 1. PDF Processing (`1_pdf_process_run.py`)

Extract text and images from PDFs:

```bash
python 1_pdf_process_run.py \
  --dataset DAC_sample_papers \
  --input_dir ./data/DAC_sample_papers \
  --output_dir ./data/pdfs2textimgs/DAC_sample_papers \
  --batch_size 10
```

**Parameters:**
- `--dataset`: Dataset name (creates subdirectory in output)
- `--input_dir`: Directory containing PDF files
- `--output_dir`: Output directory for processed PDFs
- `--batch_size`: Number of PDFs to process per batch
- `--num_papers`: Limit number of papers to process

### 2. Article Classification (`2_dac_article_classifying.py`)

Classify papers by experimental method:

```bash
python 2_dac_article_classifying.py \
  --dataset DAC \
  --max_workers 10 \
  --num_words 1000 \
  --output data/DAC_extract/classification.csv
```

**Parameters:**
- `--dataset`: Dataset name
- `--max_workers`: Number of parallel workers
- `--num_words`: Number of words to extract per paper
- `--output`: Output CSV file path
- `--num_papers`: Limit number of papers to process

### 3. VLM Processing (`2_vlm_process_run.py`)

Extract data from figures and tables using Vision-Language Models:

```bash
python 2_vlm_process_run.py \
  --dataset DAC_sample_papers \
  --vlm_model qwen3-vl-plus \
  --target_params adsorption \
  --max_workers 10 \
  --num_papers 20
```

**Parameters:**
- `--dataset`: Dataset name
- `--vlm_model`: VLM model to use (qwen-vl-plus, qwen-vl-max, qwen3-vl-plus)
- `--target_params`: Target parameters to extract
- `--max_workers`: Number of parallel workers
- `--class`: Filter by article classification
- `--num_papers`: Limit number of papers to process

### 4. LLM Extraction (`3_exp_extraction_process_run.py`)

Extract structured data using LLMs:

```bash
python 3_exp_extraction_process_run.py \
  --dataset DAC_sample_papers \
  --llm_model qwen3-max \
  --vlm_model qwen3-vl-plus \
  --prompt DAC_prompt_SCaSE \
  --target_params adsorption \
  --num_papers 20
```

**Parameters:**
- `--dataset`: Dataset name
- `--llm_model`: LLM model to use
- `--vlm_model`: VLM model to use (for combined analysis)
- `--prompt`: Prompt template name (without .txt extension)
- `--target_params`: Target parameters to extract
- `--exp_class`: Filter by experimental class
- `--num_papers`: Limit number of papers to process

### 5. JSON to CSV Conversion (`4_dac_json2csv_run.py`)

Convert JSON extraction results to CSV:

```bash
python 4_dac_json2csv_run.py \
  --input "json_results/DAC_sample_papers/DAC_prompt_SCaSE_qwen3-max_with_vlm" \
  --prompt DAC_prompt_SCaSE \
  --model qwen3-max_qwen-vl-plus \
  --output ./data/DAC_sample_papers_extract
```

**Parameters:**
- `--input`: Input directory with JSON files
- `--prompt`: Prompt name used for extraction
- `--model`: Model combination used (format: llm_model_vlm_model)
- `--output`: Output directory for CSV files

### 6. Data Processing (`5_dac_process_run.py`)

Process and clean the extracted CSV data:

```bash
python 5_dac_process_run.py \
  --input data/DAC_extract/dac_extracted_data.csv \
  --output data/DAC_extract/dac_processed_data_by_LLM.csv
```

**Parameters:**
- `--input`: Input CSV file
- `--output`: Output CSV file

## Complete Pipeline Examples

### Sample Papers Pipeline

```bash
# Complete pipeline for sample papers
python 1_pdf_process_run.py --dataset DAC_sample_papers --input_dir ./data/DAC_sample_papers --output_dir ./data/pdfs2textimgs/DAC_sample_papers --batch_size 10
python 2_dac_article_classifying.py --dataset DAC_sample_papers --max_workers 10 --num_words 500 --output data/DAC_sample_papers_extract/classification.csv --num_papers 100
python 2_vlm_process_run.py --dataset DAC_sample_papers --vlm_model qwen3-vl-plus --target_params adsorption --max_workers 10
python 3_exp_extraction_process_run.py --dataset DAC_sample_papers --llm_model qwen3-max --vlm_model qwen3-vl-plus --prompt DAC_prompt_SCaSE --target_params adsorption
python 4_dac_json2csv_run.py --input json_results/DAC_sample_papers/DAC_prompt_SCaSE_qwen3-max_with_vlm --output data/DAC_sample_papers_extract/dac_extracted_data.csv
python 5_dac_process_run.py
```

### Full Dataset Pipeline

```bash
# Complete pipeline for full dataset
python 1_pdf_process_run.py --dataset DAC --input_dir ./data/DAC --output_dir ./data/pdfs2textimgs/DAC --batch_size 10
python 2_dac_article_classifying.py --dataset DAC --max_workers 10 --num_words 1000 --output data/DAC_extract/classification.csv
python 2_vlm_process_run.py --dataset DAC --class "Physical Impregnation Method" --vlm_model qwen3-vl-plus --target_params adsorption --max_workers 10 --num_papers 20
python 3_exp_extraction_process_run.py --dataset DAC --exp_class "Physical Impregnation Method" --llm_model qwen3-max --vlm_model qwen3-vl-plus --prompt DAC_prompt_SCaSE --target_params adsorption --num_papers 20
python 4_dac_json2csv_run.py --input json_results/DAC/DAC_prompt_SCaSE_Physical_Impregnation_Method_qwen3-max_with_vlm --output data/DAC_extract/dac_extracted_data.csv
python 5_dac_process_run.py --input data/DAC_extract/dac_extracted_data.csv --output data/DAC_extract/dac_processed_data_by_LLM.csv
```

## Prompt System

### Dynamic Prompt Discovery

The system dynamically discovers prompt files from the `prompts/` directory:

- **Generic prompts**: `prompts/{prompt_name}.txt` (e.g., `prompts/detailed_extraction.txt`)
- **Dataset-specific prompts**: `prompts/{dataset}_{prompt_name}.txt` (e.g., `prompts/DAC_detailed_extraction.txt`)

### Creating Custom Prompts

1. Create a new `.txt` file in the `prompts/` directory
2. Use placeholders like `{paper_text}` and `{images_info}` for dynamic content
3. Reference the prompt by name (without `.txt` extension) in commands

Example prompt structure:
```
Extract the following information from the paper:

Paper content:
{paper_text}

Available figures/tables:
{images_info}

Please extract...
```

## Data Schema

The extracted data follows a standardized schema documented in `column_intro.md`. Key categories include:

- **Record Identification**: DOI, Model_Name, Prompt_Name, Source_File
- **Support Properties**: Support material, surface area, pore characteristics
- **Amine Properties**: Amine types, molecular weight, organic content
- **Adsorption Conditions**: Temperature, humidity, CO₂ concentration
- **Adsorption Performance**: CO₂ capacity, amine efficiency, saturation times
- **Desorption Properties**: Desorption conditions and performance
- **Stability and Cycling**: Cycle performance and degradation
- **Thermodynamic Properties**: Heat of adsorption

## Available Models

### LLM Models (text-based)
- `qwen3-max` - Alibaba Qwen 3 Max
- `qwen3-5-max` - Alibaba Qwen 3.5 Max
- `gpt-4o` - OpenAI GPT-4o
- `gpt-4-turbo` - OpenAI GPT-4 Turbo
- `gpt-5` - OpenAI GPT-5 (when available)

### VLM Models (vision-language)
- `qwen-vl-plus` - Alibaba Qwen VL Plus
- `qwen-vl-max` - Alibaba Qwen VL Max
- `qwen3-vl-plus` - Alibaba Qwen 3 VL Plus

## Dynamic File Discovery

The system automatically discovers:

1. **PDF Files**: Scans `data/{dataset}/` for `.pdf` files
2. **Processed Papers**: Scans `data/pdfs2textimgs/{dataset}/` for paper directories
3. **Prompt Files**: Checks `prompts/` directory for `.txt` files
4. **No Hardcoded Lists**: All file discovery is dynamic based on actual filesystem contents

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure `.env` file contains valid API keys
2. **PDF Processing Failures**: Check Mineru API status and PDF file permissions
3. **Memory Issues**: Reduce `--batch_size` or `--max_workers`
4. **Encoding Errors**: Ensure PDFs are not corrupted and use standard encoding

### Debug Mode

Add `--debug` flag to any script for detailed logging:

```bash
python 1_pdf_process_run.py --dataset DAC_sample_papers --debug
```