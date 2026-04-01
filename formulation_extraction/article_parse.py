import os
import json
import pandas as pd
from pathlib import Path

def parse_articles():
    # Define source and destination directories
    source_dir = "json_results/DAC_sample_papers/DAC_prompt_SCaSE_qwen3-max_with_vlm"
    articles_csv_path = "data/articles.csv"
    experiments_dir = "data/experiments"
    
    # Create experiments directory if it doesn't exist
    Path(experiments_dir).mkdir(parents=True, exist_ok=True)
    
    # Lists to store article and experiment data
    articles_data = []
    experiment_files_created = []
    
    # Process each JSON file in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('_comprehensive_extraction.json'):
            file_path = os.path.join(source_dir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract metadata
                pdf_id = data.get('pdf_id', '').replace('.pdf-id', '')
                metadata = data.get('llm_extraction', {}).get('parsed_data', {}).get('metadata', {})
                
                # Extract required fields
                doi = metadata.get('DOI', '')
                keywords = metadata.get('Keywords', [])
                abstract_summary = metadata.get('summary', '')
                
                # Create article record
                article_record = {
                    'id': pdf_id,
                    'DOI': doi,
                    'Keywords': ', '.join(keywords) if isinstance(keywords, list) else str(keywords),
                    'Abstract_Summary': abstract_summary
                }
                
                articles_data.append(article_record)
                
                # Save experimental data to individual JSON file
                experimental_records = data.get('llm_extraction', {}).get('parsed_data', {}).get('experimental_records', {})
                
                experiment_file_path = os.path.join(experiments_dir, f"{pdf_id}.json")
                with open(experiment_file_path, 'w', encoding='utf-8') as exp_file:
                    json.dump(experimental_records, exp_file, indent=2, ensure_ascii=False)
                
                experiment_files_created.append(experiment_file_path)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(articles_data)
    df.to_csv(articles_csv_path, index=False)
    
    print(f"Successfully processed {len(articles_data)} articles.")
    print(f"Articles saved to {articles_csv_path}")
    print(f"Experimental data saved to {len(experiment_files_created)} files in {experiments_dir}/ directory")
    
    return articles_data, experiment_files_created

if __name__ == "__main__":
    articles, experiments = parse_articles()