#!/usr/bin/env python3
# dac_json2csv_run.py
import argparse
import os
import re
from pathlib import Path

import pandas as pd

from utils.dac_extractedJson2df import process_json_file, find_json_files


def main():
    parser = argparse.ArgumentParser(description='Convert DAC JSON extraction files to DataFrame - UPDATED')
    parser.add_argument('--input', type=str,
                        default='json_results/DAC_sample_papers/DAC_prompt_SCaSE_qwen3-max_with_vlm',
                        help='Input path (directory with JSON files or single JSON file)')
    parser.add_argument('--output', type=str, default='data/DAC_sample_papers_extract/dac_data_exact.csv',
                        help='Output CSV file path')

    args = parser.parse_args()

    # Find JSON files
    json_files = find_json_files(args.input)

    if not json_files:
        print(f"No JSON files found in {args.input}")
        return

    print(f"Found {len(json_files)} JSON file(s) to process")

    all_records = []

    for json_file in json_files:
        print(f"Processing {json_file}...")
        records = process_json_file(json_file)
        all_records.extend(records)
        print(f"  Extracted {len(records)} records")

    # Create DataFrame
    if all_records:
        df = pd.DataFrame(all_records)

        # Define column order based on your requirements
        column_order = [
            "Record_ID", "Material_ID", "DOI", "Source_File", "Support",
            "Support_Type", "Amine_1_or_Additive_1", "Amine_Type", "MW_Mn_g_mol",
            "Organic_Content_pct", "N_Content_mmol_g",
            "Amine_2_or_Additive_2", "Amine_2_or_Additive_2_Type",
            "Amine_3_or_Additive_3", "Amine_3_or_Additive_3_Type",
            "BET_Bare_Surface_Area_m2_g", "Bare_Pore_Volume_cm3_g",
            "Average_Bare_Pore_Diameter_nm", "BET_Surface_Area_m2_g",
            "Pore_Volume_cm3_g", "Average_Pore_Diameter_nm",
            "Relative_Humidity_pct", "CO2_Concentration", "CO2_Concentration_vol_pct",
            "Flow_Rate", "Flow_Rate_mL_min", "Interfering_Gases",
            "Adsorption_Temperature_C", "CO2_Test_Method",
            "CO2_Capacity", "CO2_Capacity_mmol_g", "Amine_Efficiency",
            "Amine_Efficiency_mmol_mmol", "Heat_of_Adsorption_kJ_mol",
            "Time_to_Half_Saturation", "Time_to_90_Saturation",
            "Kinetic_name", "Kinetic_label", "Kinetic_values", "Kinetic_units",
            "Kinetic_percent_saturation", "Desorption_Gas",
            "Desorption_Temperature_C", "Desorption_Time",
            "Weight_Loss_Stability_pct", "Capacity_Loss_Stability_pct",
            "Cycle_1_name", "Cycle_1_label", "Cycle_1_values", "Cycle_1_units",
            "Cycle_1_max_cycle", "Cycle_2_name", "Cycle_2_label",
            "Cycle_2_values", "Cycle_2_units", "Cycle_2_max_cycle",
            # Additional columns from extraction info
            "Model_Name", "Prompt_Name"
        ]

        # Add any missing columns
        for col in column_order:
            if col not in df.columns:
                df[col] = None

        # Reorder columns
        df = df[column_order]

        # Create output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"\nData saved to {output_path}")
        print(f"Total records: {len(df)}")
        print(f"Total columns: {len(df.columns)}")

        # Display sample
        print("\nFirst 3 records:")
        print(df.head(3).to_string())

        # Summary statistics
        print("\nSummary of extracted data:")
        print(f"Number of unique materials: {df['Material_ID'].nunique()}")
        print(f"Number of unique supports: {df['Support'].nunique()}")

        # Count records with CO2 capacity data
        capacity_data = df[df['CO2_Capacity_mmol_g'].notna()]
        print(f"Records with CO2 capacity data: {len(capacity_data)}")

        # Display unique test methods
        test_methods = df['CO2_Test_Method'].unique()
        print(f"Test methods found: {test_methods}")

    else:
        print("No records extracted.")


if __name__ == "__main__":
    main()