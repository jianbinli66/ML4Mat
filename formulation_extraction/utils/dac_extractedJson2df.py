"""
DAC JSON to DataFrame Converter - UPDATED
Converts extracted JSON files from DAC papers to structured DataFrame format.
Updated to handle the specific JSON structure from the extraction.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import re
from typing import Dict, List, Any, Optional, Tuple


def safe_float_conversion(value, extract_unit=False):
    """Safely convert value to float, handling various edge cases"""
    if value is None:
        return (None, None) if extract_unit else None

    if isinstance(value, (int, float, np.integer, np.floating)):
        return (float(value), None) if extract_unit else float(value)

    if isinstance(value, str):
        value = value.strip()
        if value == "-" or value == "" or value.lower() == "null":
            return (None, None) if extract_unit else None

        # Extract unit if requested
        unit = None
        if extract_unit:
            # Find unit in the string (letters after the number)
            match = re.search(r'[-+]?\d*\.?\d+\s*([a-zA-Z/%°]+)', value)
            if match:
                unit = match.group(1).strip()

        # Handle special cases like "~158 min" -> extract 158
        clean_value = value.replace("~", "").replace(">", "").replace("<", "")

        # Extract the first number found
        match = re.search(r'[-+]?\d*\.?\d+', clean_value)
        if match:
            try:
                num_val = float(match.group())
                return (num_val, unit) if extract_unit else num_val
            except ValueError:
                return (None, None) if extract_unit else None
        return (None, None) if extract_unit else None

    return (None, None) if extract_unit else None


def extract_all_kinetic_data(kinetic_data: Dict) -> Dict:
    """Extract all kinetic data into a structured dictionary"""
    kinetic_results = {}

    if not kinetic_data:
        return kinetic_results

    try:
        # Process all kinetic data types
        for kinetic_type, kinetic_content in kinetic_data.items():
            if not kinetic_content:
                continue

            # Process each data series within the kinetic type
            for series_name, series_data in kinetic_content.items():
                if not isinstance(series_data, dict):
                    continue

                # Process each label within the series data
                for label, label_data in series_data.items():
                    if not isinstance(label_data, dict):
                        continue

                    # Store the data
                    kinetic_results = {
                        "Kinetic_name": kinetic_type,
                        "Kinetic_label": label,
                        "Kinetic_values": str(label_data.get("Values", [])),
                        "Kinetic_units": label_data.get("Unit", ""),
                        "Kinetic_percent_saturation": str(label_data.get("Percent_Saturation", [])),
                    }
                    # Only keep the first kinetic data found
                    return kinetic_results

    except Exception as e:
        print(f"  Warning: Error extracting kinetic data: {e}")

    return kinetic_results


def extract_profile_data(kinetic_data: Dict, profile_type: str) -> List[str]:
    """Extract profile data from kinetic section"""
    profile_list = []

    if not kinetic_data:
        return profile_list

    try:
        # Handle different profile types
        if profile_type == "TG Curve/ CO2 adsorption profile":
            for key in ["Weight percentage", "CO₂ uptake (mmol/g)", "CO₂ Uptake (wt.%)", "Weight percentage (wt%)"]:
                if key in kinetic_data:
                    data_item = kinetic_data[key]
                    if isinstance(data_item, dict) and "Values" in data_item:
                        data_points = data_item["Values"]
                        for point in data_points:
                            if len(point) >= 2:
                                profile_list.append(f"[{point[0]},{point[1]}]")
                    break

        elif profile_type == "Cyclic CO2 profile":
            for key in ["CO₂ uptake (mmol/g)", "Adsorption Amount", "CO2 Uptake (wt %)", "CO₂ Uptake (wt %)"]:
                if key in kinetic_data:
                    data_item = kinetic_data[key]
                    if isinstance(data_item, dict) and "Values" in data_item:
                        data_points = data_item["Values"]
                        for point in data_points:
                            if len(point) >= 2:
                                profile_list.append(f"[{point[0]},{point[1]}]")
                    break
    except Exception as e:
        print(f"  Warning: Error extracting profile data: {e}")

    return profile_list


def calculate_saturation_times(values: List[List], unit: str = "min") -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate time to half saturation and 90% saturation from BTC/TGA data
    using linear interpolation formula
    """
    if not values:
        return None, None

    # Safely convert all values to floats
    cleaned_values = []
    for point in values:
        if len(point) >= 2:
            time_val = safe_float_conversion(point[0])
            measure_val = safe_float_conversion(point[1])
            if time_val is not None and measure_val is not None:
                cleaned_values.append([time_val, measure_val])

    if not cleaned_values:
        return None, None

    # Convert to numpy arrays for easier manipulation
    try:
        times = np.array([point[0] for point in cleaned_values])
        measurements = np.array([point[1] for point in cleaned_values])

        # Normalize measurements to 0-1 range
        min_val = np.min(measurements)
        max_val = np.max(measurements)

        if max_val == min_val:  # No variation in data
            return None, None

        normalized = (measurements - min_val) / (max_val - min_val)

        # Find times for half (0.5) and 90% (0.9) saturation
        half_time = None
        ninety_time = None

        # Use linear interpolation to find times
        for i in range(len(times) - 1):
            t_i, t_next = times[i], times[i + 1]
            f_i, f_next = normalized[i], normalized[i + 1]

            # Check for half saturation (0.5)
            if half_time is None and f_i <= 0.5 <= f_next:
                if f_next != f_i:  # Avoid division by zero
                    half_time = t_i + (0.5 - f_i) / (f_next - f_i) * (t_next - t_i)

            # Check for 90% saturation (0.9)
            if ninety_time is None and f_i <= 0.9 <= f_next:
                if f_next != f_i:  # Avoid division by zero
                    ninety_time = t_i + (0.9 - f_i) / (f_next - f_i) * (t_next - t_i)
        return half_time, ninety_time

    except Exception as e:
        print(f"  Warning: Error calculating saturation times: {e}")
        return None, None


def extract_time_with_unit(time_str: str) -> Tuple[Optional[float], Optional[str]]:
    """Extract time value and unit from a string like '~158 min' or '>158 min'"""
    if not time_str or time_str == "-":
        return None, None

    # Extract value and unit
    clean_str = str(time_str).strip()

    # Remove comparison symbols
    clean_str = clean_str.replace("~", "").replace(">", "").replace("<", "").strip()

    # Split by space to separate number and unit
    parts = clean_str.split()
    if not parts:
        return None, None

    # Try to extract number from first part
    match = re.search(r'[-+]?\d*\.?\d+', parts[0])
    if match:
        try:
            time_val = float(match.group())
            unit = parts[1] if len(parts) > 1 else ""  # Default to min if no unit specified
            return time_val, unit
        except ValueError:
            return None, None

    return None, None


def process_json_file(json_path: Path) -> List[Dict]:
    """Process a single JSON file and return list of records - UPDATED"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  Error: Invalid JSON format in {json_path.name}: {e}")
        return []
    except Exception as e:
        print(f"  Error reading {json_path.name}: {e}")
        return []

    records = []

    # Get to the parsed data - Based on your JSON structure
    if "llm_extraction" in data and "parsed_data" in data["llm_extraction"]:
        parsed_data = data["llm_extraction"]["parsed_data"]
    else:
        parsed_data = data  # Fallback if structure is different

    # Extract metadata
    metadata = parsed_data.get("metadata", {})

    # Get extraction info if available
    extraction_info = data.get("llm_extraction", {})

    # Process each material record
    exp_records = parsed_data.get("experimental_records", {})
    print("len exp_records,", len(exp_records))
    # Check if exp_records is a list or dict
    if isinstance(exp_records, list):
        # Convert list to dict with numerical keys
        exp_records_dict = {f"material_{i}": record for i, record in enumerate(exp_records)}
        exp_records = exp_records_dict

    for material_id, material_data in exp_records.items():
        # Handle case where material_data might be a list
        if isinstance(material_data, list):
            print(f"  Warning: Material data for {material_id} is a list, skipping")
            continue

        # Get chemical properties - UPDATED based on your JSON structure
        chemical = material_data.get("Chemical", ["-"] * 12)  # Changed to 12 to match your structure

        # Ensure chemical list has at least 12 elements
        while len(chemical) < 12:
            # print(f"  Warning: Material data chemical for {material_id} is less 12, skipping")
            # continue
            chemical.append("-")

        # Extract chemical properties based on JSON structure
        support = chemical[0] if chemical[0] != "-" else None
        support_type = chemical[1] if len(chemical) > 1 and chemical[1] != "-" else None
        amine_1 = chemical[2] if len(chemical) > 2 and chemical[2] != "-" else None
        amine_type = chemical[3] if len(chemical) > 3 and chemical[3] != "-" else None

        # MW/Mn could be a string like "g/mol not specified" or a number
        mw_mn = chemical[4] if len(chemical) > 4 and chemical[4] != "-" else None
        if mw_mn and isinstance(mw_mn, str) and "not specified" in mw_mn.lower():
            mw_mn = None

        # Organic content is chemical[6] (index 6, the 7th element)
        organic_content_str = chemical[6] if len(chemical) > 6 and chemical[6] != "-" else None
        organic_content = safe_float_conversion(organic_content_str) if organic_content_str else None

        # N content is chemical[7] (index 7)
        n_content_str = chemical[7] if len(chemical) > 7 and chemical[7] != "-" else None
        n_content = safe_float_conversion(n_content_str) if n_content_str else None

        # Extract additives from chemical[8] (index 8 contains additives list)
        additives = []
        if len(chemical) > 8 and isinstance(chemical[8], list):
            additives = chemical[8]

        # Extract additive types from chemical[9]
        additive_types = []
        if len(chemical) > 9 and isinstance(chemical[9], list):
            additive_types = chemical[9]

        # OH to N ratio is chemical[10]
        oh_to_n = chemical[10] if len(chemical) > 10 and chemical[10] != "-" else None

        # Loading is chemical[11] (index 11, the 12th element)
        loading_str = chemical[11] if len(chemical) > 11 and chemical[11] != "-" else None
        loading = safe_float_conversion(loading_str) if loading_str else None

        # Process organic_content
        if organic_content is None:
            organic_content = loading
        # still got none, then use ?% from record id
        if organic_content is None:
            matches = re.findall(r'(\d+(?:\.\d+)?)%', material_id)
            if matches:
                # print("Values found:", matches)
                # Get the first occurrence
                organic_content = matches[0]
                # print(f"First value before %: {organic_content}")
        # if still being none, use N_contetn
        if organic_content is None:
            organic_content = n_content
            n_content = None
        # Get textural properties
        textural = material_data.get("Textural", [[], [], [], [], [], []])

        # Process bare textural properties
        bare_sa_list = textural[0] if textural and len(textural) > 0 and textural[0] is not None else []
        bare_pv_list = textural[1] if textural and len(textural) > 1 and textural[1] is not None else []
        bare_pd_list = textural[2] if textural and len(textural) > 2 and textural[2] is not None else []

        # Process after-impregnation textural properties
        sa_list = textural[3] if textural and len(textural) > 3 and textural[3] is not None else []
        pv_list = textural[4] if textural and len(textural) > 4 and textural[4] is not None else []
        pd_list = textural[5] if textural and len(textural) > 5 and textural[5] is not None else []

        # Helper function to safely calculate average
        def safe_sum(value_list):
            if not value_list:
                return None

            valid_values = []
            for val in value_list:
                if isinstance(val, list):
                    # Handle nested lists - take first value
                    if val:
                        f_val = safe_float_conversion(val[0])
                        if f_val is not None:
                            valid_values.append(f_val)
                else:
                    f_val = safe_float_conversion(val)
                    if f_val is not None:
                        valid_values.append(f_val)

            return np.sum(valid_values) if valid_values else None

        # Calculate averages
        bare_sa_avg = safe_sum(bare_sa_list)
        bare_pv_avg = safe_sum(bare_pv_list)
        bare_pd_avg = safe_sum(bare_pd_list)
        sa_avg = safe_sum(sa_list)
        pv_avg = safe_sum(pv_list)
        pd_avg = safe_sum(pd_list)

        # Process each experimental test for this material
        for key in material_data.keys():
            if key in ["Chemical", "Textural"]:
                continue

            test_name = key
            test_data = material_data[key]

            # Create base record with UPDATED structure matching your column list
            record = {
                "Record_ID": f"{material_id}_{test_name}",
                "Material_ID": material_id,
                "Support": support,
                "Support_Type": support_type,
                "Amine_1_or_Additive_1": amine_1,
                "Amine_Type": amine_type,
                "MW_Mn_g_mol": safe_float_conversion(mw_mn) if mw_mn and mw_mn != "-" else None,
                # "Loading_pct": loading,
                "Organic_Content_pct": organic_content,
                "N_Content_mmol_g": n_content,
                # Add additives - handle up to 2 additives based on your column list
                "Amine_2_or_Additive_2": additives[0] if len(additives) > 0 else None,
                "Amine_2_or_Additive_2_Type": additive_types[0] if len(additive_types) > 0 else None,
                "Amine_3_or_Additive_3": additives[1] if len(additives) > 1 else None,
                "Amine_3_or_Additive_3_Type": additive_types[1] if len(additive_types) > 1 else None,
                "BET_Bare_Surface_Area_m2_g": bare_sa_avg,
                "Bare_Pore_Volume_cm3_g": bare_pv_avg,
                "Average_Bare_Pore_Diameter_nm": bare_pd_avg,
                "BET_Surface_Area_m2_g": sa_avg,
                "Pore_Volume_cm3_g": pv_avg,
                "Average_Pore_Diameter_nm": pd_avg,
                "DOI": metadata.get("DOI", "-"),
                "Model_Name": extraction_info.get("model_used", "-"),
                "Prompt_Name": extraction_info.get("prompt_used", "-"),
                "Source_File": str(json_path.name)
            }

            # Process Operational Parameters
            if "Operational_Parameters" in test_data:
                op_params = test_data["Operational_Parameters"]
                if isinstance(op_params, list):
                    # Extract operational parameters
                    if len(op_params) > 0:
                        record["Relative_Humidity_pct"] = safe_float_conversion(op_params[0]) if op_params[0] not in ["-", ""] else None
                    if len(op_params) > 1:
                        record["CO2_Concentration"] = op_params[1] if op_params[1] not in ["-", ""] else None
                    if len(op_params) > 2:
                        record["CO2_Concentration_vol_pct"] = safe_float_conversion(op_params[2]) if op_params[2] not in ["-", ""] else None
                    if len(op_params) > 3:
                        record["Flow_Rate"] = op_params[3] if op_params[3] not in ["-", ""] else None
                    if len(op_params) > 4:
                        record["Flow_Rate_mL_min"] = safe_float_conversion(op_params[4]) if op_params[4] not in ["-", ""] else None
                    if len(op_params) > 5:
                        # Interfering gases might be a list or string
                        if isinstance(op_params[5], list) and op_params[5]:
                            record["Interfering_Gases"] = ", ".join(str(g) for g in op_params[5])
                        else:
                            record["Interfering_Gases"] = op_params[5] if op_params[5] not in ["-", ""] else None
                    if len(op_params) > 6:
                        record["Adsorption_Temperature_C"] = safe_float_conversion(op_params[6]) if op_params[6] not in ["-", ""] else None
                    if len(op_params) > 7:
                        record["CO2_Test_Method"] = op_params[7] if op_params[7] not in ["-", ""] else None
                    if len(op_params) > 8:
                        record["Desorption_Temperature_C"] = safe_float_conversion(op_params[8]) if op_params[8] not in ["-", ""] else None
                    if len(op_params) > 9:
                        record["Desorption_Gas"] = op_params[9] if op_params[9] not in ["-", ""] else None
                    if len(op_params) > 10:
                        des_time = op_params[10]
                        if des_time not in ["-", ""]:
                            des_time_val, des_time_unit = extract_time_with_unit(str(des_time))
                            if des_time_val is not None:
                                record["Desorption_Time"] = f"{des_time_val} {des_time_unit}" if des_time_unit else f"{des_time_val}"
                            else:
                                record["Desorption_Time"] = str(des_time)

            # Process Performance Indicators
            if "Performance_Indicators" in test_data:
                perf_data = test_data["Performance_Indicators"]
                if isinstance(perf_data, list):
                    # Extract performance data
                    if len(perf_data) > 0 and perf_data[0] not in ["-", ""]:
                        record["CO2_Capacity"] = str(perf_data[0])
                    if len(perf_data) > 1 and perf_data[1] not in ["-", ""]:
                        record["CO2_Capacity_mmol_g"] = safe_float_conversion(perf_data[1])
                    if len(perf_data) > 2 and perf_data[2] not in ["-", ""]:
                        record["Amine_Efficiency"] = str(perf_data[2])
                    if len(perf_data) > 3 and perf_data[3] not in ["-", ""]:
                        record["Amine_Efficiency_mmol_mmol"] = safe_float_conversion(perf_data[3])
                    if len(perf_data) > 4 and perf_data[4] not in ["-", ""]:
                        record["Weight_Loss_Stability_pct"] = safe_float_conversion(perf_data[4])
                    if len(perf_data) > 5 and perf_data[5] not in ["-", ""]:
                        record["Capacity_Loss_Stability_pct"] = safe_float_conversion(perf_data[5])
                    if len(perf_data) > 6 and perf_data[6] not in ["-", ""]:
                        record["Heat_of_Adsorption_kJ_mol"] = safe_float_conversion(perf_data[6])

            # Process Kinetic data
            half_saturation = None
            half_saturation_unit = ""
            ninety_saturation = None
            ninety_saturation_unit = ""

            # Initialize kinetic data lists
            kinetic_names = []
            kinetic_labels = []
            kinetic_values_list = []
            kinetic_units_list = []
            kinetic_percent_saturation_list = []

            if "Kinetic" in test_data:
                kinetic = test_data["Kinetic"]

                # Extract ALL kinetic data
                for kin_name, kin_content in kinetic.items():
                    if not kin_content:
                        continue

                    if isinstance(kin_content, dict):
                        for kin_label, kin_data in kin_content.items():
                            if isinstance(kin_data, dict):
                                # Store kinetic data
                                kinetic_names.append(kin_name)
                                kinetic_labels.append(kin_label)
                                kinetic_values_list.append(str(kin_data.get("Values", [])))
                                kinetic_units_list.append(kin_data.get("Unit", ""))
                                kinetic_percent_saturation_list.append(str(kin_data.get("Percent_Saturation", [])))

                                # Extract saturation times
                                ps = kin_data.get("Percent_Saturation", [])
                                if ps and len(ps) >= 2:
                                    if half_saturation is None and ps[0] != "-":
                                        half_val, half_unit = extract_time_with_unit(str(ps[0]))
                                        if half_val is not None:
                                            half_saturation = half_val
                                            half_saturation_unit = half_unit

                                    if ninety_saturation is None and len(ps) > 1 and ps[1] != "-":
                                        ninety_val, ninety_unit = extract_time_with_unit(str(ps[1]))
                                        if ninety_val is not None:
                                            ninety_saturation = ninety_val
                                            ninety_saturation_unit = ninety_unit

                                # Calculate from Values if needed
                                if (half_saturation is None or ninety_saturation is None) and "Values" in kin_data:
                                    values = kin_data["Values"]
                                    unit_info = kin_data.get("Unit", "")
                                    time_unit = unit_info.split(",")[0].strip() if "," in unit_info else ""

                                    calc_half, calc_ninety = calculate_saturation_times(values, time_unit)
                                    half_saturation = half_saturation or calc_half
                                    ninety_saturation = ninety_saturation or calc_ninety

            # Combine kinetic data into strings
            kinetic_name_str = "; ".join(kinetic_names) if kinetic_names else ""
            kinetic_label_str = "; ".join(kinetic_labels) if kinetic_labels else ""
            kinetic_values_str = "; ".join(kinetic_values_list) if kinetic_values_list else ""
            kinetic_units_str = "; ".join(kinetic_units_list) if kinetic_units_list else ""
            kinetic_percent_saturation_str = "; ".join(kinetic_percent_saturation_list) if kinetic_percent_saturation_list else ""

            # Format saturation times
            half_saturation_str = f"{half_saturation}" if half_saturation is not None else None
            ninety_saturation_str = f"{ninety_saturation}" if ninety_saturation is not None else None

            # Add kinetic columns
            record.update({
                "Kinetic_name": kinetic_name_str,
                "Kinetic_label": kinetic_label_str,
                "Kinetic_values": kinetic_values_str,
                "Kinetic_units": kinetic_units_str,
                "Kinetic_percent_saturation": kinetic_percent_saturation_str,
                "Time_to_Half_Saturation": half_saturation_str,
                "Time_to_90_Saturation": ninety_saturation_str,
            })

            # Process cycle data
            cycle_data_list = []

            if "Cycles" in test_data and test_data["Cycles"]:
                cycles = test_data["Cycles"]

                for cycle_idx, (cycle_name_key, cycle_content) in enumerate(cycles.items(), 1):
                    if not cycle_content:
                        continue

                    cycle_info = {
                        "name": cycle_name_key,
                        "label": "",
                        "values": "",
                        "units": "",
                        "max_cycle": None
                    }

                    if isinstance(cycle_content, dict):
                        for cycle_label_key, label_data in cycle_content.items():
                            if isinstance(label_data, dict):
                                # Store cycle data
                                cycle_info["label"] = cycle_label_key
                                cycle_info["values"] = str(label_data.get("Values", []))
                                cycle_info["units"] = label_data.get("Unit", "")

                                # Find max cycle number
                                values = label_data.get("Values", [])
                                if values:
                                    cycle_numbers = []
                                    for point in values:
                                        if len(point) >= 1:
                                            cycle_num = safe_float_conversion(point[0])
                                            if cycle_num is not None:
                                                cycle_numbers.append(cycle_num)
                                    if cycle_numbers:
                                        cycle_info["max_cycle"] = max(cycle_numbers)

                                break  # Only take first label

                    cycle_data_list.append(cycle_info)

                    # Only keep up to 2 cycles as per your column list
                    if cycle_idx >= 2:
                        break

            # Add cycle columns
            for i in range(1, 3):
                cycle_key = f"Cycle_{i}"
                if i <= len(cycle_data_list):
                    cycle_info = cycle_data_list[i-1]
                    record.update({
                        f"{cycle_key}_name": cycle_info["name"],
                        f"{cycle_key}_label": cycle_info["label"],
                        f"{cycle_key}_values": cycle_info["values"],
                        f"{cycle_key}_units": cycle_info["units"],
                        f"{cycle_key}_max_cycle": cycle_info["max_cycle"],
                    })
                else:
                    record.update({
                        f"{cycle_key}_name": None,
                        f"{cycle_key}_label": None,
                        f"{cycle_key}_values": None,
                        f"{cycle_key}_units": None,
                        f"{cycle_key}_max_cycle": None,
                    })

            records.append(record)

    return records


def find_json_files(input_path: str) -> List[Path]:
    """Find all JSON files in the input directory"""
    input_dir = Path(input_path)
    json_files = []

    if input_dir.is_file() and input_dir.suffix == '.json':
        # Single file
        return [input_dir]
    elif input_dir.is_dir():
        # Directory - find all JSON files
        json_files = list(input_dir.glob("*.json"))
    else:
        print(f"Error: Input path '{input_path}' does not exist or is not a valid file/directory")
        return []

    return json_files


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