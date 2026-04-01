# process_dac_data.py
"""
DAC Data Processing Module

This module processes extracted DAC (Direct Air Capture) data with various transformations and calculations.
It standardizes units, fills missing values, and performs calculations to create a consistent dataset.
"""

import pandas as pd
import numpy as np
import re
import ast
import logging

import pandas as pd
import numpy as np


def fill_pore_properties(df,
                         sa_col='BET_Surface_Area_m2_g',
                         pv_col='Pore_Volume_cm3_g',
                         pd_col='Average_Pore_Diameter_nm',
                         bare_prefix=False):
    """
    Fill missing values for BET surface area, pore volume, or pore diameter
    using the cylindrical pore approximation formula: D = 4V/S

    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe containing pore property columns
    sa_col : str
        Column name for BET Surface Area (m²/g)
    pv_col : str
        Column name for Pore Volume (cm³/g)
    pd_col : str
        Column name for Average Pore Diameter (nm)
    bare_prefix : bool
        If True, use 'Bare_' prefixed columns instead

    Returns:
    --------
    pandas DataFrame with filled values (copy of original)
    """
    df = df.copy()

    # Override column names if using bare support properties
    if bare_prefix:
        sa_col = 'BET_Bare_Surface_Area_m2_g'
        pv_col = 'Bare_Pore_Volume_cm3_g'
        pd_col = 'Average_Bare_Pore_Diameter_nm'

    for idx in df.index:
        sa = df.at[idx, sa_col]
        pv = df.at[idx, pv_col]
        pd_val = df.at[idx, pd_col]

        # Count missing values among the three properties
        missing = pd.isna([sa, pv, pd_val]).sum()

        # Only fill if exactly 1 value is missing AND 2 are available AND non-zero
        if missing == 1:
            if pd.isna(sa) and pd.notna(pv) and pd.notna(pd_val) and pd_val > 0:
                # Calculate Surface Area: S = 4000*V/D
                df.at[idx, sa_col] = (4000 * pv) / pd_val

            elif pd.isna(pv) and pd.notna(sa) and pd.notna(pd_val) and sa > 0:
                # Calculate Pore Volume: V = (D*S)/4000
                df.at[idx, pv_col] = (pd_val * sa) / 4000

            elif pd.isna(pd_val) and pd.notna(sa) and pd.notna(pv) and sa > 0:
                # Calculate Pore Diameter: D = 4000*V/S
                df.at[idx, pd_col] = (4000 * pv) / sa

    return df


def fill_pore_properties_vectorized(df):
    """
    Vectorized version for better performance on large datasets.
    Handles both bare and functionalized columns automatically.
    """
    df = df.copy()

    # Define column pairs: (functionalized, bare)
    col_pairs = [
        ('BET_Surface_Area_m2_g', 'BET_Bare_Surface_Area_m2_g'),
        ('Pore_Volume_cm3_g', 'Bare_Pore_Volume_cm3_g'),
        ('Average_Pore_Diameter_nm', 'Average_Bare_Pore_Diameter_nm')
    ]

    for func_col, bare_col in col_pairs:
        # Skip if columns don't exist
        if func_col not in df.columns or bare_col not in df.columns:
            continue

        # Process functionalized columns
        df = _fill_missing_vectorized(df, func_col)
        # Process bare columns
        df = _fill_missing_vectorized(df, bare_col)

    return df


def _fill_missing_vectorized(df, col_prefix):
    """Helper function to fill missing values for a set of three related columns."""
    # Infer the three column names from prefix
    if 'Bare' in col_prefix:
        sa_col = 'BET_Bare_Surface_Area_m2_g'
        pv_col = 'Bare_Pore_Volume_cm3_g'
        pd_col = 'Average_Bare_Pore_Diameter_nm'
    else:
        sa_col = 'BET_Surface_Area_m2_g'
        pv_col = 'Pore_Volume_cm3_g'
        pd_col = 'Average_Pore_Diameter_nm'

    # Skip if any column missing from dataframe
    if not all(c in df.columns for c in [sa_col, pv_col, pd_col]):
        return df

    # Fill missing Surface Area: S = 4000*V/D
    mask_sa = df[sa_col].isna() & df[pv_col].notna() & df[pd_col].notna() & (df[pd_col] > 0)
    df.loc[mask_sa, sa_col] = (4000 * df.loc[mask_sa, pv_col]) / df.loc[mask_sa, pd_col]

    # Fill missing Pore Volume: V = (D*S)/4000
    mask_pv = df[pv_col].isna() & df[sa_col].notna() & df[pd_col].notna() & (df[sa_col] > 0)
    df.loc[mask_pv, pv_col] = (df.loc[mask_pv, pd_col] * df.loc[mask_pv, sa_col]) / 4000

    # Fill missing Pore Diameter: D = 4000*V/S
    mask_pd = df[pd_col].isna() & df[sa_col].notna() & df[pv_col].notna() & (df[sa_col] > 0)
    df.loc[mask_pd, pd_col] = (4000 * df.loc[mask_pd, pv_col]) / df.loc[mask_pd, sa_col]

    return df
def process_dac_data(input_file='data/DAC_sample_papers_extract/dac_extracted_data.csv',
                     output_file='data/DAC_sample_papers_extract/dac_processed_data_by_LLM.csv'):
    """
    Process extracted DAC data with various transformations and calculations.

    Args:
        input_file (str): Path to input CSV file containing raw extracted data
        output_file (str): Path to output CSV file for processed data

    Returns:
        pandas.DataFrame: Processed dataframe with standardized units and filled values
    """

    # Load data
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records from {input_file}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file}")
    except Exception as e:
        raise Exception(f"Error reading input file: {str(e)}")

    # ==================== 1. Fill default MW_Mn for Amine 1 ====================
    default_mw = {
        'PEI': 600,  # Typical average molecular weight for branched PEI
        'BPEI': 600,  # Branched Polyethyleneimine
        'TEPA': 189.3,  # Tetraethylenepentamine
        'DEA': 105.14,  # Diethanolamine
        'DETA': 103.17,  # Diethylenetriamine
        'LPEI': 423,  # Linear Polyethyleneimine (approximate)
        'Ph-3-ED': 136.19,  # Phenyl-3-ethylenediamine
        'Ph-3-PD': 150.22,  # Phenyl-3-propanediamine
        'Ph-6-ED': 178.27,  # Phenyl-6-ethylenediamine
        'Ph-6-PD': 192.3,  # Phenyl-6-propanediamine
        'PEG200': 200,  # Polyethylene Glycol 200
        'TETA': 146.23,  # Triethylenetetramine
        'TPTA': 232.38,  # Tetrapropylenetetramine
        'EI-Den': 716.9,  # Ethylenediamine-core Dendrimer (G1, approximate)
        'PI-Den': 716.9,  # Propylenimine-core Dendrimer (G1, approximate)
        'AM-TEPA': 189.3,  # Amidoamine-TEPA derivative
        'PAA': 72.06,  # Poly(acrylic acid) monomer unit
        'GPAA': 72.06,  # Grafted Poly(acrylic acid)
        'CTMA+': 284.5,  # Cetyltrimethylammonium cation
        'PPG': 400,  # Polypropylene glycol (approximate)
        'LPPI': 57.11,  # Linear Polypropylenimine monomer unit
        'PGA': 129.12,  # Polyglutamic acid monomer unit
        'PZ': 86.14,  # Piperazine
        'MEA': 61.08,  # Monoethanolamine
        'EDA': 60.1,  # Ethylenediamine
        'Spermine': 202.34,  # Natural polyamine
        'Spermidine': 145.25,  # Natural triamine
        'TREN': 146.23,  # Tris(2-aminoethyl)amine
        'EP': 93.13,  # Epichlorohydrin-modified amine (approximate)
        'EB-TEPA': 189.3,  # Ethylene-Bridged TEPA
        'PEHA': 232.38,  # Pentaethylenehexamine
        'AN-TEPA': 189.3  # Acrylonitrile-modified TEPA
    }

    def transform_co2_concentration(conc_str):
        """
        Transform CO2 concentration to vol% (volume percentage).
        Handles ppm, %, Torr, and other common units.
        """
        if pd.isna(conc_str) or conc_str == '':
            return np.nan

        try:
            conc_str = str(conc_str).strip().lower()

            # Extract numeric value
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', conc_str)
            if not numbers:
                return np.nan

            value = float(numbers[0])

            # Convert based on unit
            if 'ppm' in conc_str:
                # 1 ppm = 0.0001%
                return value * 0.0001
            elif 'torr' in conc_str:
                # Assuming standard atmospheric pressure (760 Torr)
                # vol% = (pressure in Torr / 760) * 100
                return (value / 760) * 100
            elif '%' in conc_str or 'vol%' in conc_str or 'vol %' in conc_str:
                return value
            elif 'ppmv' in conc_str:
                # ppmv is equivalent to ppm for gases
                return value * 0.0001
            elif 'vol' in conc_str and 'frac' in conc_str:
                # Volume fraction (e.g., 0.0004) to percentage
                return value * 100
            else:
                # If no unit specified, assume it's already in %
                # But check if value is suspiciously small (likely ppm)
                if value < 1 and value > 0.0001:
                    # Could be ppm mislabeled, but we'll keep as is
                    return value
                elif value <= 0.0001:
                    # Very small values might be ppm
                    return value * 0.0001 * 100  # Convert ppm to %
                else:
                    return value
        except (ValueError, IndexError):
            return np.nan
    df['CO2_Concentration_vol_pct'] = df['CO2_Concentration'].apply(transform_co2_concentration)

    def fill_mw_mn(row):
        amine = row['Amine_1_or_Additive_1']
        current_mw = row['MW_Mn_g_mol']

        # If MW is missing or empty and amine exists, fill with default
        if pd.isna(current_mw) or (isinstance(current_mw, str) and current_mw.strip() == ''):
            if isinstance(amine, str) and amine.strip() != '':
                return default_mw.get(amine.strip(), np.nan)
        return current_mw

    df['MW_Mn_g_mol'] = df.apply(fill_mw_mn, axis=1)
    print(f"Filled MW_Mn_g_mol for {df['MW_Mn_g_mol'].notna().sum()} records")

    # ==================== 2. Enhance N_content if needed ====================
    # Since N_Content_mmol_g already exists in the input data, we'll enhance it based on Organic_Content_pct if needed
    if 'N_Content_mmol_g' in df.columns and 'Organic_Content_pct' in df.columns:
        def enhance_n_content(row):
            try:
                # If N_Content_mmol_g is missing, calculate from Organic_Content_pct
                if pd.isna(row['N_Content_mmol_g']):
                    org_content = str(row['Organic_Content_pct']).replace('%', '')

                    # Skip if organic content is missing or empty
                    if pd.isna(org_content) or (isinstance(org_content, str) and org_content.strip() == ''):
                        return row['N_Content_mmol_g']

                    # Convert to float if it's a string
                    if isinstance(org_content, str):
                        # Clean the string
                        org_content = org_content.strip()
                        if org_content == '':
                            return row['N_Content_mmol_g']
                        org_val = float(org_content)
                    else:
                        org_val = float(org_content)

                    # Calculate N content: Organic_Content(%) / 4.3
                    # This gives N content in mmol/g
                    n_content = org_val / 4.3

                    # Return the calculated value if it makes sense
                    if n_content > 0:
                        return n_content
                    else:
                        return row['N_Content_mmol_g']
                else:
                    # If N_Content_mmol_g is not missing, return as is
                    return row['N_Content_mmol_g']
            except (ValueError, TypeError):
                return row['N_Content_mmol_g']

        df['N_Content_mmol_g_enhanced'] = df.apply(enhance_n_content, axis=1)
        # Update N_Content_mmol_g with enhanced values where original was missing
        df['N_Content_mmol_g'] = df['N_Content_mmol_g'].fillna(df['N_Content_mmol_g_enhanced'])
        df.drop(columns=['N_Content_mmol_g_enhanced'], inplace=True)
        print(f"Enhanced N_content for {df['N_Content_mmol_g'].notna().sum()} records")
    else:
        print("N_Content_mmol_g or Organic_Content_pct not found in input data")

    # ==================== 3. Default Relative_Humidity ====================
    if 'Relative_Humidity_pct' in df.columns:
        def set_default_rh(row):
            co2_capacity = row['CO2_Capacity_mmol_g']
            current_rh = row['Relative_Humidity_pct']

            # Check if CO2 capacity exists and is not empty
            co2_exists = (not pd.isna(co2_capacity)) and (
                not (isinstance(co2_capacity, str) and co2_capacity.strip() == ''))

            # If RH is missing/empty and CO2 capacity exists, set to 0
            if (pd.isna(current_rh) or (isinstance(current_rh, str) and str(current_rh).strip() == '')) and co2_exists:
                return 0.0
            return current_rh

        df['Relative_Humidity_pct'] = df.apply(set_default_rh, axis=1)
    else:
        print("Relative_Humidity_pct not found in input data")
    print(f"Set default RH for records with CO2 capacity")
    # ==================== 4. Surface Area, Volume, Diameter fill up ====================

    df = fill_pore_properties_vectorized(df)

    # ==================== Fill missing values with mode from same Source_File ====================
    def fill_with_group_mode(df, group_col='Source_File', fill_cols=None):
        """
        Fill missing values in specified columns with the mode (most frequent value)
        from rows with the same Source_File.

        Parameters:
        -----------
        df : pandas DataFrame
            Input dataframe
        group_col : str
            Column to group by (default: 'Source_File')
        fill_cols : list
            List of columns to fill missing values for
        """
        if fill_cols is None:
            fill_cols = [
                'Relative_Humidity_pct',
                'CO2_Concentration_vol_pct',
                'Flow_Rate_mL_min',
                'Adsorption_Time_min',
                'Adsorption_Temperature_C'
            ]

        # Check if group column exists
        if group_col not in df.columns:
            print(f"Warning: {group_col} not found in dataframe. Skipping group mode filling.")
            return df

        df = df.copy()

        for col in fill_cols:
            if col not in df.columns:
                print(f"Warning: {col} not found in dataframe. Skipping.")
                continue

            # Only process if there are missing values
            if df[col].isna().any():
                # Calculate mode for each group
                group_modes = df.groupby(group_col)[col].agg(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                )

                # Create a mapping dictionary
                mode_dict = group_modes.to_dict()

                # Fill missing values using the mode from same Source_File
                missing_mask = df[col].isna()

                for idx in df[missing_mask].index:
                    source_file = df.at[idx, group_col]
                    if source_file in mode_dict and not pd.isna(mode_dict[source_file]):
                        df.at[idx, col] = mode_dict[source_file]

                # Count how many were filled
                filled_count = missing_mask.sum() - df[col].isna().sum()
                print(f"Filled {filled_count} missing values in '{col}' using mode from same {group_col}")

        return df

    # Apply the function before cleaning numerical columns
    print("\n=== Filling missing values with group mode ===")
    df = fill_with_group_mode(df)
    # ==================== 12. Clean and format numerical columns ====================
    # List of columns to keep only numerical values
    numerical_columns = [
        'Organic_Content_pct', 'N_Content_mmol_g',
        'BET_Bare_Surface_Area_m2_g',  # New: Bare support properties
        'Bare_Pore_Volume_cm3_g',  # New
        'Average_Bare_Pore_Diameter_nm',  # New
        'BET_Surface_Area_m2_g',  # After loading properties
        'Pore_Volume_cm3_g', 'Average_Pore_Diameter_nm', 'Relative_Humidity_pct',
        'CO2_Concentration_vol_pct',
        'Flow_Rate_mL_min', 'Adsorption_Time_min',
        'Adsorption_Temperature_C',
        'Amine_Efficiency_mmol_mmol', 'Time_to_Half_Saturation',
        'Time_to_90_Saturation', 'Desorption_Temperature_C',
        'Desorption_Time', 'Weight_Loss_Stability_pct', 'Capacity_Loss_Stability_pct',
        'Heat_of_Adsorption_kJ_mol'
    ]

    def extract_numeric_value(val):
        if pd.isna(val):
            return np.nan

        if isinstance(val, (int, float, np.number)):
            return float(val)

        if isinstance(val, str):
            # Remove non-numeric characters except decimal point and minus sign
            clean_val = re.sub(r'[^\d\.\-]', '', val)
            if clean_val and clean_val != '-':
                try:
                    return float(clean_val)
                except ValueError:
                    return np.nan

        return np.nan

    # Apply numeric extraction to relevant columns
    for col in numerical_columns:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric_value)

    print(f"Cleaned numerical columns")

    # ==================== 13. Create summary ====================
    summary = {
        'total_records': len(df),
        'records_with_co2_capacity': df['CO2_Capacity_mmol_g'].notna().sum(),
        'records_with_amine_efficiency': df['Amine_Efficiency_mmol_mmol'].notna().sum(),
        'records_with_heat_of_adsorption': df['Heat_of_Adsorption_kJ_mol'].notna().sum(),
        'unique_supports': df['Support'].nunique(),
        'unique_amines': df['Amine_1_or_Additive_1'].nunique(),
        'unique_dois': df['DOI'].nunique()
    }

    print("\n=== Processing Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # ==================== 14. Save processed data ====================
    # Create output DataFrame with available columns
    df_output = df

    # Fill missing values for specific columns
    str_columns = [
        "Support",
        "Amine_1_or_Additive_1",
        "Amine_2_or_Additive_2",
        "Amine_3_or_Additive_3",
    ]

    numerical_columns_fill = [
        "Relative_Humidity_pct",
    ]

    # Create a copy to avoid SettingWithCopyWarning
    df_output = df_output.copy()

    # Fill string columns with NaN (not 0)
    for col in str_columns:
        if col in df_output.columns:
            df_output[col] = df_output[col].fillna(0)

    # Fill numerical columns with 0
    for col in numerical_columns_fill:
        if col in df_output.columns:
            df_output[col] = df_output[col].fillna(0)

    # Save to CSV
    df_output.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to {output_file}")

    # Display sample of processed data
    print("\n=== Sample of Processed Data (first 5 rows) ===")
    print(df_output.head())

    return df_output


if __name__ == "__main__":
    # Process the data
    processed_df = process_dac_data()

    # Display basic statistics
    print("\n=== Basic Statistics ===")
    print(processed_df.describe())