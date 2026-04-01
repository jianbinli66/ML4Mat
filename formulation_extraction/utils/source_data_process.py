import pandas as pd
import openpyxl
import numpy as np


def get_formula_cells(file_path, sheet_name=None):
    """
    Get coordinates of all formula cells
    """
    wb = openpyxl.load_workbook(file_path, data_only=False)
    if sheet_name:
        ws = wb[sheet_name]
    else:
        ws = wb.active

    formula_cells = []

    for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=False), start=2):
        for cell in row:
            if cell.data_type == 'f':  # Formula cell
                formula_cells.append({
                    'row': row_idx - 2,  # Convert to 0-based DataFrame index
                    'col': cell.column - 1,  # Convert to 0-based DataFrame index
                    'address': cell.coordinate,
                    'formula': cell.value
                })

    return formula_cells


def remove_formula_values(df, formula_cells):
    """
    Remove values from DataFrame at formula cell positions
    """
    df_cleaned = df.copy()

    for cell in formula_cells:
        row_idx = cell['row']
        col_idx = cell['col']

        # Check if indices are within DataFrame bounds
        if 0 <= row_idx < len(df_cleaned) and 0 <= col_idx < len(df_cleaned.columns):
            df_cleaned.iloc[row_idx, col_idx] = np.nan

    return df_cleaned


# Usage
file_path = 'data/DAC_sample_papers_extract/source_data.xlsx'

# Read the original data
df_original = pd.read_excel(file_path, sheet_name='Main Dataset')

# Get all formula cells
formula_cells = get_formula_cells(file_path, sheet_name='Main Dataset')

# Remove formula values
df_cleaned = remove_formula_values(df_original, formula_cells)

print(f"Found {len(formula_cells)} formula cells")
print("DataFrame with formula values removed:")
print(df_cleaned.head())

# Optional: Show which cells were formulas
print("\nFormula cells found:")
for cell in formula_cells[:10]:  # Show first 10
    print(f"Cell {cell['address']}: {cell['formula']}")

df_cleaned.to_csv('data/DAC_sample_papers_extract/source_data_processed.csv',index=False)