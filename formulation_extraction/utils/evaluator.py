import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
from collections import defaultdict
import ast

warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)


class LLMExtractionEvaluator:
    """
    Class to evaluate LLM-based data extraction performance against human-extracted data
    """

    def __init__(self, llm_data: pd.DataFrame, human_data: pd.DataFrame, category_columns, numerical_columns):
        """
        Initialize the evaluator with LLM-extracted and human-extracted data

        Args:
            llm_data: DataFrame containing LLM-extracted data
            human_data: DataFrame containing human-extracted data
            category_columns: List of categorical column names
            numerical_columns: List of numerical column names
        """
        self.llm_data = llm_data.copy()
        self.human_data = human_data.copy()

        # Define column categories
        self.category_columns = category_columns
        self.numerical_columns = numerical_columns

        # Material matching columns - UPDATED: removed MW_Mn_g_mol
        self.material_columns = [
            "Support",
            "Amine_1_or_Additive_1",
            "Amine_2_or_Additive_2",
            "Amine_3_or_Additive_3",
            "Organic_Content_pct",
        ]

        # Columns that may have multiple values in LLM data due to different test methods
        self.multi_value_columns = ["CO2_Capacity_mmol_g"]

        # Prepare dataframes
        self._prepare_dataframes()

    def _prepare_dataframes(self):
        """Clean and prepare dataframes for comparison"""
        # Standardize column names
        self.llm_data.columns = [col.strip().replace(" ", "_").replace("__", "_") for col in self.llm_data.columns]
        self.human_data.columns = [col.strip().replace(" ", "_").replace("__", "_") for col in self.human_data.columns]

        # Convert numerical columns to numeric in human data
        for col in self.numerical_columns:
            if col in self.human_data.columns:
                self.human_data[col] = pd.to_numeric(self.human_data[col], errors='coerce')

        # Convert material columns to string for comparison
        for col in self.material_columns:
            if col in self.llm_data.columns:
                self.llm_data[col] = self.llm_data[col].astype(str).str.strip()
            if col in self.human_data.columns:
                self.human_data[col] = self.human_data[col].astype(str).str.strip()

    def _find_matching_rows(self, llm_row: pd.Series) -> pd.DataFrame:
        """
        Find matching rows in human data for a given LLM row
        ONLY using material_columns for matching (updated: without MW_Mn_g_mol)

        Args:
            llm_row: Single row from LLM data

        Returns:
            DataFrame of matching rows from human data
        """
        # Match by DOI first
        doi = str(llm_row.get('DOI', '')).strip()
        if doi == 'nan' or not doi:
            return pd.DataFrame()

        # Get all human rows with matching DOI
        human_matches = self.human_data[self.human_data['DOI'].astype(str).str.strip() == doi].copy()

        if len(human_matches) == 0:
            return pd.DataFrame()

        # For material matching, ONLY check material columns (updated list)
        matching_rows = []
        for _, human_row in human_matches.iterrows():
            match_score = 0
            total_material_cols = 0
            non_empty_cols = 0

            # Check ONLY material columns
            for col in self.material_columns:
                if col in llm_row and col in human_row:
                    total_material_cols += 1
                    llm_val = str(llm_row[col]).strip() if pd.notna(llm_row[col]) else ''
                    human_val = str(human_row[col]).strip() if pd.notna(human_row[col]) else ''

                    # Count non-empty columns
                    if llm_val != '' and llm_val != 'nan':
                        non_empty_cols += 1

                    if llm_val == human_val or (not llm_val and not human_val):
                        match_score += 1
                    elif pd.isna(llm_row[col]) and pd.isna(human_row[col]):
                        match_score += 1

            # Calculate match percentage based on non-empty columns
            if non_empty_cols > 0:
                match_percentage = match_score / non_empty_cols
            else:
                match_percentage = 0

            # Consider it a match if at least 80% of non-empty material columns match
            if match_percentage >= 0.8:
                matching_rows.append(human_row)

        return pd.DataFrame(matching_rows)

    def _calculate_column_accuracy(self, llm_val, human_val, col_name: str) -> Tuple[bool, str, float]:
        """
        Calculate if a single column value matches
        Special handling for multi-value columns and MAE for CO2_Capacity_mmol_g

        Args:
            llm_val: Value from LLM data
            human_val: Value from human data
            col_name: Column name

        Returns:
            Tuple of (is_correct, error_type, error_value)
            For CO2_Capacity_mmol_g, error_value is the MAE
            For other columns, error_value is error percentage
        """
        # Handle NaN/None values
        if pd.isna(llm_val) and pd.isna(human_val):
            return True, "Both NaN", 0.0
        elif pd.isna(llm_val) or pd.isna(human_val):
            return False, "Missing value", 100.0

        # Special handling for CO2_Capacity_mmol_g - use MAE
        if col_name == "CO2_Capacity_mmol_g":
            return self._calculate_co2_mae(llm_val, human_val)

        # Special handling for other multi-value columns
        if col_name in self.multi_value_columns:
            return self._calculate_multi_value_accuracy(llm_val, human_val, col_name)

        # Convert to string for category columns
        if col_name in self.category_columns:
            llm_str = str(llm_val).strip().lower()
            human_str = str(human_val).strip().lower()

            # Handle numeric values in category columns
            try:
                llm_num = float(llm_str)
                human_num = float(human_str)
                if llm_num == human_num:
                    return True, "Exact match", 0.0
                else:
                    return False, f"Mismatch", 100.0
                # else:
                #     error_pct = abs(llm_num - human_num) / abs(human_num) * 100 if human_num != 0 else 100
                #     return False, f"Numerical mismatch", error_pct
            except:
                if llm_str == human_str:
                    return True, "Exact match", 0.0
                else:
                    return False, "Text mismatch", 100.0

        # Numerical columns with 10% tolerance
        elif col_name in self.numerical_columns:
            try:
                llm_num = float(llm_val)
                human_num = float(human_val)

                if human_num == 0:
                    if llm_num == 0:
                        return True, "Zero value", 0.0
                    else:
                        return False, "Non-zero vs zero", 100.0

                error_pct = abs(llm_num - human_num) / abs(human_num) * 100
                if error_pct <= 10:
                    return True, f"Within {error_pct:.1f}%", error_pct
                else:
                    return False, f"Outside {error_pct:.1f}%", error_pct
            except:
                return False, "Conversion error", 100.0

        return False, "Unknown column type", 100.0

    def _calculate_co2_mae(self, llm_val, human_val) -> Tuple[bool, str, float]:
        """
        Calculate MAE for CO2_Capacity_mmol_g
        If multiple values exist, find the one with minimum absolute error

        Args:
            llm_val: Value from LLM data (can be single value or list)
            human_val: Single value from human data

        Returns:
            Tuple of (is_correct, error_type, mae_value)
            Note: is_correct is always False for MAE calculation
        """
        # Convert human value to float
        try:
            human_num = float(human_val)
        except:
            return False, "Human value not numeric", float('nan')

        # Convert llm_val to list if it's not already
        if isinstance(llm_val, list):
            llm_values = llm_val
        elif isinstance(llm_val, str):
            # Try to parse string as list
            try:
                # Remove extra quotes and brackets
                clean_str = llm_val.replace('"', '').replace("'", "")
                if clean_str.startswith('[') and clean_str.endswith(']'):
                    llm_values = ast.literal_eval(clean_str)
                    if not isinstance(llm_values, list):
                        llm_values = [llm_values]
                else:
                    llm_values = [clean_str]
            except:
                llm_values = [llm_val]
        else:
            llm_values = [llm_val]

        # Handle empty llm_values
        if not llm_values or len(llm_values) == 0:
            return False, "Empty LLM value", float('nan')

        # Calculate absolute errors for each value and find minimum
        min_absolute_error = float('inf')
        best_llm_value = None
        valid_values_found = False

        for llm_item in llm_values:
            if pd.isna(llm_item):
                continue

            try:
                llm_float = float(llm_item)
                absolute_error = abs(llm_float - human_num)

                if not np.isnan(absolute_error):
                    valid_values_found = True
                    if absolute_error < min_absolute_error:
                        min_absolute_error = absolute_error
                        best_llm_value = llm_item
            except:
                continue

        if valid_values_found:
            return False, f"MAE = {min_absolute_error:.4f}", min_absolute_error
        else:
            return False, "No valid numeric values", float('nan')

    def _calculate_multi_value_accuracy(self, llm_val, human_val, col_name: str) -> Tuple[bool, str, float]:
        """
        Calculate accuracy for multi-value columns (excluding CO2_Capacity_mmol_g)
        If any value in llm_val matches human_val (within tolerance), it's considered correct

        Args:
            llm_val: Value from LLM data (can be single value or list)
            human_val: Single value from human data
            col_name: Column name

        Returns:
            Tuple of (is_correct, error_type, error_percentage)
        """
        # Convert human value to float
        try:
            human_num = float(human_val)
        except:
            return False, "Human value not numeric", 100.0

        # Convert llm_val to list if it's not already
        if isinstance(llm_val, list):
            llm_values = llm_val
        elif isinstance(llm_val, str):
            # Try to parse string as list
            try:
                # Remove extra quotes and brackets
                clean_str = llm_val.replace('"', '').replace("'", "")
                if clean_str.startswith('[') and clean_str.endswith(']'):
                    llm_values = ast.literal_eval(clean_str)
                    if not isinstance(llm_values, list):
                        llm_values = [llm_values]
                else:
                    llm_values = [clean_str]
            except:
                llm_values = [llm_val]
        else:
            llm_values = [llm_val]

        # Handle empty llm_values
        if not llm_values or len(llm_values) == 0:
            return False, "Empty LLM value", 100.0

        # Check each value in the list
        best_error_pct = float('inf')
        best_llm_value = None

        for llm_item in llm_values:
            if pd.isna(llm_item):
                continue

            try:
                llm_float = float(llm_item)

                if human_num == 0:
                    if llm_float == 0:
                        return True, "Zero value match", 0.0
                    else:
                        error_pct = 100.0
                else:
                    error_pct = abs(llm_float - human_num) / abs(human_num) * 100

                # Track best match
                if error_pct < best_error_pct:
                    best_error_pct = error_pct
                    best_llm_value = llm_item

                # Early return if we find a good match (within 10%)
                if error_pct <= 10:
                    return True, f"Multi-value match within {error_pct:.1f}%", error_pct

            except:
                continue

        # If we get here, no value was within 10%
        if best_error_pct < float('inf'):
            return False, f"Best match {best_error_pct:.1f}% away", best_error_pct
        else:
            return False, "No valid values in LLM data", 100.0

    def _calculate_precision_recall_f1(self, evaluation_results: Dict, include_unmatched: bool = True) -> Dict:
        """
        Calculate precision, recall, and F1 scores for each column and overall
        Now includes accuracy in the metrics and optionally considers unmatched rows

        Args:
            evaluation_results: Results from evaluate_extraction()
            include_unmatched: Whether to include unmatched rows in metrics

        Returns:
            Dictionary with accuracy, precision, recall, and F1 scores
        """
        column_metrics = {}

        # For each column, calculate TP, FP, FN
        for col in self.category_columns + self.numerical_columns:
            if col not in evaluation_results['column_accuracy']:
                continue

            stats = evaluation_results['column_accuracy'][col]
            total_checked = stats['total_checked']
            correct = stats['correct']
            errors = stats['errors']
            accuracy = stats['accuracy_rate']

            # Get unmatched count for this column if including unmatched rows
            unmatched_count = 0
            if include_unmatched and 'unmatched_stats' in evaluation_results:
                unmatched_count = evaluation_results['unmatched_stats'].get(col, 0)

            # For extraction tasks:
            # - True Positives: Correctly extracted values
            # - False Positives: Incorrectly extracted values (errors)
            # - False Negatives: Values that should have been extracted but weren't
            #   Unmatched rows contribute to false negatives
            tp = correct
            fp = errors
            fn = unmatched_count  # Unmatched rows are false negatives

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            column_metrics[col] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'total_matched': total_checked,
                'unmatched': unmatched_count,
                'total_expected': total_checked + unmatched_count
            }

        # Calculate overall metrics (macro average)
        if column_metrics:
            overall_accuracy = np.mean([m['accuracy'] for m in column_metrics.values()])
            overall_precision = np.mean([m['precision'] for m in column_metrics.values()])
            overall_recall = np.mean([m['recall'] for m in column_metrics.values()])
            overall_f1 = np.mean([m['f1_score'] for m in column_metrics.values()])

            # Weighted by total_expected
            total_expected_all = sum([m['total_expected'] for m in column_metrics.values()])
            if total_expected_all > 0:
                weighted_accuracy = sum(
                    [m['accuracy'] * m['total_expected'] for m in column_metrics.values()]) / total_expected_all
                weighted_precision = sum(
                    [m['precision'] * m['total_expected'] for m in column_metrics.values()]) / total_expected_all
                weighted_recall = sum(
                    [m['recall'] * m['total_expected'] for m in column_metrics.values()]) / total_expected_all
                weighted_f1 = sum(
                    [m['f1_score'] * m['total_expected'] for m in column_metrics.values()]) / total_expected_all
            else:
                weighted_accuracy = weighted_precision = weighted_recall = weighted_f1 = 0
        else:
            overall_accuracy = overall_precision = overall_recall = overall_f1 = 0
            weighted_accuracy = weighted_precision = weighted_recall = weighted_f1 = 0

        # Calculate metrics by column type
        cat_cols = [col for col in self.category_columns if col in column_metrics]
        num_cols = [col for col in self.numerical_columns if col in column_metrics]

        cat_metrics = {}
        if cat_cols:
            cat_metrics = {
                'accuracy': np.mean([column_metrics[col]['accuracy'] for col in cat_cols]),
                'precision': np.mean([column_metrics[col]['precision'] for col in cat_cols]),
                'recall': np.mean([column_metrics[col]['recall'] for col in cat_cols]),
                'f1_score': np.mean([column_metrics[col]['f1_score'] for col in cat_cols]),
                'total_columns': len(cat_cols),
                'total_tp': sum([column_metrics[col]['tp'] for col in cat_cols]),
                'total_fp': sum([column_metrics[col]['fp'] for col in cat_cols]),
                'total_fn': sum([column_metrics[col]['fn'] for col in cat_cols])
            }

        num_metrics = {}
        if num_cols:
            num_metrics = {
                'accuracy': np.mean([column_metrics[col]['accuracy'] for col in num_cols]),
                'precision': np.mean([column_metrics[col]['precision'] for col in num_cols]),
                'recall': np.mean([column_metrics[col]['recall'] for col in num_cols]),
                'f1_score': np.mean([column_metrics[col]['f1_score'] for col in num_cols]),
                'total_columns': len(num_cols),
                'total_tp': sum([column_metrics[col]['tp'] for col in num_cols]),
                'total_fp': sum([column_metrics[col]['fp'] for col in num_cols]),
                'total_fn': sum([column_metrics[col]['fn'] for col in num_cols])
            }

        return {
            'column_metrics': column_metrics,
            'overall': {
                'macro_avg_accuracy': overall_accuracy,
                'macro_avg_precision': overall_precision,
                'macro_avg_recall': overall_recall,
                'macro_avg_f1': overall_f1,
                'weighted_avg_accuracy': weighted_accuracy,
                'weighted_avg_precision': weighted_precision,
                'weighted_avg_recall': weighted_recall,
                'weighted_avg_f1': weighted_f1
            },
            'by_type': {
                'categorical': cat_metrics,
                'numerical': num_metrics
            }
        }

    def _calculate_co2_mae_metrics(self, evaluation_results: Dict) -> Dict:
        """
        Calculate MAE metrics specifically for CO2_Capacity_mmol_g

        Args:
            evaluation_results: Results from evaluate_extraction()

        Returns:
            Dictionary with MAE metrics
        """
        co2_mae_values = []

        # Collect MAE values from matched rows
        for row_error in evaluation_results['row_detailed_errors']:
            if row_error['Match_Found'] and row_error['Errors']:
                for error in row_error['Errors']:
                    if error['column'] == 'CO2_Capacity_mmol_g':
                        if 'MAE' in error['error_type']:
                            co2_mae_values.append(error['error_percentage'])

        # Add unmatched rows as MAE = human value (since LLM value is missing)
        # This assumes the human value is the expected value
        unmatched_dois = set()
        for result in evaluation_results['detailed_results'].iterrows():
            if not result[1]['Match_Found']:
                unmatched_dois.add(result[1]['DOI'])

        # For each unmatched DOI, add MAE for each human row
        for doi in unmatched_dois:
            human_rows = self.human_data[self.human_data['DOI'].astype(str).str.strip() == doi]
            for _, human_row in human_rows.iterrows():
                if pd.notna(human_row.get('CO2_Capacity_mmol_g')):
                    try:
                        human_val = float(human_row['CO2_Capacity_mmol_g'])
                        co2_mae_values.append(human_val)  # MAE = human value (since LLM value is 0 or missing)
                    except:
                        pass

        if co2_mae_values:
            return {
                'mean_mae': np.mean(co2_mae_values),
                'median_mae': np.median(co2_mae_values),
                'std_mae': np.std(co2_mae_values),
                'min_mae': np.min(co2_mae_values),
                'max_mae': np.max(co2_mae_values),
                'count': len(co2_mae_values),
                'mae_values': co2_mae_values
            }
        else:
            return {
                'mean_mae': 0,
                'median_mae': 0,
                'std_mae': 0,
                'min_mae': 0,
                'max_mae': 0,
                'count': 0,
                'mae_values': []
            }

    def evaluate_extraction(self, include_unmatched_in_metrics: bool = True) -> Dict:
        """
        Main evaluation function with enhanced row-based analysis
        Now includes unmatched rows in metrics and MAE for CO2_Capacity_mmol_g

        Args:
            include_unmatched_in_metrics: Whether to include unmatched rows in precision/recall/F1

        Returns:
            Dictionary containing all evaluation metrics
        """
        results = []
        row_detailed_errors = []
        column_stats = {col: {'total': 0, 'correct': 0, 'errors': []}
                        for col in self.category_columns + self.numerical_columns}

        # Track unmatched rows per column
        unmatched_stats = {col: 0 for col in self.category_columns + self.numerical_columns}

        total_llm_rows = len(self.llm_data)
        matched_rows = 0
        total_errors_by_row = []

        # New metrics
        row_accuracy_scores = []
        row_error_counts = []
        row_error_types = defaultdict(list)
        column_error_frequencies = defaultdict(int)

        for idx, llm_row in self.llm_data.iterrows():
            # Find matching human rows using ONLY material columns
            human_matches = self._find_matching_rows(llm_row)

            if len(human_matches) == 0:
                # No match found - track as unmatched for each column
                for col in self.category_columns + self.numerical_columns:
                    if col in llm_row and pd.notna(llm_row[col]):
                        unmatched_stats[col] += 1

                results.append({
                    'LLM_Index': idx,
                    'DOI': llm_row.get('DOI', ''),
                    'Match_Found': False,
                    'Num_Matches': 0,
                    'Best_Match_Index': None,
                    'Row_Errors': None,
                    'Accuracy': 0,
                    'Perfect_Match': False
                })
                row_detailed_errors.append({
                    'LLM_Index': idx,
                    'DOI': llm_row.get('DOI', ''),
                    'Match_Found': False,
                    'Errors': [],
                    'Error_Count': 0,
                    'Category_Errors': 0,
                    'Numerical_Errors': 0
                })
                continue

            matched_rows += 1

            # For each matching row, calculate errors
            match_results = []
            for human_idx, human_row in human_matches.iterrows():
                row_errors = {}
                row_error_details = []
                total_columns = 0
                correct_columns = 0
                category_errors = 0
                numerical_errors = 0

                # Check all columns
                for col in self.category_columns + self.numerical_columns:
                    if col in llm_row and col in human_row:
                        total_columns += 1
                        is_correct, error_type, error_value = self._calculate_column_accuracy(
                            llm_row[col], human_row[col], col
                        )

                        if not is_correct:
                            row_errors[col] = {
                                'llm_value': llm_row[col],
                                'human_value': human_row[col],
                                'error_type': error_type,
                                'error_value': error_value
                            }
                            row_error_details.append({
                                'column': col,
                                'llm_value': llm_row[col],
                                'human_value': human_row[col],
                                'error_type': error_type,
                                'error_value': error_value
                            })

                            # Count error types
                            if col in self.category_columns:
                                category_errors += 1
                            else:
                                numerical_errors += 1

                            column_error_frequencies[col] += 1
                            row_error_types[error_type].append(col)
                        else:
                            correct_columns += 1

                accuracy = correct_columns / total_columns if total_columns > 0 else 0
                num_errors = len(row_errors)

                match_results.append({
                    'human_index': human_idx,
                    'errors': row_errors,
                    'error_details': row_error_details,
                    'num_errors': num_errors,
                    'accuracy': accuracy,
                    'correct_columns': correct_columns,
                    'total_columns': total_columns,
                    'category_errors': category_errors,
                    'numerical_errors': numerical_errors,
                    'human_row_data': human_row.to_dict()
                })

            # Choose the match with maximum accuracy (minimum errors)
            if match_results:
                best_match = max(match_results, key=lambda x: x['accuracy'])
                is_perfect_match = best_match['num_errors'] == 0

                # Update column statistics and collect row metrics
                row_error_counts.append(best_match['num_errors'])
                row_accuracy_scores.append(best_match['accuracy'])

                for col in self.category_columns + self.numerical_columns:
                    if col in llm_row and col in best_match['errors']:
                        column_stats[col]['errors'].append({
                            'llm_index': idx,
                            'llm_value': best_match['errors'][col]['llm_value'],
                            'human_value': best_match['errors'][col]['human_value'],
                            'error_type': best_match['errors'][col]['error_type'],
                            'error_value': best_match['errors'][col]['error_value']
                        })
                        column_stats[col]['total'] += 1
                    elif col in llm_row:
                        column_stats[col]['correct'] += 1
                        column_stats[col]['total'] += 1

                total_errors_by_row.append(best_match['num_errors'])

                results.append({
                    'LLM_Index': idx,
                    'DOI': llm_row.get('DOI', ''),
                    'Match_Found': True,
                    'Num_Matches': len(human_matches),
                    'Best_Match_Index': best_match['human_index'],
                    'Row_Errors': best_match['num_errors'],
                    'Accuracy': best_match['accuracy'],
                    'Perfect_Match': is_perfect_match,
                    'Category_Errors': best_match['category_errors'],
                    'Numerical_Errors': best_match['numerical_errors']
                })

                row_detailed_errors.append({
                    'LLM_Index': idx,
                    'DOI': llm_row.get('DOI', ''),
                    'Match_Found': True,
                    'Errors': best_match['error_details'],
                    'Error_Count': best_match['num_errors'],
                    'Category_Errors': best_match['category_errors'],
                    'Numerical_Errors': best_match['numerical_errors'],
                    'Human_Row_Index': best_match['human_index']
                })

        # Calculate overall statistics
        overall_accuracy = 0
        perfect_matches = 0
        if results:
            matched_results = [r for r in results if r['Match_Found']]
            if matched_results:
                overall_accuracy = np.mean([r['Accuracy'] for r in matched_results])
                perfect_matches = sum(1 for r in matched_results if r['Perfect_Match'])

        # Calculate row-based statistics (only for matched rows)
        if row_accuracy_scores:
            avg_row_accuracy = np.mean(row_accuracy_scores)
            median_row_accuracy = np.median(row_accuracy_scores)
            row_accuracy_std = np.std(row_accuracy_scores)
        else:
            avg_row_accuracy = median_row_accuracy = row_accuracy_std = 0

        # Calculate column accuracy rates (only for matched rows)
        column_accuracy = {}
        for col, stats in column_stats.items():
            if stats['total'] > 0:
                column_accuracy[col] = {
                    'accuracy_rate': stats['correct'] / stats['total'],
                    'total_checked': stats['total'],
                    'correct': stats['correct'],
                    'errors': len(stats['errors']),
                    'error_examples': stats['errors'][:5]  # Limit to 5 examples
                }

        # Prepare evaluation results dictionary
        evaluation_results = {
            'summary': {
                'total_llm_rows': total_llm_rows,
                'matched_rows': matched_rows,
                'unmatched_rows': total_llm_rows - matched_rows,
                'match_rate': matched_rows / total_llm_rows if total_llm_rows > 0 else 0,
                'overall_accuracy': overall_accuracy,
                'avg_errors_per_row': np.mean(total_errors_by_row) if total_errors_by_row else 0,
                'median_errors_per_row': np.median(total_errors_by_row) if total_errors_by_row else 0,
                'perfect_matches': perfect_matches,
                'perfect_match_rate': perfect_matches / matched_rows if matched_rows > 0 else 0,
                'avg_row_accuracy': avg_row_accuracy,
                'median_row_accuracy': median_row_accuracy,
                'row_accuracy_std': row_accuracy_std,
                'total_category_errors': sum(r.get('Category_Errors', 0) for r in results if r['Match_Found']),
                'total_numerical_errors': sum(r.get('Numerical_Errors', 0) for r in results if r['Match_Found'])
            },
            'detailed_results': pd.DataFrame(results),
            'row_detailed_errors': row_detailed_errors,
            'column_accuracy': column_accuracy,
            'unmatched_stats': unmatched_stats,
            'lowest_accuracy_columns': sorted(column_accuracy.items(), key=lambda x: x[1]['accuracy_rate'])[:10],
            'highest_error_columns': sorted(column_error_frequencies.items(), key=lambda x: x[1], reverse=True)[:10],
            'error_type_distribution': {k: len(v) for k, v in row_error_types.items()},
            'column_stats': column_stats,
            'row_metrics': {
                'accuracy_scores': row_accuracy_scores,
                'error_counts': row_error_counts
            }
        }

        # Calculate precision, recall, F1 scores with accuracy (including unmatched rows if specified)
        prf_metrics = self._calculate_precision_recall_f1(evaluation_results,
                                                          include_unmatched=include_unmatched_in_metrics)
        evaluation_results['precision_recall_f1'] = prf_metrics

        # Calculate CO2 MAE metrics
        co2_mae_metrics = self._calculate_co2_mae_metrics(evaluation_results)
        evaluation_results['co2_mae_metrics'] = co2_mae_metrics

        return evaluation_results

    def print_detailed_statistics(self, evaluation_results: Dict):
        """Print detailed statistics from evaluation results including CO2 MAE and unmatched rows"""
        print("=" * 80)
        print("DATA EXTRACTION PERFORMANCE EVALUATION - DETAILED STATISTICS")
        print("=" * 80)

        summary = evaluation_results['summary']
        print(f"\n1. OVERALL STATISTICS:")
        print(f"   • Total LLM-extracted rows: {summary['total_llm_rows']}")
        print(f"   • Matched rows (using material columns only): {summary['matched_rows']}")
        print(f"   • Unmatched rows: {summary['unmatched_rows']}")
        print(f"   • Match rate: {summary['match_rate']:.2%}")
        print(f"   • Overall accuracy (matched rows only): {summary['overall_accuracy']:.2%}")
        print(f"   • Perfect matches: {summary['perfect_matches']}")
        print(f"   • Perfect match rate: {summary['perfect_match_rate']:.2%}")
        print(f"   • Average errors per row: {summary['avg_errors_per_row']:.2f}")
        print(f"   • Median errors per row: {summary['median_errors_per_row']:.2f}")
        print(f"   • Average row accuracy: {summary['avg_row_accuracy']:.2%}")
        print(f"   • Median row accuracy: {summary['median_row_accuracy']:.2%}")
        print(f"   • Row accuracy std dev: {summary['row_accuracy_std']:.2%}")
        print(f"   • Total category errors: {summary['total_category_errors']}")
        print(f"   • Total numerical errors: {summary['total_numerical_errors']}")

        # Print CO2 MAE metrics
        co2_mae = evaluation_results.get('co2_mae_metrics', {})
        if co2_mae:
            print(f"\n2. CO2 CAPACITY MAE METRICS:")
            print("-" * 80)
            print(f"   • Mean MAE: {co2_mae.get('mean_mae', 0):.4f} mmol/g")
            print(f"   • Median MAE: {co2_mae.get('median_mae', 0):.4f} mmol/g")
            print(f"   • Std MAE: {co2_mae.get('std_mae', 0):.4f} mmol/g")
            print(f"   • Min MAE: {co2_mae.get('min_mae', 0):.4f} mmol/g")
            print(f"   • Max MAE: {co2_mae.get('max_mae', 0):.4f} mmol/g")
            print(f"   • Number of evaluations: {co2_mae.get('count', 0)}")

        # Print precision, recall, F1 scores with accuracy
        prf_metrics = evaluation_results.get('precision_recall_f1', {})
        if prf_metrics:
            print(f"\n3. PRECISION, RECALL, F1 SCORES WITH ACCURACY (including unmatched rows):")
            print("-" * 80)

            overall = prf_metrics.get('overall', {})
            print(f"   Overall (Macro Average):")
            print(f"     • Accuracy: {overall.get('macro_avg_accuracy', 0):.2%}")
            print(f"     • Precision: {overall.get('macro_avg_precision', 0):.2%}")
            print(f"     • Recall: {overall.get('macro_avg_recall', 0):.2%}")
            print(f"     • F1 Score: {overall.get('macro_avg_f1', 0):.2%}")

            print(f"\n   Overall (Weighted Average):")
            print(f"     • Accuracy: {overall.get('weighted_avg_accuracy', 0):.2%}")
            print(f"     • Precision: {overall.get('weighted_avg_precision', 0):.2%}")
            print(f"     • Recall: {overall.get('weighted_avg_recall', 0):.2%}")
            print(f"     • F1 Score: {overall.get('weighted_avg_f1', 0):.2%}")

            by_type = prf_metrics.get('by_type', {})
            cat_metrics = by_type.get('categorical', {})
            if cat_metrics:
                print(f"\n   Categorical Columns ({cat_metrics.get('total_columns', 0)} columns):")
                print(f"     • Accuracy: {cat_metrics.get('accuracy', 0):.2%}")
                print(f"     • Precision: {cat_metrics.get('precision', 0):.2%}")
                print(f"     • Recall: {cat_metrics.get('recall', 0):.2%}")
                print(f"     • F1 Score: {cat_metrics.get('f1_score', 0):.2%}")
                print(
                    f"     • TP: {cat_metrics.get('total_tp', 0)}, FP: {cat_metrics.get('total_fp', 0)}, FN: {cat_metrics.get('total_fn', 0)}")

            num_metrics = by_type.get('numerical', {})
            if num_metrics:
                print(f"\n   Numerical Columns ({num_metrics.get('total_columns', 0)} columns):")
                print(f"     • Accuracy: {num_metrics.get('accuracy', 0):.2%}")
                print(f"     • Precision: {num_metrics.get('precision', 0):.2%}")
                print(f"     • Recall: {num_metrics.get('recall', 0):.2%}")
                print(f"     • F1 Score: {num_metrics.get('f1_score', 0):.2%}")
                print(
                    f"     • TP: {num_metrics.get('total_tp', 0)}, FP: {num_metrics.get('total_fp', 0)}, FN: {num_metrics.get('total_fn', 0)}")

        print(f"\n4. COLUMN-WISE METRICS (Top 10 lowest accuracy):")
        print("-" * 100)
        print(
            f"{'Column':<35} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Errors':<8} {'Unmatched':<10} {'Total':<8}")
        print("-" * 100)

        column_metrics = prf_metrics.get('column_metrics', {})
        for col, stats in evaluation_results['lowest_accuracy_columns']:
            col_prf = column_metrics.get(col, {})
            accuracy = col_prf.get('accuracy', 0)
            precision = col_prf.get('precision', 0)
            recall = col_prf.get('recall', 0)
            f1 = col_prf.get('f1_score', 0)
            errors = stats['errors']
            unmatched = col_prf.get('unmatched', 0)
            total = col_prf.get('total_expected', stats['total_checked'])

            print(
                f"{col:<35} {accuracy:<10.2%} {precision:<10.2%} {recall:<10.2%} {f1:<10.2%} {errors:<8} {unmatched:<10} {total:<8}")

        print(f"\n5. ERROR ANALYSIS BY COLUMN TYPE:")
        print("-" * 80)

        # Analyze category vs numerical columns
        cat_cols = [col for col in self.category_columns if col in evaluation_results['column_accuracy']]
        num_cols = [col for col in self.numerical_columns if col in evaluation_results['column_accuracy']]

        if cat_cols:
            cat_accuracy = np.mean([evaluation_results['column_accuracy'][col]['accuracy_rate'] for col in cat_cols])
            cat_errors = sum([evaluation_results['column_accuracy'][col]['errors'] for col in cat_cols])
            cat_unmatched = sum([column_metrics.get(col, {}).get('unmatched', 0) for col in cat_cols])

            print(f"   • Category columns:")
            print(f"     - Average accuracy: {cat_accuracy:.2%}")
            print(f"     - Total errors: {cat_errors}")
            print(f"     - Total unmatched: {cat_unmatched}")
            print(f"     - Columns analyzed: {len(cat_cols)}")

        if num_cols:
            num_accuracy = np.mean([evaluation_results['column_accuracy'][col]['accuracy_rate'] for col in num_cols])
            num_errors = sum([evaluation_results['column_accuracy'][col]['errors'] for col in num_cols])
            num_unmatched = sum([column_metrics.get(col, {}).get('unmatched', 0) for col in num_cols])

            print(f"   • Numerical columns:")
            print(f"     - Average accuracy: {num_accuracy:.2%}")
            print(f"     - Total errors: {num_errors}")
            print(f"     - Total unmatched: {num_unmatched}")
            print(f"     - Columns analyzed: {len(num_cols)}")

        print(f"\n6. ERROR DISTRIBUTION IN MATCHED ROWS:")
        matched_results = evaluation_results['detailed_results'][evaluation_results['detailed_results']['Match_Found']]
        if not matched_results.empty:
            print(f"   Error count distribution:")
            error_distribution = matched_results['Row_Errors'].value_counts().sort_index()
            for errors, count in error_distribution.items():
                percentage = count / len(matched_results) * 100
                print(f"     • {errors} errors: {count} rows ({percentage:.1f}%)")

            print(f"\n   Row accuracy distribution:")
            accuracy_bins = pd.cut(matched_results['Accuracy'], bins=[0, 0.5, 0.7, 0.9, 1.0],
                                   labels=['Poor (<50%)', 'Fair (50-70%)', 'Good (70-90%)', 'Excellent (90-100%)'])
            accuracy_dist = accuracy_bins.value_counts()
            for category, count in accuracy_dist.items():
                percentage = count / len(matched_results) * 100
                print(f"     • {category}: {count} rows ({percentage:.1f}%)")

        print(f"\n7. TOP ERROR-PRONE COLUMNS:")
        print("-" * 100)
        print(f"{'Column':<35} {'Error Count':<12} {'Unmatched':<10} {'Column Type':<15} {'Accuracy':<10} {'F1':<10}")
        print("-" * 100)
        for col, count in evaluation_results['highest_error_columns'][:10]:
            col_type = 'Category' if col in self.category_columns else 'Numerical'
            col_prf = column_metrics.get(col, {})
            accuracy = col_prf.get('accuracy', 0)
            f1 = col_prf.get('f1_score', 0)
            unmatched = col_prf.get('unmatched', 0)
            print(f"{col:<35} {count:<12} {unmatched:<10} {col_type:<15} {accuracy:<10.2%} {f1:<10.2%}")

        print(f"\n8. ERROR TYPE DISTRIBUTION:")
        print("-" * 80)
        for error_type, count in evaluation_results['error_type_distribution'].items():
            print(f"   • {error_type}: {count} occurrences")

    def get_row_error_report(self, evaluation_results: Dict, n_rows: int = 10) -> pd.DataFrame:
        """
        Get a detailed error report for the first n rows

        Args:
            evaluation_results: Results from evaluate_extraction()
            n_rows: Number of rows to include in the report

        Returns:
            DataFrame with detailed error information for each row
        """
        detailed_errors = evaluation_results['row_detailed_errors']

        report_data = []
        for i, row_error in enumerate(detailed_errors[:n_rows]):
            if row_error['Match_Found'] and row_error['Error_Count'] > 0:
                for error in row_error['Errors']:
                    error_value_display = f"{error['error_value']:.4f}" if isinstance(error['error_value'],
                                                                                      (int, float)) and not np.isnan(
                        error['error_value']) else str(error['error_value'])
                    report_data.append({
                        'Row_Index': row_error['LLM_Index'],
                        'DOI': row_error['DOI'],
                        'Column': error['column'],
                        'LLM_Value': error['llm_value'],
                        'Human_Value': error['human_value'],
                        'Error_Type': error['error_type'],
                        'Error_Value': error_value_display,
                        'Is_Category': error['column'] in self.category_columns,
                        'Total_Row_Errors': row_error['Error_Count']
                    })
            elif not row_error['Match_Found']:
                report_data.append({
                    'Row_Index': row_error['LLM_Index'],
                    'DOI': row_error['DOI'],
                    'Column': 'ALL',
                    'LLM_Value': 'NO MATCH',
                    'Human_Value': 'NO MATCH',
                    'Error_Type': 'No matching row found',
                    'Error_Value': 'N/A',
                    'Is_Category': False,
                    'Total_Row_Errors': 'N/A'
                })

        return pd.DataFrame(report_data)

    def print_row_error_summary(self, evaluation_results: Dict, show_top_n: int = 5):
        """
        Print a summary of errors for each row

        Args:
            evaluation_results: Results from evaluate_extraction()
            show_top_n: Number of rows to show detailed errors for
        """
        print("\n" + "=" * 80)
        print("ROW-BY-ROW ERROR ANALYSIS")
        print("=" * 80)

        detailed_results = evaluation_results['detailed_results']
        matched_rows = detailed_results[detailed_results['Match_Found']]

        if len(matched_rows) == 0:
            print("No matched rows found for analysis.")
            return

        print(f"\nTotal matched rows with errors: {len(matched_rows[matched_rows['Row_Errors'] > 0])}")
        print(f"Perfect matches: {len(matched_rows[matched_rows['Row_Errors'] == 0])}")
        print(f"Unmatched rows: {len(detailed_results[~detailed_results['Match_Found']])}")

        # Sort by number of errors (descending)
        rows_with_errors = matched_rows[matched_rows['Row_Errors'] > 0].sort_values('Row_Errors', ascending=False)

        print(f"\nTop {show_top_n} rows with most errors:")
        print("-" * 80)

        for idx, row in rows_with_errors.head(show_top_n).iterrows():
            print(f"\nRow {int(row['LLM_Index'])} (DOI: {row['DOI']}):")
            print(f"  • Total errors: {row['Row_Errors']}")
            print(f"  • Accuracy: {row['Accuracy']:.2%}")
            print(f"  • Category errors: {row.get('Category_Errors', 'N/A')}")
            print(f"  • Numerical errors: {row.get('Numerical_Errors', 'N/A')}")

            # Get detailed errors for this row
            row_errors = [err for err in evaluation_results['row_detailed_errors']
                          if err['LLM_Index'] == row['LLM_Index'] and err['Match_Found']]

            if row_errors and row_errors[0]['Errors']:
                print(f"  • Error details:")
                for error in row_errors[0]['Errors'][:5]:  # Show first 5 errors
                    error_value_display = f"{error['error_value']:.4f}" if isinstance(error['error_value'],
                                                                                      (int, float)) and not np.isnan(
                        error['error_value']) else str(error['error_value'])
                    print(
                        f"    - {error['column']}: LLM='{error['llm_value']}' vs Human='{error['human_value']}' ({error['error_type']} = {error_value_display})")
                if len(row_errors[0]['Errors']) > 5:
                    print(f"    - ... and {len(row_errors[0]['Errors']) - 5} more errors")

        # Show rows with perfect matches
        perfect_rows = matched_rows[matched_rows['Row_Errors'] == 0]
        if len(perfect_rows) > 0:
            print(f"\nPerfect matches (0 errors): {len(perfect_rows)} rows")
            if len(perfect_rows) <= 10:
                print("  Rows:", ", ".join([str(int(idx)) for idx in perfect_rows['LLM_Index'].values]))

        # Error distribution by error count
        print(f"\nError count distribution:")
        error_counts = matched_rows['Row_Errors'].value_counts().sort_index()
        for error_count, row_count in error_counts.items():
            percentage = row_count / len(matched_rows) * 100
            print(f"  • {error_count} errors: {row_count} rows ({percentage:.1f}%)")

    def generate_visualization_data(self, evaluation_results: Dict) -> pd.DataFrame:
        """
        Generate data for visualization including accuracy, precision, recall, F1 and CO2 MAE

        Note: This method returns a DataFrame for backward compatibility
        """
        # Create dataframe for column accuracy and metrics
        accuracy_data = []
        column_metrics = evaluation_results.get('precision_recall_f1', {}).get('column_metrics', {})

        for col, stats in evaluation_results['column_accuracy'].items():
            col_metrics = column_metrics.get(col, {})
            accuracy_data.append({
                'Column': col,
                'Accuracy': col_metrics.get('accuracy', stats['accuracy_rate']),
                'Precision': col_metrics.get('precision', 0),
                'Recall': col_metrics.get('recall', 0),
                'F1_Score': col_metrics.get('f1_score', 0),
                'Error_Count': stats['errors'],
                'Unmatched_Count': col_metrics.get('unmatched', 0),
                'Total_Expected': col_metrics.get('total_expected', stats['total_checked']),
                'Total_Checked': stats['total_checked'],
                'Column_Type': 'Category' if col in self.category_columns else 'Numerical'
            })

        # Add CO2 MAE as a separate row if needed
        co2_mae = evaluation_results.get('co2_mae_metrics', {})
        if co2_mae and co2_mae.get('count', 0) > 0:
            accuracy_data.append({
                'Column': 'CO2_Capacity_mmol_g (MAE)',
                'Accuracy': co2_mae.get('mean_mae', 0),
                'Precision': 0,
                'Recall': 0,
                'F1_Score': 0,
                'Error_Count': co2_mae.get('count', 0),
                'Unmatched_Count': 0,
                'Total_Expected': co2_mae.get('count', 0),
                'Total_Checked': co2_mae.get('count', 0),
                'Column_Type': 'MAE Metric'
            })

        return pd.DataFrame(accuracy_data)

    def export_detailed_report(self, evaluation_results: Dict, output_path: str = 'llm_evaluation_report.xlsx'):
        """
        Export detailed evaluation report to Excel including accuracy, precision, recall, F1 and CO2 MAE

        Args:
            evaluation_results: Results from evaluate_extraction()
            output_path: Path to save the Excel file
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([evaluation_results['summary']])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # CO2 MAE sheet
            co2_mae = evaluation_results.get('co2_mae_metrics', {})
            if co2_mae:
                co2_df = pd.DataFrame([{
                    'Metric': 'Mean MAE',
                    'Value': co2_mae.get('mean_mae', 0)
                }, {
                    'Metric': 'Median MAE',
                    'Value': co2_mae.get('median_mae', 0)
                }, {
                    'Metric': 'Std MAE',
                    'Value': co2_mae.get('std_mae', 0)
                }, {
                    'Metric': 'Min MAE',
                    'Value': co2_mae.get('min_mae', 0)
                }, {
                    'Metric': 'Max MAE',
                    'Value': co2_mae.get('max_mae', 0)
                }, {
                    'Metric': 'Count',
                    'Value': co2_mae.get('count', 0)
                }])
                co2_df.to_excel(writer, sheet_name='CO2_MAE', index=False)

            # Precision-Recall-F1 sheet with accuracy
            prf_metrics = evaluation_results.get('precision_recall_f1', {})
            if prf_metrics:
                # Overall metrics
                overall_data = []
                overall = prf_metrics.get('overall', {})
                overall_data.append({
                    'Metric': 'Macro Avg Accuracy',
                    'Value': overall.get('macro_avg_accuracy', 0)
                })
                overall_data.append({
                    'Metric': 'Macro Avg Precision',
                    'Value': overall.get('macro_avg_precision', 0)
                })
                overall_data.append({
                    'Metric': 'Macro Avg Recall',
                    'Value': overall.get('macro_avg_recall', 0)
                })
                overall_data.append({
                    'Metric': 'Macro Avg F1',
                    'Value': overall.get('macro_avg_f1', 0)
                })
                overall_data.append({
                    'Metric': 'Weighted Avg Accuracy',
                    'Value': overall.get('weighted_avg_accuracy', 0)
                })
                overall_data.append({
                    'Metric': 'Weighted Avg Precision',
                    'Value': overall.get('weighted_avg_precision', 0)
                })
                overall_data.append({
                    'Metric': 'Weighted Avg Recall',
                    'Value': overall.get('weighted_avg_recall', 0)
                })
                overall_data.append({
                    'Metric': 'Weighted Avg F1',
                    'Value': overall.get('weighted_avg_f1', 0)
                })
                pd.DataFrame(overall_data).to_excel(writer, sheet_name='PRF_Overall', index=False)

                # By type metrics
                by_type = prf_metrics.get('by_type', {})
                type_data = []

                cat = by_type.get('categorical', {})
                if cat:
                    type_data.append({
                        'Type': 'Categorical',
                        'Accuracy': cat.get('accuracy', 0),
                        'Precision': cat.get('precision', 0),
                        'Recall': cat.get('recall', 0),
                        'F1': cat.get('f1_score', 0),
                        'TP': cat.get('total_tp', 0),
                        'FP': cat.get('total_fp', 0),
                        'FN': cat.get('total_fn', 0),
                        'Columns': cat.get('total_columns', 0)
                    })

                num = by_type.get('numerical', {})
                if num:
                    type_data.append({
                        'Type': 'Numerical',
                        'Accuracy': num.get('accuracy', 0),
                        'Precision': num.get('precision', 0),
                        'Recall': num.get('recall', 0),
                        'F1': num.get('f1_score', 0),
                        'TP': num.get('total_tp', 0),
                        'FP': num.get('total_fp', 0),
                        'FN': num.get('total_fn', 0),
                        'Columns': num.get('total_columns', 0)
                    })

                if type_data:
                    pd.DataFrame(type_data).to_excel(writer, sheet_name='PRF_By_Type', index=False)

            # Detailed results sheet
            evaluation_results['detailed_results'].to_excel(writer, sheet_name='Row_Results', index=False)

            # Column metrics sheet with precision, recall, F1, accuracy and unmatched
            column_data = []
            column_metrics = prf_metrics.get('column_metrics', {})
            for col, stats in evaluation_results['column_accuracy'].items():
                col_metrics = column_metrics.get(col, {})
                column_data.append({
                    'Column': col,
                    'Accuracy': col_metrics.get('accuracy', stats['accuracy_rate']),
                    'Precision': col_metrics.get('precision', 0),
                    'Recall': col_metrics.get('recall', 0),
                    'F1_Score': col_metrics.get('f1_score', 0),
                    'True_Positives': col_metrics.get('tp', 0),
                    'False_Positives': col_metrics.get('fp', 0),
                    'False_Negatives': col_metrics.get('fn', 0),
                    'Total_Matched': stats['total_checked'],
                    'Unmatched': col_metrics.get('unmatched', 0),
                    'Total_Expected': col_metrics.get('total_expected', stats['total_checked']),
                    'Errors': stats['errors'],
                    'Error_Rate': stats['errors'] / stats['total_checked'] if stats['total_checked'] > 0 else 0,
                    'Column_Type': 'Category' if col in self.category_columns else 'Numerical'
                })
            pd.DataFrame(column_data).to_excel(writer, sheet_name='Column_Metrics', index=False)

            # Row errors sheet
            error_report = self.get_row_error_report(evaluation_results, n_rows=1000)
            error_report.to_excel(writer, sheet_name='Detailed_Errors', index=False)

            # Error type distribution
            error_type_df = pd.DataFrame(list(evaluation_results['error_type_distribution'].items()),
                                         columns=['Error_Type', 'Count'])
            error_type_df.to_excel(writer, sheet_name='Error_Types', index=False)

            # Unmatched rows sheet
            unmatched_df = evaluation_results['detailed_results'][
                ~evaluation_results['detailed_results']['Match_Found']]
            if not unmatched_df.empty:
                unmatched_df.to_excel(writer, sheet_name='Unmatched_Rows', index=False)

        print(f"Report exported to {output_path}")


def create_visualizations(viz_data: pd.DataFrame):
    """
    Create visualizations for the evaluation results including accuracy, precision, recall, F1 and CO2 MAE

    Args:
        viz_data: DataFrame from generate_visualization_data()
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 14))

        # 1. Metrics by column type
        plt.subplot(3, 2, 1)
        # Filter out MAE metric row for this plot
        plot_data = viz_data[viz_data['Column_Type'] != 'MAE Metric']
        if not plot_data.empty:
            metrics_by_type = plot_data.groupby('Column_Type')[['Accuracy', 'Precision', 'Recall', 'F1_Score']].mean()
            metrics_by_type.plot(kind='bar', ax=plt.gca())
            plt.title('Average Metrics by Column Type')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.legend(loc='lower right')
            plt.xticks(rotation=0)

        # 2. Top 10 least accurate columns with all metrics
        plt.subplot(3, 2, 2)
        plot_data = viz_data[viz_data['Column_Type'] != 'MAE Metric']
        if not plot_data.empty:
            bottom_10 = plot_data.nsmallest(10, 'Accuracy')
            x = range(len(bottom_10))
            width = 0.2
            plt.bar([i - 1.5 * width for i in x], bottom_10['Accuracy'], width, label='Accuracy', color='lightcoral')
            plt.bar([i - 0.5 * width for i in x], bottom_10['Precision'], width, label='Precision', color='lightgreen')
            plt.bar([i + 0.5 * width for i in x], bottom_10['Recall'], width, label='Recall', color='skyblue')
            plt.bar([i + 1.5 * width for i in x], bottom_10['F1_Score'], width, label='F1', color='orange')
            plt.xlabel('Columns')
            plt.ylabel('Score')
            plt.title('Top 10 Least Accurate Columns - All Metrics')
            plt.xticks(x, bottom_10['Column'], rotation=45, ha='right')
            plt.legend(loc='lower right')
            plt.ylim(0, 1)

        # 3. Distribution of metrics
        plt.subplot(3, 2, 3)
        plot_data = viz_data[viz_data['Column_Type'] != 'MAE Metric']
        if not plot_data.empty:
            plot_data[['Accuracy', 'Precision', 'Recall', 'F1_Score']].hist(bins=20, alpha=0.7, ax=plt.gca())
            plt.title('Distribution of Metrics')
            plt.xlabel('Score')
            plt.ylabel('Frequency')

        # 4. Correlation heatmap
        plt.subplot(3, 2, 4)
        plot_data = viz_data[viz_data['Column_Type'] != 'MAE Metric']
        if not plot_data.empty:
            corr = plot_data[['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Error_Count', 'Unmatched_Count']].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=plt.gca())
            plt.title('Correlation Between Metrics')

        # 5. Error count vs F1 score with unmatched size
        plt.subplot(3, 2, 5)
        plot_data = viz_data[viz_data['Column_Type'] != 'MAE Metric']
        if not plot_data.empty:
            colors = ['blue' if t == 'Category' else 'red' for t in plot_data['Column_Type']]
            sizes = plot_data['Unmatched_Count'] * 50 + 50  # Scale for visibility
            plt.scatter(plot_data['Error_Count'], plot_data['F1_Score'],
                        c=colors, s=sizes, alpha=0.6)
            plt.xlabel('Error Count')
            plt.ylabel('F1 Score')
            plt.title('Error Count vs F1 Score (size=unmatched)')
            plt.grid(True, alpha=0.3)

            # Add legend for column types
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='blue', alpha=0.6, label='Category'),
                               Patch(facecolor='red', alpha=0.6, label='Numerical')]
            plt.legend(handles=legend_elements, loc='lower right')

        # 6. CO2 MAE histogram if available
        plt.subplot(3, 2, 6)
        co2_row = viz_data[viz_data['Column'] == 'CO2_Capacity_mmol_g (MAE)']
        if not co2_row.empty:
            plt.text(0.5, 0.5, f"CO2 MAE: {co2_row['Accuracy'].values[0]:.4f} mmol/g",
                     ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title('CO2 Capacity MAE')
            plt.axis('off')
        else:
            plt.text(0.5, 0.5, "No CO2 MAE data available",
                     ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('CO2 Capacity MAE')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        # 7. Detailed metrics plot for all columns
        plt.figure(figsize=(14, 10))
        plot_data = viz_data[viz_data['Column_Type'] != 'MAE Metric']
        if not plot_data.empty:
            viz_data_sorted = plot_data.sort_values('F1_Score')

            # Create bar plot for all metrics
            x = range(len(viz_data_sorted))
            width = 0.2

            plt.barh([i - 1.5 * width for i in x], viz_data_sorted['Accuracy'], width, label='Accuracy',
                     color='lightcoral')
            plt.barh([i - 0.5 * width for i in x], viz_data_sorted['Precision'], width, label='Precision',
                     color='lightgreen')
            plt.barh([i + 0.5 * width for i in x], viz_data_sorted['Recall'], width, label='Recall', color='skyblue')
            plt.barh([i + 1.5 * width for i in x], viz_data_sorted['F1_Score'], width, label='F1', color='orange')

            plt.yticks(x, viz_data_sorted['Column'])
            plt.xlabel('Score')
            plt.title('All Metrics for Each Column (sorted by F1)')
            plt.xlim(0, 1)
            plt.grid(axis='x', alpha=0.3)
            plt.legend(loc='lower right')

            # Add vertical lines for thresholds
            plt.axvline(x=0.5, color='red', linestyle=':', alpha=0.5, label='50% threshold')
            plt.axvline(x=0.8, color='orange', linestyle=':', alpha=0.5, label='80% threshold')

            plt.tight_layout()
            plt.show()

    except ImportError:
        print("Matplotlib and/or seaborn not installed. Install with: pip install matplotlib seaborn")


# Example usage with all the features
def run_complete_evaluation():
    """Run a complete evaluation with all features including CO2 MAE and unmatched rows"""

    # Create sample data
    category_columns = [
        "DOI",
        "Support",
        "Amine_1_or_Additive_1",
        "Amine_2_or_Additive_2",
        "Amine_3_or_Additive_3",
        "MW_Mn_g_mol",
        "Organic_Content_pct",
        "Relative_Humidity_pct",
        "CO2_Concentration_vol_pct",
    ]

    numerical_columns = [
        "CO2_Capacity_mmol_g",
        "Amine_Efficiency_mmol_mmol",
        "Heat_of_Adsorption_kJ_mol"
    ]

    # Sample human data
    human_data = pd.DataFrame({
        'DOI': ['10.1021/acs.iecr.9b00576', '10.1021/ie300257j', '10.1002/sus2.141'],
        'Support': ['MCM-48', 'M4', 'SBA-15'],
        'Amine_1_or_Additive_1': ['PEI', 'AM-TEPA', 'TEPA'],
        'Amine_2_or_Additive_2': ['', '', 'DEA'],
        'Amine_3_or_Additive_3': ['', '', ''],
        'MW_Mn_g_mol': ['600', '189.3', '189.0'],
        'Organic_Content_pct': ['40', '48.92', '50'],
        'CO2_Capacity_mmol_g': [2.16, 2.93, 2.27],
        'Amine_Efficiency_mmol_mmol': [0.22, 0.22, 0.22],
        'Heat_of_Adsorption_kJ_mol': [61.0, 0.0, 0.0],
        'Relative_Humidity_pct': ['0', '0', '0'],
        'CO2_Concentration_vol_pct': ['100', '100', '0.04']
    })

    # Sample LLM data with errors and an unmatched row
    llm_data = pd.DataFrame({
        'DOI': ['10.1021/acs.iecr.9b00576', '10.1021/acs.iecr.9b00576', '10.1021/ie300257j', '10.1002/sus2.141',
                '10.9999/unmatched'],
        'Support': ['MCM-48', 'MCM-48', 'M4', 'SBA-15', 'Unknown'],
        'Amine_1_or_Additive_1': ['PEI', 'PEI', 'AM-TEPA', 'TEPA', 'Unknown'],
        'Amine_2_or_Additive_2': ['', '', '', 'DEA', ''],
        'Amine_3_or_Additive_3': ['', '', '', '', ''],
        'MW_Mn_g_mol': ['600', '600', '189.3', '189.0', '500'],
        'Organic_Content_pct': ['42', '37', '48.92', '50', '45'],
        'CO2_Capacity_mmol_g': [
            '[2.16, 1.01, 2.15]',  # Multiple values, one matches (MAE will be small)
            2.59,  # Single value, doesn't match human (MAE = |2.59-2.16| = 0.43)
            2.9266,  # Close match (MAE = |2.9266-2.93| = 0.0034)
            '[1.62, 1.54, 2.27, 1.6, 1.4, 2.05, 1.63, 1.42]',  # Multiple values, one matches
            '[2.5, 2.6, 2.7]'  # Unmatched row, will be counted in MAE as well
        ],
        'Amine_Efficiency_mmol_mmol': [0.22, 0.30, 0.22, 0.22, 0.25],
        'Heat_of_Adsorption_kJ_mol': [61.0, 60.0, 0.0, 0.0, 65.0],
        'Relative_Humidity_pct': ['0', '0', '0', '0', '0'],
        'CO2_Concentration_vol_pct': ['100', '100', '100', '0.04', '100']
    })

    print("Initializing evaluator...")
    evaluator = LLMExtractionEvaluator(llm_data, human_data, category_columns, numerical_columns)

    print("\nRunning evaluation...")
    results = evaluator.evaluate_extraction(include_unmatched_in_metrics=True)

    print("\n" + "=" * 80)
    print("BASIC SUMMARY")
    print("=" * 80)
    summary = results['summary']
    print(f"Total LLM rows: {summary['total_llm_rows']}")
    print(f"Matched rows: {summary['matched_rows']}")
    print(f"Unmatched rows: {summary['unmatched_rows']}")
    print(f"Match rate: {summary['match_rate']:.2%}")
    print(f"Overall accuracy (matched rows): {summary['overall_accuracy']:.2%}")
    print(f"Perfect matches: {summary['perfect_matches']}")
    print(f"Average row accuracy: {summary['avg_row_accuracy']:.2%}")

    print("\n" + "=" * 80)
    print("CO2 CAPACITY MAE METRICS")
    print("=" * 80)
    co2_mae = results.get('co2_mae_metrics', {})
    if co2_mae:
        print(f"Mean MAE: {co2_mae.get('mean_mae', 0):.4f} mmol/g")
        print(f"Median MAE: {co2_mae.get('median_mae', 0):.4f} mmol/g")
        print(f"Std MAE: {co2_mae.get('std_mae', 0):.4f} mmol/g")
        print(f"Min MAE: {co2_mae.get('min_mae', 0):.4f} mmol/g")
        print(f"Max MAE: {co2_mae.get('max_mae', 0):.4f} mmol/g")
        print(f"Count: {co2_mae.get('count', 0)}")

    print("\n" + "=" * 80)
    print("PRECISION, RECALL, F1 SCORES WITH ACCURACY (including unmatched rows)")
    print("=" * 80)
    prf_metrics = results.get('precision_recall_f1', {})
    if prf_metrics:
        overall = prf_metrics.get('overall', {})
        print(f"Macro Avg Accuracy: {overall.get('macro_avg_accuracy', 0):.2%}")
        print(f"Macro Avg Precision: {overall.get('macro_avg_precision', 0):.2%}")
        print(f"Macro Avg Recall: {overall.get('macro_avg_recall', 0):.2%}")
        print(f"Macro Avg F1: {overall.get('macro_avg_f1', 0):.2%}")
        print(f"Weighted Avg Accuracy: {overall.get('weighted_avg_accuracy', 0):.2%}")
        print(f"Weighted Avg Precision: {overall.get('weighted_avg_precision', 0):.2%}")
        print(f"Weighted Avg Recall: {overall.get('weighted_avg_recall', 0):.2%}")
        print(f"Weighted Avg F1: {overall.get('weighted_avg_f1', 0):.2%}")

    print("\n" + "=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)
    evaluator.print_detailed_statistics(results)

    print("\n" + "=" * 80)
    print("ROW ERROR SUMMARY")
    print("=" * 80)
    evaluator.print_row_error_summary(results, show_top_n=3)

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    viz_data = evaluator.generate_visualization_data(results)
    create_visualizations(viz_data)

    print("\n" + "=" * 80)
    print("DETAILED ERROR REPORT (first 10 rows)")
    print("=" * 80)
    error_report = evaluator.get_row_error_report(results, n_rows=10)
    print(error_report.to_string())

    print("\n" + "=" * 80)
    print("EXPORTING REPORT")
    print("=" * 80)
    evaluator.export_detailed_report(results, 'llm_evaluation_report.xlsx')

    print("\n" + "=" * 80)
    print("ACCESSING SPECIFIC METRICS")
    print("=" * 80)
    print(f"Perfect matches: {results['summary']['perfect_matches']}")
    print(f"Average row accuracy: {results['summary']['avg_row_accuracy']:.2%}")
    print(f"Median row accuracy: {results['summary']['median_row_accuracy']:.2%}")
    print(f"Total category errors: {results['summary']['total_category_errors']}")
    print(f"Total numerical errors: {results['summary']['total_numerical_errors']}")
    print(f"Match rate: {results['summary']['match_rate']:.2%}")
    print(f"CO2 Mean MAE: {results['co2_mae_metrics'].get('mean_mae', 0):.4f} mmol/g")

    # Show column-specific metrics including unmatched counts
    print("\nColumn metrics details (including unmatched rows):")
    column_metrics = results.get('precision_recall_f1', {}).get('column_metrics', {})
    for col, stats in results['column_accuracy'].items():
        col_metrics = column_metrics.get(col, {})
        print(f"  {col}:")
        print(f"    - Accuracy: {col_metrics.get('accuracy', 0):.2%}")
        print(f"    - Precision: {col_metrics.get('precision', 0):.2%}")
        print(f"    - Recall: {col_metrics.get('recall', 0):.2%}")
        print(f"    - F1: {col_metrics.get('f1_score', 0):.2%}")
        print(f"    - TP: {col_metrics.get('tp', 0)}, FP: {col_metrics.get('fp', 0)}, FN: {col_metrics.get('fn', 0)}")
        print(
            f"    - Correct: {stats['correct']}/{stats['total_checked']} (matched), Unmatched: {col_metrics.get('unmatched', 0)}")

    return evaluator, results, viz_data


if __name__ == "__main__":
    print("LLM DATA EXTRACTION EVALUATION SYSTEM")
    print("=" * 80)
    print("Updated with:")
    print(
        "  • Material columns: Support, Amine_1_or_Additive_1, Amine_2_or_Additive_2, Amine_3_or_Additive_3, Organic_Content_pct")
    print("  • CO2_Capacity_mmol_g evaluated with MAE")
    print("  • Unmatched rows included in precision/recall/F1 as false negatives")
    print("=" * 80)

    evaluator, results, viz_data = run_complete_evaluation()

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("Report exported to 'llm_evaluation_report.xlsx'")
    print("Visualizations displayed above")