import pandas as pd
import numpy as np


class PropertyDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.formula_columns = None
        self.property_columns = None

    def load_data(self):
        """Load data from CSV file"""
        self.data = pd.read_csv(self.file_path)
        return self.data

    def get_formula_data(self):
        if self.data is None:
            self.load_data()

        # Formula columns are from Formula_1 to Formula_35
        formula_cols = [col for col in self.data.columns if 'Formula' in col]
        self.formula_columns = formula_cols
        return self.data[formula_cols]
    def get_condition_data(self):
        if self.data is None:
            self.load_data()

        # Property columns are the last 3 columns
        condition_cols = [col for col in self.data.columns if 'Condition' in col]
        self.property_columns = condition_cols
        return self.data[condition_cols]
    def get_property_data(self):
        """Extract property features (last 3 columns)"""
        if self.data is None:
            self.load_data()

        # Property columns are the last 3 columns
        property_cols = [col for col in self.data.columns if 'Property' in col]
        self.property_columns = property_cols
        return self.data[property_cols]

    def get_full_data(self):
        """Return the complete dataset"""
        if self.data is None:
            self.load_data()
        return self.data