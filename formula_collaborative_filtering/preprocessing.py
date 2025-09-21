import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self):
        self.formula_imputer = None
        self.condition_imputer = None
        self.formula_scaler = None
        self.condition_scaler = None

    def preprocess_formula_data(self, formula_data):
        """Preprocess formula data: handle missing values and scale"""
        # Create imputer and scaler if not exists
        if self.formula_imputer is None:
            self.formula_imputer = SimpleImputer(strategy='mean')
            formula_data_imputed = self.formula_imputer.fit_transform(formula_data)
        else:
            formula_data_imputed = self.formula_imputer.transform(formula_data)

        if self.formula_scaler is None:
            self.formula_scaler = StandardScaler()
            formula_data_scaled = self.formula_scaler.fit_transform(formula_data_imputed)
        else:
            formula_data_scaled = self.formula_scaler.transform(formula_data_imputed)

        return formula_data_scaled

    def preprocess_condition_data(self, condition_data):
        """Preprocess condition data: handle missing values and scale"""
        # Create imputer and scaler if not exists
        if self.condition_imputer is None:
            self.condition_imputer = SimpleImputer(strategy='mean')
            condition_data_imputed = self.condition_imputer.fit_transform(condition_data)
        else:
            condition_data_imputed = self.condition_imputer.transform(condition_data)

        if self.condition_scaler is None:
            self.condition_scaler = StandardScaler()
            condition_data_scaled = self.condition_scaler.fit_transform(condition_data_imputed)
        else:
            condition_data_scaled = self.condition_scaler.transform(condition_data_imputed)

        return condition_data_scaled