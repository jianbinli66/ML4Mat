from sklearn.decomposition import PCA


class PCAReducer:
    def __init__(self, n_components=0.95):
        self.n_components = n_components
        self.formula_pca = None
        self.condition_pca = None
        self.formula_explained_variance_ = None
        self.condition_explained_variance_ = None

    def fit_transform_formula(self, formula_data):
        """Apply PCA to formula data"""
        self.formula_pca = PCA(n_components=self.n_components)
        formula_reduced = self.formula_pca.fit_transform(formula_data)
        self.formula_explained_variance_ = self.formula_pca.explained_variance_ratio_
        return formula_reduced

    def transform_formula(self, formula_data):
        """Transform new formula data using fitted PCA"""
        if self.formula_pca is None:
            raise ValueError("PCA must be fitted first")
        return self.formula_pca.transform(formula_data)

    def fit_transform_condition(self, property_data):
        """Apply PCA to condition data"""
        self.condition_pca = PCA(n_components=self.n_components)
        property_reduced = self.condition_pca.fit_transform(property_data)
        self.condition_explained_variance_ = self.condition_pca.explained_variance_ratio_
        return property_reduced

    def transform_condition(self, condition_data):
        """Transform new condition data using fitted PCA"""
        if self.condition_pca is None:
            raise ValueError("PCA must be fitted first")
        return self.condition_pca.transform(condition_data)