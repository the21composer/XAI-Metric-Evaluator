import shap


class ShapXAI:
    def __init__(self, f, x):
        self.f = f
        self.x = x
        self.explainer = shap.Explainer(self.f, self.x)
        self.expected_values = []

    def explain(self, x):
        shap_values = self.explainer(x)
        self.expected_values, shap_values = shap_values.base_values, shap_values.values
        return shap_values


class KernelShap:
    def __init__(self, f, x, **kwargs):
        self.f = f
        self.x = x
        self.explainer = shap.KernelExplainer(self.f, self.x, **kwargs)
        self.expected_values = []

    def explain(self, x):
        shap_values = self.explainer(x)
        self.expected_values, shap_values = shap_values.base_values, shap_values.values
        return shap_values
