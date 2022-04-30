import numpy as np
from tqdm import tqdm
import lime.lime_tabular


class LimeTabular:
    def __init__(self, model, data, mode="classification", kernel_width=0.75):
        self.model = model
        assert mode in ["classification", "regression"]
        self.mode = mode

        if str(type(data)).endswith("pandas.core.frame.DataFrame'>"):
            data = data.values
        self.data = data
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            data, mode=mode, kernel_width=kernel_width * np.sqrt(data.shape[-1])
        )

        out = self.model(data[0:1])
        if len(out.shape) == 1:
            self.out_dim = 1
            self.flat_out = True
            if mode == "classification":
                def pred(x):
                    predictions = self.model(x).reshape(-1, 1)
                    p0 = 1 - predictions
                    return np.hstack((p0, predictions))

                self.model = pred
        else:
            self.out_dim = self.model(data[0:1]).shape[1]
            self.flat_out = False

    def explain(self, x, num_features=None):
        num_features = x.shape[1] if num_features is None else num_features

        if str(type(x)).endswith("pandas.core.frame.DataFrame'>"):
            x = x.values

        out = [np.zeros(x.shape) for j in range(self.out_dim)]
        for i in tqdm(range(x.shape[0])):
            exp = self.explainer.explain_instance(
                x[i], self.model, labels=range(self.out_dim), num_features=num_features
            )
            for j in range(self.out_dim):
                for k, v in exp.local_exp[j]:
                    out[j][i, k] = v

        if self.mode == "regression":
            for i in range(len(out)):
                out[i] = -out[i]

        return out[0] if self.flat_out else out


class LimeXAI:
    def __init__(self, f, x, **kwargs):
        self.f = f
        self.x = x
        self.explainer = LimeTabular(self.f, self.x, mode="regression", **kwargs)
        self.expected_values = []

    def explain(self, x):
        shap_values = self.explainer.explain(x)
        self.expected_values = np.zeros(x.shape[0])
        return shap_values
