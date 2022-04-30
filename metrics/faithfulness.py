import numpy as np


class Faithfulness:
    def __init__(self, model, dataset, **kwargs):
        self.model = model
        self.dataset = dataset

    def evaluate(self, x, y, weights, true_weights, x_train=None, y_train=None, n_sample=100,
                 x_train_weights=None):
        x = x.values
        num_elements, num_features = x.shape

        faithfulness_list = []

        for i in range(num_elements):
            # current prediction for dataset element
            y_predict = np.squeeze(self.model.predict(np.array([x[i]])))

            # generating predictions with changed features
            y_affected_predictions = np.zeros_like(x[i])
            for j in range(num_features):
                # generate a mask and "disable" only one feature
                mask = np.ones_like(x[i])
                mask[j] = 0
                # generate dataset with "disabled" feature
                x_sampled, _ = self.dataset.generate(mask=mask, x=x[i], n_sample=n_sample)
                # get predictions for affected dataset
                y_affected_predictions[j] = np.mean(np.squeeze(self.model.predict(x_sampled)))

            # calculate correlation between predictions diff and weights of the features
            faithfulness = np.corrcoef(
                abs(weights)[i],
                [abs(y_predict - y_affected_predictions[j]) for j in range(num_features)]
            )[0, 1]
            if np.isnan(faithfulness) or not np.isfinite(faithfulness):
                faithfulness = 0
            faithfulness_list.append(faithfulness)

        return np.mean(faithfulness_list)
