import numpy as np


class Monotonicity:
    def __init__(self, model, dataset, **kwargs):
        self.model = model
        self.dataset = dataset

    def evaluate(self, x, y, weights, true_weights, x_train=None, y_train=None, n_sample=100,
                 x_train_weights=None):
        x = x.values
        num_elements, num_features = x.shape

        monotonicity_list = []
        y_predict = np.mean(np.squeeze(self.model.predict(x)))

        for i in range(num_elements):
            # generate zero mask and sort weights in ascending order
            mask = np.zeros_like(x[i])
            ascending_weights_idx = np.argsort(abs(weights)[i])
            # current prediction for dataset element
            y_predict_list = np.zeros(len(x[i]) + 1)
            y_predict_list[0] = y_predict

            # enable feature one by one and save predictions
            for j in ascending_weights_idx:
                mask[j] = 1
                x_sampled, _ = self.dataset.generate(mask=mask, x=x[i], n_sample=n_sample)
                y_predict_list[j + 1] = np.mean(np.squeeze(self.model.predict(x_sampled)))

            # 1 if all deltas are >= 0, 0 if not
            monotonicity_list.append(int(np.all(np.diff(y_predict_list)) >= 0))
            # try: int(np.all(np.diff(np.abs(np.diff(y_predict_list))) >= 0))

        return np.mean(monotonicity_list)

