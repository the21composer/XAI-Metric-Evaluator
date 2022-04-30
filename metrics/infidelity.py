import numpy as np


def get_exp(ind, exp):
    return exp[ind.astype(int)]


def set_zero_infidelity(array, size, point):
    ind = np.random.choice(size, point, replace=False)
    rand_noise = np.random.normal(size=point) * 0.2 + array[ind]
    rand_noise = np.minimum(array[ind], rand_noise)
    rand_noise = np.maximum(array[ind] - 1, rand_noise)
    array[ind] -= rand_noise
    return np.concatenate([array, ind, rand_noise])


class Infidelity:
    def __init__(self, model, dataset=None, **kwargs):
        self.model = model
        self.dataset = dataset

    def evaluate(self, x, y, weights, true_weights, avg=True, x_train=None, y_train=None, n_sample=100,
                 x_train_weights=None):
        x = x.values
        num_elements, num_features = x.shape
        infidelity_list = []

        for i in range(num_elements):
            x_orig = np.tile(x[i], [n_sample, 1])
            weight = np.copy(weights[i])
            # generating affected dataset with perturbations
            val = np.apply_along_axis(set_zero_infidelity, 1, x_orig, num_features, num_features)
            x_new, ind, rand = val[:, :num_features], \
                               val[:, num_features: 2 * num_features], \
                               val[:, 2 * num_features: 3 * num_features]
            exp_sum = np.sum(rand * np.apply_along_axis(get_exp, 1, ind, weight), axis=1)
            ks = np.ones(n_sample)
            y_predict = self.model.predict([x[i]])
            y_affected_predict = self.model.predict(x_new)
            diff = y_predict - y_affected_predict

            beta = np.mean(ks * diff * exp_sum) / np.mean(ks * exp_sum * exp_sum)
            exp_sum *= beta
            infidelity_list.append(np.mean(ks * np.square(diff - exp_sum)) / np.mean(ks))

        return np.mean(infidelity_list)
