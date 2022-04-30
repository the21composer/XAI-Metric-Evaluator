from metrics import available_metrics


class Metric:
    def __init__(self, name, **kwargs):
        if name not in available_metrics.keys():
            return
        self.name = name
        # decorator to init metric
        self.metric = lambda model, data_class: available_metrics[name](model, data_class, **kwargs)
        # decorator to evaluate metric
        self.evaluate = available_metrics[name].evaluate
