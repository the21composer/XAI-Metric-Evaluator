from xai_methods import available_methods


class Explainer:
    def __init__(self, name, **kwargs):
        if name not in available_methods.keys():
            return
        self.name = name
        # decorator to call explainer
        self.explainer = lambda clf, data: available_methods[name](clf, data, **kwargs)
