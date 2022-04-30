from datasets import available_datasets
import datasets
import numpy as np


class Data:
    def __init__(self, name: str, mode: str, **kwargs):
        if name not in available_datasets[mode].keys():
            raise NotImplementedError()
        self.name = name
        self.kwargs = kwargs
        data_kwargs = {k: eval(str(v)) for k, v in kwargs.items()}
        self.mode = mode
        self.data = available_datasets[mode][name](**data_kwargs)
        self.data_class = None
        if isinstance(self.data, datasets.CustomDataset):
            self.data_class = (
                self.data
            )
            self.data = self.data.get_dataset()
            self.val_data = self.data_class.get_dataset(self.data_class.num_val_samples)
            self.data[0].columns = [f"feat_{i}" for i in range(self.data[0].shape[1])]
            self.val_data[0].columns = [
                f"feat_{i}" for i in range(self.val_data[0].shape[1])
            ]
