import argparse
import numpy as np
import random
import glob
import os
import re
import json
import logging
import pandas as pd
import dill as pickle


def valid_string(values):
    return f"Valid choices are: {list(values)}"


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def save_results(results: dict, results_dir: str):
    log_file_name = f"{results['dataset']}"
    rho = str(results["dataset_kwargs"]["rho"]) if "rho" in results["dataset_kwargs"] else "na"
    save_name = f"{results_dir}/{log_file_name}_{rho}.log"
    logging.info("Saving results in %s", save_name)
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    with open(save_name, "w") as f:
        f.write(json.dumps(results, indent=4))


def save_results_csv(results: dict, results_dir: str):
    log_file_name = f"{results['dataset']}"
    rho = str(results["dataset_kwargs"]["rho"]) if "rho" in results["dataset_kwargs"] else "na"
    save_name = f"{results_dir}/csv/{log_file_name}_{rho}.csv"
    logging.info("Saving results in %s", save_name)
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    with open(save_name, "w") as f:
        for model in results["models"]:
            f.write(str(model).upper() + "," + ",".join(results["model_perfs"][model]) + "\n")
            df = pd.read_json(json.dumps(results["models"][model]))
            df = df.transpose()
            f.write(df.to_csv())
            f.write("\n\n")


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def save_experiment(experiment, checkpoint_dir: str, rho):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    log_file_name = f"{experiment.dataset.name}"
    save_name = f"{checkpoint_dir}/{log_file_name}_{rho}.pkl"
    save_object(experiment, save_name)
