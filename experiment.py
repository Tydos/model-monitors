from itertools import product
import yaml
import copy
from train import train_script
import logging
logging.basicConfig(level=logging.INFO)

def load_base_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        return config

def parameter_sweep(config):
    model_params= {
            "max_depth": list(range(1, 10, 2)),
            # "min_samples_split": list(range(2, 10, 2)),
            # "min_samples_leaf": list(range(1, 10, 2)),
            # "criterion": ["gini", "entropy"],
            # "max_features": ["sqrt", "log2"]
        }

    hyperparameters = list(model_params.keys())
    values = list(model_params.values())

    combinations = list(product(*values))
    print(f"Total combinations: {len(combinations)}")

    for combo_id in combinations:
        # Convert tuple to dict
        combo_dict = dict(zip(hyperparameters, combo_id))

        # Copy base config
        config_run = copy.deepcopy(config)

        # Update model_params section in the YAML
        config_run['model_params'].update(combo_dict)

        # Save updated config to a temp YAML
        with open("config_run.yaml", "w") as f:
            yaml.safe_dump(config_run, f)

        #invoke train script
        logging.info(f"Running Experiment: {combo_id}")
        if train_script(config_run):
            print("Training completed successfully for this combination.")
        else:
            print("Training failed for this combination.")

if( __name__ == "__main__"):
    load_base_config()
    config = load_base_config()
    parameter_sweep(config)