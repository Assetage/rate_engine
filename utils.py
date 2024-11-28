import pandas as pd
import numpy as np
import yaml

def load_data(file_path):
    return pd.read_csv(file_path)

def save_results(df_eval, eval_target, predicted_rates, mape):
    results = df_eval.copy()
    results["actual_rate"] = eval_target
    results["predicted_rate"] = predicted_rates
    results["mape"] = mape
    results.to_csv("dataset/eval_results.csv", index=False)

def save_model(model):
    model.save_model('cb_model')

def calculate_mape(real_rates, predicted_rates):
    return np.abs((real_rates - predicted_rates) / real_rates) * 100

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        for k,v in config["best_params"].items():
            if k in ["depth","iterations","min_data_in_leaf","early_stopping_rounds"]:
                config["best_params"][k]=int(v)
            elif k in ["l2_leaf_reg","learning_rate","rsm"]:
                config["best_params"][k]=float(v)
    return config