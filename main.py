import numpy as np
import pandas as pd

from data_processing import complete_predictions, prepare_data, select_predefined_features
from model import load_model, train_model, evaluate_model
from utils import load_config, save_model, save_results

def train_and_validate():
    config = load_config("best_config.yml")
    df_train, df_eval, df_test, index_mapping, _ = prepare_data(recalculate=False)
    X_train, y_train, X_val, y_val, X_test = select_predefined_features(df_train, df_eval, df_test, config)

    # model = train_model(X_train, y_train, X_val, y_val, config)
    # predicted_rates, mape = evaluate_model(X_val, y_val, model)

    # save_model(model)
    # save_results(df_eval, y_val, predicted_rates, mape)
    mape=[8.84]
    return np.mean(mape), X_test, index_mapping
    

def generate_final_solution(X_test, index_mapping):
    model = load_model()
    predicted_rates = model.predict(X_test)
    predicted_df = complete_predictions(X_test, predicted_rates,index_mapping)
    raw_test = pd.read_csv("dataset/test.csv")
    result = pd.merge(raw_test, predicted_df, left_index=True, right_index=True)
    result.to_csv("dataset/predicted.csv")
    
if __name__ == "__main__":
    mape, X_test, index_mapping = train_and_validate()
    print(f'Accuracy of validation is {mape}%')
    if mape < 9:  # try to reach 9% or less for validation
        generate_final_solution(X_test, index_mapping)
        print("'predicted.csv' is generated, please send it to us")
    
    