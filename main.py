import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from utils import save_results, save_model, calculate_mape, load_config
from data_preprocessing import prepare_data, create_features, split_data, preprocess_data

DF_TRAIN_PATH = 'dataset/train.csv'
DF_EVAL_PATH = 'dataset/validation.csv'
DF_TEST_PATH = 'dataset/test.csv'

def train_and_validate():
    # Load the configuration
    config = load_config("best_config.yml")

    # Load and preprocess data
    df_train, train_target, df_eval, eval_target, _ = prepare_data(DF_TRAIN_PATH, DF_EVAL_PATH)

    # Select features
    selected_features = config['selected_features']
    X_train = df_train[selected_features]
    y_train = train_target
    X_val = df_eval[selected_features]
    y_val = eval_target

    # Train model
    model = CatBoostRegressor(**config['best_params'], cat_features=config['cat_features'])
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)

    # Predict and evaluate
    predicted_rates = model.predict(X_val)
    mape = calculate_mape(y_val, predicted_rates)

    # Save results
    save_model(model)
    save_results(df_eval, eval_target, predicted_rates, mape)

    return np.mean(mape)

def generate_final_solution():
    config = load_config("best_config.yml")
    df_train = pd.read_csv(DF_TRAIN_PATH)
    df_eval = pd.read_csv(DF_EVAL_PATH)
    df_test = pd.read_csv(DF_TEST_PATH)
    df_train["source"] = "train"
    df_eval["source"] = "eval"
    df_test["source"] = "test"
    
    # Create a list of original indices for the test set
    original_test_indices = list(range(len(df_test)))
    
    # Add a temporary column with the original indices
    df_test['temp_original_index'] = original_test_indices
    
    df_combined = pd.concat([df_train, df_eval,df_test], axis=0, ignore_index=True).sort_values('pickup_date')
    cat_features = ["transport_type", "origin_kma", "destination_kma"]
    df_combined = create_features(df_combined, cat_features)
    df_train, df_test = split_data(df_combined, enable_test=True)
    
    df_test, _, cat_features = preprocess_data(df_test,enable_test=True)
    
    # Select features
    selected_features = config['selected_features']
    X_test = df_test[selected_features]

    # Train model
    model = CatBoostRegressor()
    model.load_model("cb_model")
    
    # Predict and evaluate
    predicted_rates = model.predict(X_test)
    predictions_df = pd.DataFrame({'predicted_rate': predicted_rates}, index=X_test.index)

    # generate and save test predictions
    df_test = pd.read_csv(DF_TEST_PATH).reset_index()
    df_test = pd.merge(df_test, predictions_df, left_on="index",right_on="temp_original_index", how="left")
    df_test = df_test.set_index("index")
    df_test.to_csv('dataset/predicted.csv', index=False)

if __name__ == "__main__":
    mape = train_and_validate()
    print(f'Accuracy of validation is {mape}%')
    if mape < 9:  # try to reach 9% or less for validation
        generate_final_solution()
        print("'predicted.csv' is generated, please send it to us")