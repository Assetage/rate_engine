import yaml
import numpy as np
from catboost import CatBoostRegressor
from utils import save_results, save_model, calculate_mape, load_config
from data_preprocessing import prepare_data

DF_TRAIN_PATH = 'dataset/train.csv'
DF_EVAL_PATH = 'dataset/validation.csv'

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
    # combine train and validation to improve final predictions
    df = pd.read_csv('dataset/train.csv')
    df_val = pd.read_csv('dataset/validation.csv')
    df = df.append(df_val).reset_index(drop=True)

    model = Model()
    model.fit(df, df.rate)

    # generate and save test predictions
    df_test = pd.read_csv('dataset/test.csv')
    df_test['predicted_rate'] = model.predict(df_test)
    df_test.to_csv('dataset/predicted.csv', index=False)

if __name__ == "__main__":
    mape = train_and_validate()
    print(f'Accuracy of validation is {mape}%')