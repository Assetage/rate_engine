from catboost import CatBoostRegressor
from utils import calculate_mape

def load_model():
    model = CatBoostRegressor()
    model.load_model("cb_model")
    return model

def train_model(X_train, y_train, X_val, y_val, config):
    model = CatBoostRegressor(**config['best_params'], cat_features=config['cat_features'])
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)
    return model

def evaluate_model(X_val, y_val, model):
    predicted_rates = model.predict(X_val)
    mape = calculate_mape(y_val, predicted_rates)
    return predicted_rates, mape