import numpy as np
import yaml
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from catboost import CatBoostRegressor


from data_preprocessing import prepare_data, time_based_split
from feature_engineering import analyze_feature_importance, select_features

DF_TRAIN_PATH = 'dataset/train.csv'
DF_EVAL_PATH = 'dataset/validation.csv'

def random_search_tuning(df_train, train_target, df_eval, eval_target, cat_features, n_iter=5, n_splits=5):
    # Time-based split for cross-validation
    splits = time_based_split(df_train, n_splits=n_splits)
    
    # Feature importance analysis
    base_model = CatBoostRegressor(learning_rate=0.03, 
                                   random_seed=42, 
                                   iterations=1000,
                                   depth=6,
                                   l2_leaf_reg=8,
                                   early_stopping_rounds=200)
    feature_importance = analyze_feature_importance(base_model, df_train, train_target, splits, cat_features)
    
    # Select top features
    selected_features = select_features(feature_importance, method="common_top", n_features=15)
    print("Selected features:", selected_features)
    
    X_train = df_train[selected_features]
    y_train = train_target
    X_val = df_eval[selected_features]
    y_val = eval_target
    
    cat_features = [cf for cf in cat_features if cf in selected_features]
    
    # Define the parameter space
    param_dist = {
        'learning_rate': np.logspace(-2, -1, num=100),  # Lowered upper bound
        'depth': [2, 4, 6, 8, 10],
        'l2_leaf_reg': np.linspace(1, 15.0, num=10),  # Increased lower bound
        'iterations': [5000],
        'min_data_in_leaf': [1, 5, 8, 10,12, 20],
        'subsample': np.linspace(0.3, 0.9, num=6),
        'boosting_type': ['Plain', 'Ordered'],
        'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
        'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
        'early_stopping_rounds': [500]
    }

    # Create the CatBoost model
    model = CatBoostRegressor(cat_features=cat_features, random_seed=42, verbose=False)

    # Create the random search object
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=splits,
        random_state=42,
        n_jobs=-1,
        scoring='neg_mean_absolute_percentage_error',
        verbose=2
    )

    # Perform the random search
    random_search.fit(X_train, y_train)

    # Get the best model
    best_model = random_search.best_estimator_

    # Evaluate the best model on the validation set
    y_pred = best_model.predict(X_val)
    val_mape = mean_absolute_percentage_error(y_val, y_pred)

    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {-random_search.best_score_:.4f} MAPE")
    print(f"Validation MAPE: {val_mape:.4f}")
    best_params = {k:str(v) for k,v in random_search.best_params_.items()}
    
    # Create and save config
    config = {
        'selected_features': selected_features,
        'cat_features': cat_features,
        'best_params': best_params
    }
    
    with open('best_config.yml', 'w') as f:
        yaml.dump(config, f)

    return best_model, config, val_mape

if __name__ == "__main__":
    df_train, train_target, df_eval, eval_target, cat_features = prepare_data(DF_TRAIN_PATH, DF_EVAL_PATH)
    best_model, config, val_mape = random_search_tuning(df_train, train_target, df_eval, eval_target, cat_features)
    print(f"Optimization complete. Config saved to 'best_config.yml'. Validation MAPE: {val_mape:.4f}")