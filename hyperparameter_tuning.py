import numpy as np
import yaml
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_percentage_error
from catboost import CatBoostRegressor, Pool

from data_processing import prepare_data
from feature_engineering import select_features, time_based_split, analyze_feature_importance

def objective(params, X_train, y_train, X_val, y_val, cat_features):
    model = CatBoostRegressor(
        eval_metric="MAPE",
        bootstrap_type="MVS",
        learning_rate=0.1,
        depth=int(params['depth']),
        l2_leaf_reg=params['l2_leaf_reg'],
        iterations=1000,
        min_data_in_leaf=int(params['min_data_in_leaf']),
        subsample=params['subsample'],
        cat_features=cat_features,
        random_seed=42,
        verbose=False
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=200)
    y_pred = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    return {'loss': mape, 'status': STATUS_OK}

def hyperopt_tuning(df_train, df_eval, cat_features, max_evals=30):
    train_target = df_train.rate
    eval_target = df_eval.rate
    drop_features = ["pickup_date", "date", "rate"]
    df_train, df_eval = df_train.drop(drop_features, axis=1), df_eval.drop(drop_features, axis=1)
    
    splits = time_based_split(df_train, n_splits=3)
    # Feature importance analysis
    base_model = CatBoostRegressor(learning_rate=0.1,
                                   eval_metric = "MAPE",
                                   bootstrap_type="MVS",
                                   random_seed=42, 
                                   iterations=1000,
                                   depth=8,
                                   early_stopping_rounds=200)
    feature_importance = analyze_feature_importance(base_model, df_train, train_target, splits, cat_features)
    
    # Select top features
    selected_features = select_features(feature_importance, method="common_top", n_features=20)
    # selected_features = ['rate_similar_9', 'rate_similar_1', 'custom_distance_category', 
    #                    'rate_similar_10', 'directional_route', 'rate_similar_8', 
    #                    'rate_similar_5', 'rate_similar_4', 'origin_kma', 'rate_similar_7', 
    #                    'rate_similar_2', 'distance_category', 'destination_kma', 
    #                    'valid_miles', 'rate_similar_3']
    print("Selected features:", selected_features)
    
    X_train = df_train[selected_features]
    y_train = train_target
    X_val = df_eval[selected_features]
    y_val = eval_target
    
    cat_features = [cf for cf in cat_features if cf in selected_features]
    
    # Define the parameter space
    space = {
        'depth': hp.quniform('depth', 4, 10, 1),
        'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(1), np.log(10)),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 30, 1),
        'subsample': hp.uniform('subsample', 0.5, 1.0)
    }

    trials = Trials()
    best = fmin(
        fn=lambda params: objective(params, X_train, y_train, X_val, y_val, cat_features),
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )

    # Get the best parameters
    best_params = {
        'depth': int(best['depth']),
        'l2_leaf_reg': best['l2_leaf_reg'],
        'min_data_in_leaf': int(best['min_data_in_leaf']),
        'subsample': best['subsample']
    }

    # Create the best model
    best_model = CatBoostRegressor(
        **best_params,
        cat_features=cat_features,
        random_seed=42,
        verbose=False
    )

    # Fit the best model
    best_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)

    # Evaluate the best model on the validation set
    y_pred = best_model.predict(X_val)
    val_mape = mean_absolute_percentage_error(y_val, y_pred)

    print(f"Best parameters: {best_params}")
    print(f"Validation MAPE: {val_mape:.4f}")

    # Create and save config
    config = {
        'selected_features': selected_features,
        'cat_features': cat_features,
        'best_params': {k: str(v) for k, v in best_params.items()}
    }
    
    with open('best_config.yml', 'w') as f:
        yaml.dump(config, f)

    return best_model, config, val_mape

if __name__ == "__main__":
    df_train, df_eval, df_test, _, cat_features = prepare_data(recalculate=True)
    best_model, config, val_mape = hyperopt_tuning(df_train, df_eval, cat_features)
    print(f"Optimization complete. Config saved to 'best_config.yml'. Validation MAPE: {val_mape:.4f}")