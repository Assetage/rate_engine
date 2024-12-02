import os
import pandas as pd

from feature_engineering import (
    generate_time_features, 
    generate_categorized_features, 
    generate_droute_popularity,
    generate_lagged_features
)
from similarity_calculation import calculate_similar_rows, filter_10_similar, featurize_10_similar_rates

DF_TRAIN_PATH = 'dataset/train.csv'
DF_EVAL_PATH = 'dataset/validation.csv'
DF_TEST_PATH = 'dataset/test.csv'

def read_raw_data():
    raw_train = pd.read_csv(DF_TRAIN_PATH)
    raw_validation = pd.read_csv(DF_EVAL_PATH)
    raw_test = pd.read_csv(DF_TEST_PATH)
    return raw_train, raw_validation, raw_test
    

def preserve_indices(df):
    df = df.sort_values(["pickup_date","transport_type","origin_kma","destination_kma"])
    df = df.reset_index().rename({"index":"original_index"}, axis=1)
    df["original_index"] = df["original_index"].astype("uint32")
    return df


def combine_data(train,validation,test):
    train["source"] = "train"
    validation["source"] = "eval"
    test["source"] = "test"
    
    df_combined = pd.concat([train,validation,test], axis=0, ignore_index=True)
    df_combined = df_combined.reset_index().rename({"index":"order_index"}, axis=1)
    df_combined["source"] = df_combined["source"].astype("category")
    df_combined["order_index"] = df_combined["order_index"].astype("uint32")
    index_mapping = dict(zip(df_combined["order_index"], df_combined["original_index"].values))
    return df_combined, index_mapping

def split_data(df_combined):
    df_train = df_combined[df_combined["source"]=="train"].drop("source", axis=1)
    df_eval = df_combined[df_combined["source"]=="eval"].drop("source", axis=1)
    df_test = df_combined[df_combined["source"]=="test"].drop("source", axis=1)
    return df_train, df_eval, df_test


def optimize_dtypes(df):
    df["valid_miles"] = df["valid_miles"].round(2)
    df["weight"] = df["weight"].round(3)
    df["transport_type"] = df["transport_type"].astype("category")
    df["origin_kma"] = df["origin_kma"].astype("category")
    df["destination_kma"] = df["destination_kma"].astype("category")
    return df

def create_mine_predicted(row):
    if row['pickup_date_unix_ratio'] == 0.999 and row['rate_similar_1']==row['rate_similar_2']:
        return row['rate_similar_1']
    else:
        return row['predicted_rate']

def complete_predictions(X_test, predicted_rates, index_mapping):
    X_test["predicted_rate"] = predicted_rates
    X_test["predicted_rate_2"] = X_test.apply(create_mine_predicted, axis=1)
    X_test.to_csv("dataset/full_test.csv", index=True)
    predictions_df = X_test[["order_index","predicted_rate","predicted_rate_2"]]
    predictions_df["original_index"] = predictions_df["order_index"].map(index_mapping)
    predictions_df = predictions_df.set_index("original_index")
    predictions_df = predictions_df.drop("order_index", axis=1)
    return predictions_df

def define_cat_features(df_combined):
    possible_cat_features = ["transport_type","origin_kma","destination_kma",
                "quarter","month","day_of_week","is_weekend","day_of_year",
                "distance_category","custom_distance_category",
                'directional_route']
    cat_features = [feat for feat in possible_cat_features if feat in df_combined.columns]
    return cat_features

def load_or_calculate_features(df_combined, recalculate=False):
    file_path = "feature_store/df_combined.csv"
    if os.path.exists(file_path) and recalculate==False:
        df_combined = pd.read_csv(file_path)
    else:
        df_combined = calculate_features(df_combined)
        df_combined.to_csv(file_path, index=False)
    cat_features = define_cat_features(df_combined)
    return df_combined, cat_features

def calculate_features(df_combined):
    df_combined = optimize_dtypes(df_combined)
    
    df_similarity = calculate_similar_rows(df_combined)
    filtered_features = filter_10_similar(df_similarity)
    del df_similarity
    
    df_combined = featurize_10_similar_rates(df_combined, filtered_features)
    
    df_combined = generate_time_features(df_combined)
    df_combined = generate_categorized_features(df_combined, df_combined)
    df_combined = generate_droute_popularity(df_combined)
    df_combined = generate_lagged_features(df_combined)
    return df_combined

def prepare_data(recalculate=False):
    train, validation, test = read_raw_data()
    train,validation,test = preserve_indices(train), preserve_indices(validation), preserve_indices(test)
    
    df_combined, index_mapping = combine_data(train,validation,test)
    df_combined, cat_features = load_or_calculate_features(df_combined,recalculate=recalculate)
    df_train, df_eval, df_test = split_data(df_combined)
    df_train = filter_by_rate_similar_1_availability(df_train)
    return df_train, df_eval, df_test, index_mapping, cat_features

def select_predefined_features(df_train, df_eval, df_test, config):
    selected_features = config['selected_features']
    X_train, y_train = df_train[selected_features], df_train.rate
    X_val, y_val = df_eval[selected_features], df_eval.rate
    X_test = df_test[selected_features]
    return X_train, y_train, X_val, y_val, X_test

def filter_by_rate_similar_1_availability(df_train):
    df_train = df_train.dropna(subset=["rate_similar_1"])
    return df_train