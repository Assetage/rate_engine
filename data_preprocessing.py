import pandas as pd

from feature_engineering import (
    generate_time_features, 
    create_distance_features, 
    create_route_features, 
    create_distance_variation_features,
    create_transport_type_features,
    create_location_features
)
from feature_engineering import create_lag_features, analyze_feature_importance, select_features
from external_data import add_external_data
from sklearn.model_selection import TimeSeriesSplit

def load_and_combine_data(train_path, eval_path):
    df_train = pd.read_csv(train_path)
    df_eval = pd.read_csv(eval_path)
    df_train["source"] = "train"
    df_eval["source"] = "eval"
    
    df_combined = pd.concat([df_train, df_eval], axis=0, ignore_index=True).sort_values('pickup_date')
    return df_combined

def create_features(df, cat_features):
    df = create_lag_features(df, 'rate', cat_features)
    return df

def split_data(df):
    df_train = df[df["source"]=="train"].drop("source", axis=1)
    df_eval = df[df["source"]=="eval"].drop("source", axis=1)
    return df_train, df_eval


def preprocess_data(df):
    cat_features = ["transport_type", "origin_kma", "destination_kma"]
    df["pickup_date"] = pd.to_datetime(df["pickup_date"], format='%Y-%m-%d %H:%M:%S').astype('datetime64[ns]')
    df = df.sort_values(by=["pickup_date", "transport_type", "origin_kma", "destination_kma"])
    # df, time_cat_features = generate_time_features(df, "pickup_date")
    # df = add_external_data(df)
    # df = create_route_features(df)
    # df = create_distance_variation_features(df)
    # df = create_transport_type_features(df)
    # df = create_location_features(df, 'origin_kma')
    # df = create_location_features(df, 'destination_kma')
    # df = create_distance_features(df)
    
    drop_features = ["rate"]
    target = df.rate
    df = df.drop(columns=drop_features)
    
    cat_features = cat_features#+["route_frequency_bin"]#+["distance_category"] + time_cat_features #
    return df, target, cat_features

def prepare_data(train_path, eval_path):
    df_combined = load_and_combine_data(train_path, eval_path)
    cat_features = ["transport_type", "origin_kma", "destination_kma"]
    df_combined = create_features(df_combined, cat_features)
    df_train, df_eval = split_data(df_combined)
    
    df_train, train_target, cat_features = preprocess_data(df_train)
    df_eval, eval_target, _ = preprocess_data(df_eval)
    
    return df_train, train_target, df_eval, eval_target, cat_features

def time_based_split(df, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(df))