import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

def generate_time_features(df):
    # Time-based features
    df['pickup_date'] = pd.to_datetime(df['pickup_date'])
    df['year'] = df['pickup_date'].dt.year
    df['quarter'] = df['pickup_date'].dt.quarter
    df['month'] = df['pickup_date'].dt.month
    df['day_of_week'] = df['pickup_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['day_of_year'] = df['pickup_date'].dt.dayofyear
    df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
    df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
    df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
    df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))
    df['date'] = df['pickup_date'].dt.date
    return df

def calculate_category_boundaries(series, num_categories=5):
    """
    Calculate category boundaries based on quantiles of the input series.
    
    :param series: pandas Series to calculate boundaries for
    :param num_categories: number of categories to create
    :return: list of boundary values
    """
    quantiles = np.linspace(0, 1, num_categories + 1)
    boundaries = series.quantile(quantiles).unique()
    return boundaries

def apply_categories(series, boundaries, feature_name):
    """
    Apply pre-calculated boundaries to categorize a series.
    
    :param series: pandas Series to categorize
    :param boundaries: list of boundary values
    :param feature_name: name of the feature (used for category labels)
    :return: pandas Series with categories
    """
    labels = [f'{feature_name}_cat_{i+1}' for i in range(len(boundaries) - 1)]
    category_ranges = {label: f"{boundaries[i]:.2f} - {boundaries[i+1]:.2f}" 
                       for i, label in enumerate(labels)}
    return pd.cut(series, bins=boundaries, labels=labels, include_lowest=True)

def calculate_custom_distance_boundaries(series, num_categories=5):
    """
    Calculate custom distance category boundaries with more granularity for shorter distances,
    ensuring all values are covered.
    
    :param series: pandas Series of distances
    :param num_categories: Total number of categories to create
    :return: list of boundary values
    """
    min_value = series.min()
    max_value = series.max()
    
    # Calculate quantiles with more emphasis on lower values
    quantiles = [0] + [1 / (2 ** i) for i in range(num_categories - 1, 0, -1)]
    boundaries = series.quantile(quantiles).tolist()
    
    # Ensure the boundaries cover all values
    boundaries[0] = min_value
    boundaries.append(max_value)
    
    # Remove any duplicate boundaries
    boundaries = sorted(set(boundaries))
    
    return boundaries

def generate_categorized_features(df, df_train):
    distance_boundaries = calculate_category_boundaries(df_train['valid_miles'])
    custom_distance_boundaries = calculate_custom_distance_boundaries(df_train['valid_miles'])
    df['distance_category'] = apply_categories(df['valid_miles'], distance_boundaries, 'distance')
    df['custom_distance_category'] = apply_categories(df['valid_miles'], custom_distance_boundaries, 'custom_distance')
    return df

def generate_droute_popularity(df, origin_col='origin_kma', destination_col='destination_kma'):
    """
    Create route popularity features.
    
    :param df: DataFrame containing the data
    :param origin_col: Name of the column containing the origin
    :param destination_col: Name of the column containing the destination
    :return: DataFrame with added route popularity features
    """
    df[origin_col] = df[origin_col].astype(str)
    df[destination_col] = df[destination_col].astype(str)
    # Create a directional route identifier
    df['directional_route'] = df[origin_col] + '_to_' + df[destination_col]
    
    # Calculate raw popularity (count of rows for each route)
    route_popularity = df['directional_route'].value_counts()
    
    # Calculate normalized popularity (fraction of total recordings)
    total_records = len(df)
    route_popularity_normalized = route_popularity / total_records
    
    # Add the popularity features to the dataframe
    df['route_popularity_raw'] = df['directional_route'].map(route_popularity)
    df['route_popularity_normalized'] = df['directional_route'].map(route_popularity_normalized)
    df['directional_route'] = df['directional_route'].astype("category")
    df['route_popularity_raw'] = df['route_popularity_raw'].astype("uint16")
    df['route_popularity_normalized'] = df['route_popularity_normalized'].astype("float16")
    df[origin_col] = df[origin_col].astype("category")
    df[destination_col] = df[destination_col].astype("category")
    return df

def generate_lagged_features(df, 
                             cat_features=["transport_type","custom_distance_category"], 
                             date_col='date', target_col='rate_similar_1', windows=[30, 45, 90, 120]):
    # Ensure the date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort the dataframe by date
    df = df.sort_values(date_col)
    
    # Group by date and all categorical features, and calculate mean of target
    grouped = df.groupby([date_col] + cat_features)[target_col].mean().reset_index()
    
    # Sort the grouped dataframe by date
    grouped = grouped.sort_values(date_col)
    
    # Calculate lagged features
    for window in windows:
        # Calculate mean
        grouped[f"mean_{window}d"] = grouped[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        
        # Calculate EWMA
        grouped[f"ewma_{window}d"] = grouped[target_col].shift(1).ewm(span=window, adjust=False).mean()
        
        # Calculate rate change
        grouped[f"rate_change_{window}d"] = (grouped[target_col] - grouped[target_col].shift(window)) / grouped[target_col].shift(window)
    
    # Merge the new features back to the original dataframe
    result = pd.merge(df, grouped, on=[date_col] + cat_features, suffixes=('', '_mean'))
    
    # Drop the additional target column created during merging
    result = result.drop(columns=[f"{target_col}_mean"])
    
    return result

def analyze_feature_importance(model, X, y, splits, cat_features):
    shap_values_list = []
    perm_importance_list = []
    
    for train_index, val_index in splits:
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model.fit(X_train, y_train, cat_features=cat_features, metric_period=100)
        
        # SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
        shap_values_list.append(np.abs(shap_values).mean(axis=0))
        
        # Permutation importance
        perm_importance = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)
        perm_importance_list.append(perm_importance.importances_mean)
    
    # Average importance across folds
    shap_importance = np.mean(shap_values_list, axis=0)
    perm_importance = np.mean(perm_importance_list, axis=0)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'shap_importance': shap_importance,
        'perm_importance': perm_importance
    })
    # Normalize importances
    feature_importance['shap_importance_norm'] = feature_importance['shap_importance'] / feature_importance['shap_importance'].sum()
    feature_importance['perm_importance_norm'] = feature_importance['perm_importance'] / feature_importance['perm_importance'].sum()
    
    # Combine SHAP and permutation importance
    feature_importance['combined_importance'] = (feature_importance['shap_importance_norm'] + feature_importance['perm_importance_norm']) / 2
    
    feature_importance = feature_importance.sort_values('combined_importance', ascending=False)
    
    return feature_importance

def select_features(feature_importance, method='common_top', n_features=20):
    if method == 'combined':
        return feature_importance['feature'].head(n_features).tolist()
    
    elif method == 'common_top':
        top_shap = set(feature_importance.nlargest(n_features, 'shap_importance_norm')['feature'])
        top_perm = set(feature_importance.nlargest(n_features, 'perm_importance_norm')['feature'])
        return list(top_shap.intersection(top_perm))
    
    elif method == 'union_top':
        top_shap = set(feature_importance.nlargest(n_features // 2, 'shap_importance_norm')['feature'])
        top_perm = set(feature_importance.nlargest(n_features // 2, 'perm_importance_norm')['feature'])
        union = list(top_shap.union(top_perm))
        # If we don't have enough features, add more from the combined ranking
        if len(union) < n_features:
            remaining = set(feature_importance['feature']) - set(union)
            union.extend(feature_importance[feature_importance['feature'].isin(remaining)]
                         .nlargest(n_features - len(union), 'combined_importance')['feature'])
        return union[:n_features]
    
    else:
        raise ValueError("Invalid method specified. Choose 'combined', 'common_top', or 'union_top'.")
    
def time_based_split(df, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(df))

