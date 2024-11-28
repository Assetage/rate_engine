import pandas as pd
import numpy as np
from scipy import stats
import shap

from sklearn.inspection import permutation_importance
from pandas.tseries.holiday import USFederalHolidayCalendar


def generate_time_features(df, timestamp_col):
    # Ensure the timestamp column is in datetime format
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Extract basic time components
    df['year'] = df[timestamp_col].dt.year
    df['month'] = df[timestamp_col].dt.month
    df['day'] = df[timestamp_col].dt.day
    df['hour'] = df[timestamp_col].dt.hour
    df['minute'] = df[timestamp_col].dt.minute
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    
    # Weekday/Weekend
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Part of the day
    df['part_of_day'] = pd.cut(df['hour'], 
                               bins=[-1, 6, 12, 18, 24], 
                               labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # 9, 10, 11
            return 'Fall'
    
    df['season'] = df['month'].apply(get_season)
    
    # US Federal Holidays
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df[timestamp_col].min(), end=df[timestamp_col].max())
    df['is_holiday'] = df[timestamp_col].dt.date.astype('datetime64[ns]').isin(holidays).astype(int)
    
    # Is it rush hour? (You can adjust the hours as needed)
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                          (df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)
    
    # Cyclical encoding for hour, day of week, and month
    df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
    df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
    df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
    df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
    df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))
    
    df = df.drop(columns=["year","month","day","hour","minute","day_of_week"])
    cat_features = ["is_weekend","part_of_day","season","is_holiday","is_rush_hour"]
    return df, cat_features

def create_lag_features(df, target_col, group_cols):
    df = df.sort_values('pickup_date')
    
    # Function to create lag features for a given group
    def create_lags(group):
        result = group.copy()
        result['latest_rate'] = group[target_col].shift(1)
        result['latest_rate_2nd'] = group[target_col].shift(2)
        result['mean_rate_30d'] = group[target_col].shift(1).rolling(window=30, min_periods=1).mean()
        result['mean_rate_45d'] = group[target_col].shift(1).rolling(window=45, min_periods=1).mean()
        result['mean_rate_90d'] = group[target_col].shift(1).rolling(window=90, min_periods=1).mean()
        result['mean_rate_120d'] = group[target_col].shift(1).rolling(window=120, min_periods=1).mean()
        
        # Exponentially weighted moving averages (EWMA)
        result['ewma_30d'] = group[target_col].shift(1).ewm(span=30, min_periods=1).mean()
        result['ewma_45d'] = group[target_col].shift(1).ewm(span=45, min_periods=1).mean()
        result['ewma_90d'] = group[target_col].shift(1).ewm(span=90, min_periods=1).mean()
        result['ewma_120d'] = group[target_col].shift(1).ewm(span=120, min_periods=1).mean()
        
        # Rate of change features
        result['rate_change_30d'] = (group[target_col].shift(1) - group[target_col].shift(30)) / group[target_col].shift(30)
        result['rate_change_45d'] = (group[target_col].shift(1) - group[target_col].shift(45)) / group[target_col].shift(45)
        result['rate_change_90d'] = (group[target_col].shift(1) - group[target_col].shift(90)) / group[target_col].shift(90)
        
        return result
    
    df = df.groupby(group_cols, group_keys=False).apply(create_lags)
        
    for col in group_cols:
        col_prefix = f"{col}_"
        temp_df = df.groupby(col, group_keys=False).apply(create_lags)
        
        lag_columns = ['latest_rate', 'latest_rate_2nd', 'mean_rate_30d','mean_rate_45d', 'mean_rate_90d', 'mean_rate_120d',
                       'ewma_30d', 'ewma_45d','ewma_90d', 'ewma_120d', 
                       'rate_change_30d', 'rate_change_45d','rate_change_90d']
        
        for lag_col in lag_columns:
            df[f'{col_prefix}{lag_col}'] = temp_df[lag_col]
    return df

def create_route_features(df):
    # Create a unique identifier for each route, regardless of direction
    df['route_id'] = df.apply(lambda row: '_'.join(sorted([row['origin_kma'], row['destination_kma']])), axis=1)
    
    # Calculate route statistics
    route_stats = df.groupby('route_id').agg({
        'valid_miles': ['mean', 'median', 'std', 'count']
    })
    route_stats.columns = [f'route_{c[0]}_{c[1]}' for c in route_stats.columns]
    
    # Merge route statistics back to the main dataframe
    df = df.merge(route_stats, left_on='route_id', right_index=True, how='left')
    
    # Create a route frequency feature
    df['route_frequency'] = df['route_valid_miles_count']
    
    # Optionally, bin the route frequency
    df['route_frequency_bin'] = pd.qcut(df['route_frequency'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Drop the route_id column if we're not using it as a direct feature
    df = df.drop('route_id', axis=1)
    
    return df

def create_distance_variation_features(df):
    df['route_id'] = df.apply(lambda row: '_'.join(sorted([row['origin_kma'], row['destination_kma']])), axis=1)
    # Calculate the standard deviation of miles for each route
    route_mile_std = df.groupby('route_id')['valid_miles'].std().reset_index(name='route_mile_std')
    df = df.merge(route_mile_std, on='route_id', how='left')
    
    # Calculate the difference between current miles and average miles for the route
    df['mile_diff_from_avg'] = df['valid_miles'] - df['route_valid_miles_mean']
    df = df.drop('route_id', axis=1)
    return df

def create_transport_type_features(df):
    # Group statistics for miles
    transport_stats = df.groupby('transport_type').agg({
        'valid_miles': ['mean', 'median', 'std']
    })
    transport_stats.columns = [f'transport_{c[0]}_{c[1]}' for c in transport_stats.columns]
    df = df.merge(transport_stats, left_on='transport_type', right_index=True, how='left')
    
    # Transport type popularity
    transport_popularity = df.groupby('transport_type').size().reset_index(name='transport_popularity')
    df = df.merge(transport_popularity, on='transport_type', how='left')
    
    return df

def create_location_features(df, location_col):
    prefix = f'{location_col}_'
    
    # Group statistics for miles
    location_stats = df.groupby(location_col).agg({
        'valid_miles': ['mean', 'median', 'std']
    })
    location_stats.columns = [f'{prefix}{c[0]}_{c[1]}' for c in location_stats.columns]
    df = df.merge(location_stats, left_on=location_col, right_index=True, how='left')
    
    # Location popularity (number of trips)
    location_popularity = df.groupby(location_col).size().reset_index(name=f'{prefix}popularity')
    df = df.merge(location_popularity, on=location_col, how='left')
    
    return df

def get_optimal_bins(df, n_bins):
    df['temp_category'] = pd.qcut(df['valid_miles'], q=n_bins)
    bins = df['temp_category'].cat.categories
    bin_edges = [b.left for b in bins] + [bins[-1].right]
    print("Optimal bins number:", len(bin_edges)-1)
    df.drop('temp_category', axis=1, inplace=True)
    return bin_edges

def create_distance_features(df):
    
    optimal_bin_edges = get_optimal_bins(df, n_bins=15)
    # 1. Distance Bins
    df['distance_category'] = pd.cut(df['valid_miles'], 
                                     bins=optimal_bin_edges, 
                                     labels=[f'bin_{i}' for i in range(len(optimal_bin_edges)-1)])

    # 2. Distance Percentiles
    df['distance_percentile'] = df['valid_miles'].rank(pct=True)

    # 3. Non-linear Transformations
    df['distance_log'] = np.log1p(df['valid_miles'])
    df['distance_sqrt'] = np.sqrt(df['valid_miles'])
    df['distance_squared'] = df['valid_miles'] ** 2

    # 4. Distance-Time Interaction
    # df['distance_weekend'] = df['valid_miles'] * df['is_weekend']
    # df['distance_season'] = df['valid_miles'] * df['season'].astype('category').cat.codes

    # 5. Z-score of distance within transport type
    df['distance_zscore'] = df.groupby('transport_type')['valid_miles'].transform(lambda x: stats.zscore(x))

    # 6. Rolling average distance for each origin-destination pair
    df['rolling_avg_distance'] = df.groupby(['origin_kma', 'destination_kma'])['valid_miles'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())

    # 7. Distance to weight ratio
    df['distance_weight_ratio'] = df['valid_miles'] / df['weight']

    # 8. Geographical Density (without using rate)
    df['distance_density'] = df.groupby('distance_category')['valid_miles'].transform('count') / df['valid_miles']

    # 9. Average distance by transport type
    df['avg_distance_by_transport'] = df.groupby('transport_type')['valid_miles'].transform('mean')

    # 10. Distance difference from average for transport type
    df['distance_diff_from_avg'] = df['valid_miles'] - df['avg_distance_by_transport']

    return df

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

