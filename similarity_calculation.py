import numpy as np
import pandas as pd
import recordlinkage
from recordlinkage.base import BaseCompareFeature

class UnixTimestampComparison(BaseCompareFeature):
    def _compute_vectorized(self, s1, s2):
        with np.errstate(divide='ignore', invalid='ignore'):
            diff = np.round(s2/s1, 3)
        return np.where(diff >= 1, 0, diff)
    

def calculate_similar_rows(df_combined):
    original_columns = ['rate', 'valid_miles', 'transport_type', 'weight', 'pickup_date',
                        'origin_kma', 'destination_kma']
    

    df_combined = df_combined.set_index("order_index")
    df_combined = df_combined[original_columns]
    dfA = df_combined

    dfA["pickup_date_unix"] = pd.to_datetime(dfA["pickup_date"], format='%Y-%m-%d %H:%M:%S').astype('int64')
    dfA = dfA.drop("pickup_date", axis=1)

    # Indexation step
    indexer = recordlinkage.Index()
    indexer.block(["transport_type", "origin_kma", "destination_kma"])
    candidate_links = indexer.index(dfA)

    # Comparison step
    compare_cl = recordlinkage.Compare()

    compare_cl.numeric("valid_miles","valid_miles",label="valid_miles")
    compare_cl.numeric("weight","weight",label="weight")
    compare_cl.add(UnixTimestampComparison('pickup_date_unix','pickup_date_unix',
                                            label='pickup_date_unix_ratio'))

    features = compare_cl.compute(candidate_links, dfA)
    features = features.reset_index()
    level_0_index_name = "order_index_0"
    level_1_index_name = "order_index_1"
    features = features.rename({"level_0":level_0_index_name,
                                "level_1":level_1_index_name}, axis=1)
    features["score"] = features["valid_miles"]+features["weight"]+features["pickup_date_unix_ratio"]
    features = features.sort_values(["score","pickup_date_unix_ratio"], ascending=False)
    features = features[features["pickup_date_unix_ratio"]!=0]
    
    return features

def top_10_matches(group):
    return group.nlargest(10, columns='score')

def filter_10_similar(df_similarity):
    filtered_features = df_similarity.groupby('order_index_1', observed=False).apply(top_10_matches)
    return filtered_features

def featurize_10_similar_rates(df_combined, filtered_features):
    filtered_features = filtered_features.reset_index(drop=True)
    grouped = filtered_features.groupby('order_index_1').agg({
        'order_index_2': list,
        'pickup_date_unix_ratio': 'first'
    }).reset_index()

    for i in range(1, 11):
        grouped[f'order_index_similar_{i}'] = grouped['order_index_2'].apply(lambda x: x[i-1] if i <= len(x) else np.nan)

    # Drop the list column
    grouped = grouped.drop('order_index_2', axis=1)

    # Now you can proceed with the rest of your code
    # Join this grouped DataFrame with df_combined
    result = df_combined.merge(grouped, left_on='order_index', right_on='order_index_1', how='left')

    # Create a rates series
    rates = df_combined.set_index('order_index')['rate']

    # Now, let's add the similar rates
    for i in range(1, 11):
        column_name = f'order_index_similar_{i}'
        result[f'rate_similar_{i}'] = result[column_name].map(rates)

    # Drop unnecessary columns
    result = result.drop(columns=['order_index_1'] + [col for col in result.columns if col.startswith('order_index_similar_')])

    # Reorder columns
    cols = ['order_index', 'rate', 'pickup_date_unix_ratio'] + \
           [f'rate_similar_{i}' for i in range(1, 11)] + \
           [col for col in result.columns if col not in ['order_index', 'rate', 'pickup_date_unix_ratio'] and not col.startswith('rate_similar_')]
    result = result[cols]
    result["pickup_date_ur_reciprocal"] = 1/result["pickup_date_unix_ratio"]
    return result