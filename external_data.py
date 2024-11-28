import pandas as pd

DIESEL_PATH = 'dataset/external_data/Diesel_Price.csv'
FTSI_PATH = 'dataset/external_data/Freight_Transportation_Services_Index.csv'
PPI_TRUCKING_PATH = 'dataset/external_data/PPI_trucking.csv'
SPOT_RATES_PATH = 'dataset/external_data/Truck_Spot_Rates.csv'

def add_diesel_prices(df):
    diesel_df = pd.read_csv(DIESEL_PATH, sep='\t', engine='python', encoding='utf-16')
    # Convert Date in diesel_df to datetime and shift it forward by one week
    diesel_df['Date'] = pd.to_datetime(diesel_df['Date'], format='%m/%d/%Y') + pd.Timedelta(days=7)
    diesel_df = diesel_df.drop(columns=["Week of Equalized Week",
                                        "Year of Date",
                                        "Current Date - Week"])
    
    # Sort both DataFrames by date
    df = df.sort_values('pickup_date')
    diesel_df = diesel_df.sort_values('Date')
    
    # Rename the 'Date' column in diesel_df to avoid confusion
    diesel_df = diesel_df.rename(columns={'Date': 'diesel_date',
                                          'Metric Value': 'diesel_rate'})
    
    # Merge the DataFrames
    df_merged = pd.merge_asof(df, diesel_df, left_on='pickup_date', right_on='diesel_date', direction='backward')

    # Forward fill any missing values (in case there are gaps in the diesel data)
    df_merged['diesel_rate'] = df_merged['diesel_rate'].ffill()

    # If you want to drop the diesel_date column
    df_merged = df_merged.drop('diesel_date', axis=1)
    
    return df_merged

def add_ftsi(df):
    ftsi_df = pd.read_csv(FTSI_PATH, sep='\t', engine='python', encoding='utf-16')
    ftsi_df['ftsi_date'] = pd.to_datetime(ftsi_df['Date Tool Tip'], format='%B %Y') + pd.offsets.MonthEnd(0)
    ftsi_df = ftsi_df.drop(columns=["Month of Obs Date",
                                    "Year of Obs Date",
                                    "Current Date - Month",
                                    "Date Tool Tip"])

    # Sort the DataFrame by date
    df = df.sort_values('pickup_date')
    ftsi_df = ftsi_df.sort_values('ftsi_date')
    
    ftsi_df = ftsi_df.rename(columns={'Tsi Freight': 'freight_tsi'})

    # Now you can merge this with your main DataFrame
    df_merged = pd.merge_asof(df, ftsi_df, left_on='pickup_date', right_on='ftsi_date', direction='backward')

    # Forward fill any missing values
    df_merged['freight_tsi'] = df_merged['freight_tsi'].ffill()
    df_merged = df_merged.drop('ftsi_date', axis=1)
    
    return df_merged

def add_ppi_trucking(df):
    ppi_df = pd.read_csv(PPI_TRUCKING_PATH, sep='\t', engine='python', encoding='utf-16')
    ppi_df = ppi_df[ppi_df["Metric Name"] == 'PPI: General Freight Trucking']
    ppi_df['ppi_date'] = pd.to_datetime(ppi_df['Date'], format='%m/%d/%Y') + pd.offsets.MonthEnd(0)
    ppi_df = ppi_df.drop(columns=["Metric Name",
                                    "Date Month",
                                    "Current Date - Month",
                                    "Date"])

    # Sort the DataFrame by date
    df = df.sort_values('pickup_date')
    ppi_df = ppi_df.sort_values('ppi_date')
    
    ppi_df = ppi_df.rename(columns={'Metric Value': 'ppi_trucking_rate'})

    # Now you can merge this with your main DataFrame
    df_merged = pd.merge_asof(df, ppi_df, left_on='pickup_date', right_on='ppi_date', direction='backward')

    # Forward fill any missing values
    df_merged['ppi_trucking_rate'] = df_merged['ppi_trucking_rate'].ffill()
    df_merged = df_merged.drop('ppi_date', axis=1)
    
    return df_merged

def add_truck_spot_rates(df):
    spot_rates_df = pd.read_csv(SPOT_RATES_PATH, sep='\t', engine='python', encoding='utf-16')
    spot_rates_df['spot_date'] = pd.to_datetime(spot_rates_df['Date'], format='%m/%d/%Y')+ pd.Timedelta(days=8)

    
    spot_rates_pivoted = spot_rates_df.pivot(index='spot_date', columns='Measure Names', values='Measure Values')
    spot_rates_pivoted.reset_index(inplace=True)
    
    # Sort the DataFrame by date
    df = df.sort_values('pickup_date')
    spot_rates_pivoted = spot_rates_pivoted.sort_values('spot_date')
    
    spot_rates_pivoted = spot_rates_pivoted.rename(columns={'Dry Van': 'dry_van_rate',
                                                            'Flatbeds': 'flatbeds_rate',
                                                            'Refrigerated': 'refrigerated_rate'})

    # Now you can merge this with your main DataFrame
    df_merged = pd.merge_asof(df, spot_rates_pivoted, left_on='pickup_date', right_on='spot_date', direction='backward')

    # Forward fill any missing values
    for col in spot_rates_pivoted.columns:
        if col != 'spot_date':
            df_merged[col] = df_merged[col].ffill()
    df_merged = df_merged.drop('spot_date', axis=1)
    
    return df_merged

def add_external_data(df):
    df = add_diesel_prices(df)
    df = add_ftsi(df)
    df = add_ppi_trucking(df)
    df = add_truck_spot_rates(df)
    return df