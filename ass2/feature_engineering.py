import pandas as pd

def preprocess_dates(df):
    df.date_time = pd.to_datetime(df.date_time)
    df['year'], df['month'], df['day'], df['hour'], df['minute'] = df.date_time.dt.year, df.date_time.dt.month, df.date_time.dt.day, df.date_time.dt.hour, df.date_time.dt.minute
    return df