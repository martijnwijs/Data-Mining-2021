from sklearn.utils import resample
import pandas as pd
def downsample(df):
    df_majority = df[df.click_bool == 0]
    df_minority = df[df.click_bool == 1]
    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=221879,  # df_minority.count()
                                       random_state=123  # in the example this is the number
                                       )
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    return df_downsampled