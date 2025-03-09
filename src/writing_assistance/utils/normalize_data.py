import pandas as pd


def normalize_avg_score(df: pd.DataFrame) -> pd.DataFrame:
    """function to normalize data from <-3,3> to <0,1>"""
    df['avg_score'] = (df['avg_score'] + 3) / 6
    return df
