import pandas as pd


def clean_cd_column(df: pd.DataFrame, col: str = "業者CD") -> pd.DataFrame:
    valid = df[col].notna()
    df.loc[valid, col] = df.loc[valid, col].apply(lambda x: str(int(float(x))))
    return df
