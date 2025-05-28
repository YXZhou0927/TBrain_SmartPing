import pandas as pd

def rename_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names by converting them to lowercase and replacing spaces with underscores.

    Args:
        df (pd.DataFrame): Input DataFrame with columns to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized column names.
    """
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df