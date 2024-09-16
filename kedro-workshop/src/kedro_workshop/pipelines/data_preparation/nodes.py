import pandas as pd
from typing import List, Any, Dict

def process_cabin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'Cabin' column in the DataFrame to extract 'Deck' and 'Side'.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing a 'Cabin' column.

    Returns:
    pd.DataFrame: DataFrame with 'Deck' and 'Side' columns added.
    """

    # Extract deck
    df['Deck'] = df['Cabin'].apply(lambda x: x.split("/")[0] if not pd.isna(x) else None)

    # Extract side
    df['Side'] = df['Cabin'].apply(lambda x: x.split("/")[-1] if not pd.isna(x) else None)

    return df


def fill_missing_values(df: pd.DataFrame, col_values : Dict[str, Any]) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame based on the specified column values.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    col_values (Dict[str, Any]): Dictionary containing column names and fill values (e.g., '%MODE%', '%MEDIAN%', or specific values).

    Returns:
    pd.DataFrame: DataFrame with missing values filled.
    """

    for col, value in col_values.items():
        if value == "%MODE%":
            mode = df[col].mode()[0]
            df.fillna({col: mode}, inplace=True)

        elif value == "%MEDIAN%":
            median = df[col].median()
            df.fillna({col: median}, inplace=True)

        else:
            df.fillna({col: value}, inplace=True)

    return df


def drop_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Drop specified columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    cols (List[str]): List of column names to drop from the DataFrame.

    Returns:
    pd.DataFrame: DataFrame with specified columns dropped.
    """
    
    df.drop(cols, axis=1, inplace=True)

    return df