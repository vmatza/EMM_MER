import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a Stata (.dta) file.

    Parameters:
        file_path : str or bytes
            The path to the Stata file to be loaded.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    df = pd.read_stata(file_path)
    return df