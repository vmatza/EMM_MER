import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

def fit_linear_mixed_model(df: pd.DataFrame, variable: str):
    """
    Fit a linear mixed-effects model:
    Y_ij = β0 + β1*time_ij + b0_i + b1_i*time_ij + ε_ij

    Parameters:
        df : pd.DataFrame
            Input dataset.
        variable : str
            Target variable to model.

    Returns:
        MixedLMResults: Fitted linear mixed model.
    """
    df = df.copy()
    df['time_numeric'] = pd.to_numeric(df['time_numeric'], errors='coerce')
    df[variable] = pd.to_numeric(df[variable], errors='coerce')
    df = df.dropna(subset=['time_numeric', variable])
    md = sm.MixedLM.from_formula(
        f'{variable} ~ time_numeric',
        groups='respondentid',
        re_formula='~time_numeric',
        data=df
    )
    mdf = md.fit()
    return mdf

def fit_quadratic_mixed_model(df: pd.DataFrame, variable):
    """
    Fit a quadratic mixed-effects model:
    Y_ij = β0 + β1*time_ij + β2*time_ij^2 + b0_i + b1_i*time_ij + b2_i*time_ij^2 + ε_ij

    Parameters:
        df : pd.DataFrame
            Input dataset.
        variable : str
            Target variable to model.

    Returns:
        MixedLMResults: Fitted quadratic mixed model.
    """
    df = df.copy()
    df['time_numeric'] = pd.to_numeric(df['time_numeric'], errors='coerce')
    df[variable] = pd.to_numeric(df[variable], errors='coerce')
    df['time_numeric_sq'] = df['time_numeric'] ** 2
    df = df.dropna(subset=['time_numeric', 'time_numeric_sq', variable])
    md = sm.MixedLM.from_formula(
        f'{variable} ~ time_numeric + time_numeric_sq',
        groups='respondentid',
        re_formula='~time_numeric + time_numeric_sq',
        data=df
    )
    mdf = md.fit()
    return mdf

def extract_model_summary(res, var: str, nonlinear: bool):
    """
    Extract fixed and random effect parameters from a fitted model.

    Parameters:
        res : MixedLMResults
            Fitted model result.
        var : str
            Name of the target variable.
        nonlinear : bool
            Whether the model is quadratic.

    Returns:
    Dict[str, Any]: Summary of fixed effects and random effect standard deviations.
    """
    # Fixed effects
    fe = res.fe_params
    beta0 = fe.get('Intercept', np.nan)
    beta1 = fe.get('time_numeric', np.nan)
    beta2 = fe.get('time_numeric_sq', np.nan) if nonlinear else np.nan
    # Random effects standard deviations
    cov_re = res.cov_re
    sd_b0i = np.sqrt(cov_re.iloc[0, 0]) if cov_re.shape[0] > 0 else np.nan
    sd_b1i = np.sqrt(cov_re.iloc[1, 1]) if cov_re.shape[0] > 1 else np.nan
    sd_b2i = np.sqrt(cov_re.iloc[2, 2]) if (nonlinear and cov_re.shape[0] > 2) else np.nan
    return {
        'variable': var,
        'beta0': beta0,
        'beta1': beta1,
        'beta2': beta2,
        'sd_b0i': sd_b0i,
        'sd_b1i': sd_b1i,
        'sd_b2i': sd_b2i
    }