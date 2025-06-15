import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import math
from scipy.stats import shapiro
from scipy import stats

def plot_random_variable(df: pd.DataFrame, n: int = 20,
                      time_col: str = 'mm', value_col: str = 'variable'):
    """
    Plot line charts of a variable over time for n random unique respondents,
    ensuring timepoints are ordered as Intake ronde, T3, T6, T9, T12.

    Parameters:
        df : pd.DataFrame
            The DataFrame containing 'respondentid', time_col, and value_col.
        n : int
            Number of unique respondents to sample and plot.
        time_col : str
            Column name representing time points.
        value_col : str
            Column name representing the variable to plot.
    """
    required_timepoints = ['Intake ronde', 'T3', 'T6', 'T9', 'T12']
    df = df[df[time_col].isin(required_timepoints) & df[value_col].notna()].copy()
    df[time_col] = pd.Categorical(df[time_col], categories=required_timepoints, ordered=True)
    unique_ids = df['respondentid'].dropna().unique().tolist()
    if len(unique_ids) < n:
        raise ValueError(f"Not enough unique respondents: only {len(unique_ids)} available.")
    selected_ids = random.sample(unique_ids, n)
    plt.figure(figsize=(12, 6))
    for rid in selected_ids:
        patient_df = df[df['respondentid'] == rid].sort_values(time_col)
        plt.plot(patient_df[time_col], patient_df[value_col], marker='o', label=str(int(rid)))

    plt.xlabel('Time')
    plt.ylabel(value_col.capitalize())
    plt.title(f'{value_col.capitalize()} Progression for {n} Random Respondents')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

def plot_small_multiples(global_df: pd.DataFrame, subgroup_df: pd.DataFrame):
    """
    Compare global and subgroup regression curves for all target variables
    using small multiple plots (one per target).

    Parameters:
        global_df : pd.DataFrame
            Fixed-effect coefficients for the full population.
        subgroup_df : pd.DataFrame
            Fixed-effect coefficients for a specific subgroup.
    """
    ordered_targets = ['gp01', 'gp02', 'gp03', 'gp04', 'gp05', 'gp06', 'gp07',
                       'gp08', 'gp09', 'gp10', 'gp11', 'gp12', 'pasdas']
    name_map = {
        'gp01': 'Pain',
        'gp02': 'Fatigue',
        'gp03': 'Skin Problems',
        'gp04': 'Work/Leisure Impairment',
        'gp05': 'Functional Capacity',
        'gp06': 'Sleep Disturbances',
        'gp07': 'Discomfort',
        'gp08': 'Life Participation',
        'gp09': 'Emotional Wellbeing',
        'gp10': 'Social Participation',
        'gp11': 'Anxiety/Depression',
        'gp12': 'Coping',
        'pasdas': 'PASDAS Score'
    }

    times = np.array([0, 3, 6, 9, 12])
    tick_map = {0: 'T0', 3: 'T3', 6: 'T6', 9: 'T9', 12: 'T12'}
    n_targets = len(ordered_targets)
    n_cols = 4
    n_rows = math.ceil(n_targets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for idx, var in enumerate(ordered_targets):
        ax = axes[idx]
        g = global_df[global_df['variable'] == var].iloc[0]
        s = subgroup_df[subgroup_df['variable'] == var].iloc[0]
        nonlinear = pd.notna(g['beta2'])
        if nonlinear:
            y_g = g['beta0'] + g['beta1'] * times + g['beta2'] * times ** 2
            y_s = s['beta0'] + s['beta1'] * times + s['beta2'] * times ** 2
        else:
            y_g = g['beta0'] + g['beta1'] * times
            y_s = s['beta0'] + s['beta1'] * times
        ax.plot(times, y_g, 'k--', label='Global')
        ax.plot(times, y_s, 'r-', label='Subgroup')
        ax.set_title(name_map.get(var, var))
        ax.set_xticks(times)
        ax.set_xticklabels([tick_map[t] for t in times], rotation=0)
        ax.grid(True)
        if idx == 0:
            ax.legend()

    for j in range(n_targets, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Global vs Subgroup Fixed-Effect Trajectories', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=4.0)
    plt.show()

def normality_report(df: pd.DataFrame, targets: list):
    """
    Run Shapiro-Wilk test and plot Q-Q plots for each target variable.

    Parameters:
        df : pd.DataFrame
            Dataset with target columns.
        targets : list
            Target variable names.

    Returns:
        pd.DataFrame with W-statistic and p-values for normality test.
    """
    results = []
    for var in targets:
        # Force conversion to numeric, drop bad entries
        df[var] = pd.to_numeric(df[var], errors='coerce')
        data = df[var].dropna()
        if len(data) < 3:
            results.append((var, "Too few data points", None))
            continue
        stat, p = shapiro(data)
        results.append((var, stat, p))

        plt.figure()
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'Q–Q Plot for {var}')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results, columns=["Variable", "W-statistic", "p-value"])

def plot_model_fits(df: pd.DataFrame, lin_res, quad_res, variable: str):
    """
    Visualize observed mean trajectory alongside fitted linear and quadratic mixed-effects models.

    Parameters:
        df : pd.DataFrame
            Dataset containing the time series data.
        lin_res : MixedLMResults
            Fitted linear model.
        quad_res : MixedLMResults
            Fitted quadratic model.
        variable : str
            The name of the outcome variable to plot.
    """
    df['time_numeric'] = pd.to_numeric(df['time_numeric'], errors='coerce')
    df[variable] = pd.to_numeric(df[variable], errors='coerce')
    obs = df.groupby('time_numeric')[variable].mean().reset_index()
    times = np.array(sorted(df['time_numeric'].unique()))
    # Linear fixed effects
    fe_lin = lin_res.fe_params
    pred_lin = fe_lin['Intercept'] + fe_lin['time_numeric'] * times
    # Quadratic fixed effects
    fe_quad = quad_res.fe_params
    pred_quad = (
        fe_quad['Intercept'] +
        fe_quad['time_numeric'] * times +
        fe_quad['time_numeric_sq'] * (times ** 2)
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(obs['time_numeric'], obs[variable], color='black', alpha=0.6, label='Observed mean')
    plt.plot(times, pred_lin, linestyle='--', marker='o', label='Linear fit')
    plt.plot(times, pred_quad, linestyle='-', marker='o', label='Quadratic fit')

    tick_map = {0: 'Intake ronde', 3: 'T3', 6: 'T6', 9: 'T9', 12: 'T12'}
    plt.xticks(list(tick_map.keys()), list(tick_map.values()), rotation=45)

    plt.xlabel('Time')
    plt.ylabel(variable)
    plt.title(f'Mixed-Effects Model Fits: {variable} over Time (MTX subgroup)')
    plt.legend()
    plt.tight_layout()
    plt.show()