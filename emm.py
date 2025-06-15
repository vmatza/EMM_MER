import pandas as pd
import numpy as np
from itertools import combinations
from modeling import (
    fit_linear_mixed_model,
    fit_quadratic_mixed_model
)
from sklearn.preprocessing import StandardScaler

LINEAR_TARGETS = ['gp01','gp02','gp04','gp05','gp06','gp07','gp08','gp09','gp10','gp12']
NONLINEAR_TARGETS = ['pasdas','gp03','gp11']

def generate_conditions(df, variables):
    """
    Generate simple unary conditions from baseline variables for beam search.
    - For numeric variables: split at the median.
    - For categorical variables: create equality checks for each unique value.

    Returns:
        List of conditions in the form (variable, operator, value).
    """
    conditions = []
    for var in variables:
        if pd.api.types.is_numeric_dtype(df[var]):
            median = df[var].median()
            conditions.append((var, '<=', median))
            conditions.append((var, '>', median))
        else:
            vals = df[var].dropna().unique()
            for v in vals:
                conditions.append((var, '==', v))
    return conditions

def apply_condition(df, cond):
    """
    Apply a single condition to filter the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        condition (tuple): A tuple of the form (variable, operator, value).

    Returns:
        pd.DataFrame: Filtered DataFrame after applying the condition.
    """
    var, op, val = cond
    if op == '<=':
        return df[df[var] <= val]
    elif op == '>':
        return df[df[var] > val]
    elif op == '==':
        return df[df[var] == val]
    else:
        raise ValueError(f"Unknown operator {op}")


# Beam Search implementation

def compute_entropy(n, N):
    """
    Compute entropy of the split between the subgroup and its complement.

    Parameters:
        subgroup_size (int): Number of unique individuals in subgroup.
        total_size (int): Total number of individuals in the dataset.

    Returns:
        float: Entropy score favoring balanced subgroups.
    """
    n_c = N - n
    if n == 0 or n_c == 0:
        return 0.0
    p1, p2 = n / N, n_c / N
    return -p1 * np.log2(p1) - p2 * np.log2(p2)

def compute_cover_score(G_ids, beam_ids, alpha=0.5):
    """
    Compute Ω(G, Beam): penalize overlap with previously selected subgroups.

    Parameters:
        subgroup_ids (np.ndarray): Array of respondent IDs in candidate subgroup.
        existing_beam_ids (List[np.ndarray]): List of arrays of IDs from current beam.
        alpha (float): Exponential decay weight for overlap penalty.

    Returns:
        float: Average weight based on coverage frequency.
    """
    if len(G_ids) == 0:
        return 0.0
    if not beam_ids:
        return 1.0
    c_t = np.zeros(len(G_ids), dtype=int)
    for prev_ids in beam_ids:
        c_t += np.isin(G_ids, prev_ids).astype(int)
    weights = alpha ** c_t
    return np.mean(weights)


def beam_search(df, baseline_vars, beam_width, max_depth, alpha=0.5):
    """
    Beam search for exceptional subgroups with entropy and cover-based selection.

    Parameters:
        df (pd.DataFrame): Long-format dataset.
        baseline_vars (list): Time-invariant descriptor variables.
        beam_width (int): Width of the beam.
        max_depth (int): Maximum depth of condition combinations.
        alpha (float): Cover-based decay factor.

    Returns:
        top_improvers (list): Top subgroups with best improvement.
        top_deteriorators (list): Top subgroups with worst improvement.
    """
    global_b1, global_b2 = {}, {}
    scalers = {}

    # Standardize target variables
    for var in LINEAR_TARGETS + NONLINEAR_TARGETS:
        df[var] = pd.to_numeric(df[var], errors='coerce')
        mask = df[var].notna()
        scaler = StandardScaler()
        df.loc[mask, var] = scaler.fit_transform(df.loc[mask, [var]])
        scalers[var] = scaler

    for var in LINEAR_TARGETS:
        try:
            res = fit_linear_mixed_model(df, var)
            global_b1[var] = res.fe_params['time_numeric']
            global_b2[var] = 0.0
        except:
            global_b1[var], global_b2[var] = 0.0, 0.0

    for var in NONLINEAR_TARGETS:
        try:
            res = fit_quadratic_mixed_model(df, var)
            fe = res.fe_params
            global_b1[var] = fe.get('time_numeric', 0.0)
            global_b2[var] = fe.get('time_numeric_sq', 0.0)
        except:
            global_b1[var], global_b2[var] = 0.0, 0.0

    beam = [([], df, 0.0)]
    all_candidates = []
    beam_ids = []
    seen_rule_sets = set()
    conds = generate_conditions(df, baseline_vars)
    total_N = df['respondentid'].nunique()

    for depth in range(1, max_depth + 1):
        candidates = []
        for rule, subset, _ in beam:
            for cond in conds:
                if cond in rule:
                    continue
                new_rule = sorted(rule + [cond])
                rule_key = tuple(new_rule)
                if rule_key in seen_rule_sets:
                    continue
                seen_rule_sets.add(rule_key)
                df_sub = apply_condition(subset, cond)
                G_ids = df_sub['respondentid'].unique()
                n = len(G_ids)
                if n == 0:
                    continue
                try:
                    quality = 0.0
                    beta_deltas = {}
                    for var in LINEAR_TARGETS:
                        res = fit_linear_mixed_model(df_sub, var)
                        b1 = res.fe_params.get('time_numeric', 0.0)
                        delta = global_b1[var] - b1
                        beta_deltas[var] = {'beta1': b1, 'delta': delta}
                        quality += delta
                    for var in NONLINEAR_TARGETS:
                        res = fit_quadratic_mixed_model(df_sub, var)
                        fe = res.fe_params
                        b1 = fe.get('time_numeric', 0.0)
                        b2 = fe.get('time_numeric_sq', 0.0)
                        delta = 0.5 * ((global_b1[var] - b1) + (global_b2[var] - b2))
                        beta_deltas[var] = {'beta1': b1, 'beta2': b2, 'delta': delta}
                        quality += delta
                except:
                    continue
                entropy_weight = compute_entropy(n, total_N)
                cover_score = compute_cover_score(G_ids, beam_ids, alpha)
                adjusted_quality = cover_score * entropy_weight * quality
                candidates.append((new_rule, df_sub, adjusted_quality, G_ids, beta_deltas))

        candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
        beam = [(r, d, q) for r, d, q, _, _ in candidates[:beam_width]]
        beam_ids = [gids for _, _, _, gids, _ in candidates[:beam_width]]
        all_candidates.extend(candidates[:beam_width])

    # Separate top improvers and deteriorators
    top_improvers = sorted(all_candidates, key=lambda x: x[2], reverse=True)[:10]
    top_deteriorators = sorted(all_candidates, key=lambda x: x[2])[:10]

    return top_improvers, top_deteriorators