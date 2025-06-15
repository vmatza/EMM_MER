import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_loading import load_data
from preprocessing import preprocess_data
from utils import make_output_dirs
from modeling import (
    fit_linear_mixed_model,
    fit_quadratic_mixed_model,
    extract_model_summary
)
from visualizations import (
    plot_random_variable,
    plot_small_multiples,
    normality_report,
    plot_model_fits
)
from emm import (
    beam_search,
    LINEAR_TARGETS,
    NONLINEAR_TARGETS,
    apply_condition
)

def main():

    make_output_dirs()
    data_path = '...'
    df = load_data(data_path)
    df = preprocess_data(df)
    df_mtx = df[df['mtx_cov'] == 1]
    print(f'MTX subgroup: {df_mtx["respondentid"].nunique()} unique patients')

    targets = LINEAR_TARGETS + NONLINEAR_TARGETS
    baseline_vars = [
        'pl61_T0','bmi_T0','pm86b_T0','pm83b_T0','Geslacht_T0','age_T0',
        'diagnose_T0','fa04_T0','fa12_T0','pm112_T0','fa20_T0','fa21_T0',
        'fa22_T0','fa15_T0','fa16_T0','fa19a_T0'
    ]


    ############ Normality Report ############

    normality_results = normality_report(df_mtx, targets)
    print("Normality Test Results:")
    print(normality_results)


    ############ Visualize Linear vs Quadratic Fit for a specific Variable ############

    lin_res = fit_linear_mixed_model(df_mtx, variable='gp11')
    quad_res = fit_quadratic_mixed_model(df_mtx, variable='gp11')
    plot_model_fits(df_mtx, lin_res, quad_res, variable='gp11')


    ############ Standardize Targets and Fit Global Models ############

    # Standardize target variables
    scalers = {}
    for var in targets:
        df_mtx[var] = pd.to_numeric(df_mtx[var], errors='coerce')
        mask = df_mtx[var].notna()
        scaler = StandardScaler()
        df_mtx.loc[mask, var] = scaler.fit_transform(df_mtx.loc[mask, [var]])
        scalers[var] = scaler

    # Fit models on standardized data
    standardized_summaries = []
    for var in LINEAR_TARGETS:
        res = fit_linear_mixed_model(df_mtx, var)
        standardized_summaries.append(extract_model_summary(res, var, nonlinear=False))
    for var in NONLINEAR_TARGETS:
        res = fit_quadratic_mixed_model(df_mtx, var)
        standardized_summaries.append(extract_model_summary(res, var, nonlinear=True))
    global_summary_df = pd.DataFrame(standardized_summaries)
    print("Standardized Global Coefficients:")
    print(global_summary_df)


    ########### Beam search for exceptional subgroups ############

    top_improvers, top_deteriorators = beam_search(df_mtx, baseline_vars,
                        beam_width=20, max_depth=3)

    print("\nTop Subgroups That Improve More Than Average:")
    for i, (conds, subset, score, _, deltas) in enumerate(top_improvers, 1):
        cond_str = " ∧ ".join(f"{v}{op}{val}" for v, op, val in conds)
        size = subset['respondentid'].nunique()
        print(f"{i}. If {cond_str}  WITH  n={size}  score={score:.3f}")
        for var, b in deltas.items():
            print(f"   - {var}: β₁={b['beta1']:.3f}", end="")
            if 'beta2' in b:
                print(f", β₂={b['beta2']:.3f}")
            else:
                print("")

    print("\nTop Subgroups That Worsen Compared to Average:")
    for i, (conds, subset, score, _, deltas) in enumerate(top_deteriorators, 1):
        cond_str = " ∧ ".join(f"{v}{op}{val}" for v, op, val in conds)
        size = subset['respondentid'].nunique()
        print(f"{i}. If {cond_str}  WITH  n={size}  score={score:.3f}")
        for var, b in deltas.items():
            print(f"   - {var}: β₁={b['beta1']:.3f}", end="")
            if 'beta2' in b:
                print(f", β₂={b['beta2']:.3f}")
            else:
                print("")


    ############ Visualize Global vs Subgroup Trajectories ############

    subgroup_conditions = [('bmi_T0', '<=', 28.125), ('pl61_T0', '>', 97.0)]
    df_sub = df_mtx.copy()
    for cond in subgroup_conditions:
        df_sub = apply_condition(df_sub, cond)

    print(f"\nSubgroup size: {df_sub['respondentid'].nunique()} patients")

    subgroup_summaries = []
    for var in LINEAR_TARGETS:
        res = fit_linear_mixed_model(df_sub, var)
        subgroup_summaries.append(extract_model_summary(res, var, nonlinear=False))
    for var in NONLINEAR_TARGETS:
        res = fit_quadratic_mixed_model(df_sub, var)
        subgroup_summaries.append(extract_model_summary(res, var, nonlinear=True))
    sub_summary_df = pd.DataFrame(subgroup_summaries)

    print("\nPlotting global vs. subgroup trajectories…")
    plot_small_multiples(global_summary_df, sub_summary_df)


if __name__ == '__main__':
    main()
