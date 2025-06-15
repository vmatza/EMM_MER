# Exceptional Model Mining in Longitudinal Data using Mixed-Effects Regressions

This repository contains the codebase for the Evangelos Matzavinos' Bachelor End Project on **Exceptional Model Mining (EMM)** integrated with **Mixed-Effects Regression (MER)** for longitudinal subgroup discovery.

## Project Summary

We analyze patient trajectories from the DEPAR study to discover subgroups that exhibit exceptionally steep or slow improvements from the average response to Methotrexate (MTX) treatment. The framework integrates:

- Linear and quadratic Mixed-Effects Models
- Z-score normalization across outcomes
- A custom quality measure based on regression coefficients
- Beam search with entropy-based weighting and cover-based diversity


## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/emm-mer-subgroup-discovery.git
   cd emm-mer-subgroup-discovery

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Make sure to update data_path in main.py with the actual path to your .dta file.

4. Run the main pipeline:
    ```bash
    python main.py

## Thesis
This code supports the thesis: "Exceptional Model Mining in Longitudinal Data using Mixed-Effects Regressions." A PDF of the thesis is available upon request.

If anything is unclear or if you have any questions, please feel free to reach out.