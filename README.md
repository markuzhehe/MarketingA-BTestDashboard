# Marketing A/B Test Dashboard
An enterprise-grade Python framework and command-line interface (CLI) for automating A/B test analysis. This tool validates experiment integrity by checking for common issues—such as traffic allocation bugs, distribution violations, and inflated false positives—before generating final statistical results.

## Features
* **Sample Ratio Mismatch (SRM) Detection**
    Uses a Chi-Square test to verify that traffic is split as expected and to flag potential tracking or allocation issues.
* **Automated Statistical Test Selection**
    Checks for normality using the Shapiro-Wilk test and automatically routes to the appropriate statistical test:
    * Welch’s T-Test for approximately normal data
    * Mann-Whitney U Test for skewed or non-normal data
* **Segmented Testing Protection**
    Supports subgroup analysis (e.g., by platform or country) and applies Bonferroni correction to reduce false positives caused by multiple comparisons.
* **Automated Visualization**
    Generates and saves a 95% confidence interval plot for experiment results without interrupting terminal execution.

## Tech Stack
* Language: Python 3.x
* Libraries: Pandas, NumPy, SciPy, Matplotlib, Seaborn

## Example Command
python ab.py --data experiment_data.csv --variant_col variant --metric_col revenue --segment_col platform

```text
==============================================
EXPERIMENTAL INFERENCE PIPELINE OUTPUT
==============================================
SRM Integrity:          No SRM Detected (p-value: 0.48210)
Distribution Metric:    Non-Parametric (Mann-Whitney U)
Applied Math Framework: Mann-Whitney U Test
Calculated p-value:     0.00341
Calculated Effect Size: 0.1840 (Cohen's d)
Absolute Uplift Delta:  1.4205
Visual Diagnostics:     Saved 95% CI plot asset to -> ./ab_test_confidence_intervals.png

----------------------------------------------
SEGMENTED ANALYSIS BREAKDOWN (Col: platform)
----------------------------------------------
Segment      Raw_p_value    Uplift    Adjusted_p_value
Mobile       0.00120        2.1050    0.00360
Desktop      0.34012        0.2100    1.00000
Tablet       0.04510        1.1500    0.13530
==============================================
