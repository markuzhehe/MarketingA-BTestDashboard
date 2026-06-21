import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ABTestEngine:
    def __init__(self, data: pd.DataFrame, variant_col: str, metric_col: str):
        self.df = data.copy()
        self.variant_col = variant_col
        self.metric_col = metric_col
        
        # Split groups for internal use
        self.variants = self.df[self.variant_col].unique()
        if len(self.variants) != 2:
            raise ValueError("This engine currently requires exactly two variants (e.g., Control & Treatment).")
            
        self.g1 = self.df[self.df[self.variant_col] == self.variants[0]][self.metric_col]
        self.g2 = self.df[self.df[self.variant_col] == self.variants[1]][self.metric_col]

    @staticmethod
    def calculate_sample_size(p1: float, mde: float, alpha: float = 0.05, power: float = 0.80) -> int:
        """Calculates required sample size per variant for a binary conversion metric."""
        p2 = p1 + mde
        p_bar = (p1 + p2) / 2
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        n = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2 / (mde**2)
        return int(np.ceil(n))

    def run_srm_check(self, expected_ratio: float = 0.5) -> dict:
        """Checks for Sample Ratio Mismatch using a Chi-Square Goodness-of-Fit test."""
        obs_c = len(self.g1)
        obs_t = len(self.g2)
        total = obs_c + obs_t
        
        exp_c = total * expected_ratio
        exp_t = total * (1 - expected_ratio)
        
        chisq, p_val = stats.chisquare([obs_c, obs_t], f_exp=[exp_c, exp_t])
        srm_detected = p_val < 0.01
        
        return {
            "p_value": p_val,
            "srm_detected": srm_detected,
            "status": "SRM DETECTED (Tracking Bug Risk)" if srm_detected else "No SRM Detected"
        }

    def check_normality(self) -> dict:
        """Runs a Shapiro-Wilk normality test (subsamples to 5000 if large)."""
        p_a = stats.shapiro(self.g1.sample(min(len(self.g1), 5000))).pvalue
        p_b = stats.shapiro(self.g2.sample(min(len(self.g2), 5000))).pvalue
        is_normal = (p_a > 0.05) and (p_b > 0.05)
        
        return {
            f"{self.variants[0]}_p_val": p_a,
            f"{self.variants[1]}_p_val": p_b,
            "is_normal": is_normal,
            "recommended_test": "Parametric (T-Test)" if is_normal else "Non-Parametric (Mann-Whitney U)"
        }

    def run_hypothesis_test(self, alternative: str = 'two-sided') -> dict:
        """Auto-detects data distribution and runs the mathematically appropriate test."""
        normality = self.check_normality()
        
        if normality['is_normal']:
            stat, p_val = stats.ttest_ind(self.g2, self.g1, alternative=alternative, equal_var=False)
            test_name = "Welch's T-Test"
        else:
            stat, p_val = stats.mannwhitneyu(self.g2, self.g1, alternative=alternative)
            test_name = "Mann-Whitney U Test"
            
        # Calculate Cohen's d effect size
        cohens_d = (self.g2.mean() - self.g1.mean()) / np.sqrt((self.g1.var() + self.g2.var()) / 2)
        
        return {
            "applied_test": test_name,
            "p_value": p_val,
            "cohens_d": cohens_d,
            "mean_difference": self.g2.mean() - self.g1.mean()
        }

    def run_segmented_testing(self, segment_col: str, correction: str = "bonferroni") -> pd.DataFrame:
        """Evaluates results across segments and controls for Multiple Testing False Positives."""
        segments = self.df[segment_col].unique()
        results = []
        
        for seg in segments:
            seg_data = self.df[self.df[segment_col] == seg]
            g_c = seg_data[seg_data[self.variant_col] == self.variants[0]][self.metric_col]
            g_t = seg_data[seg_data[self.variant_col] == self.variants[1]][self.metric_col]
            
            if len(g_c) > 5 and len(g_t) > 5:
                _, p = stats.ttest_ind(g_t, g_c, equal_var=False)
                results.append({"Segment": seg, "Raw_p_value": p, "Uplift": g_t.mean() - g_c.mean()})
                
        res_df = pd.DataFrame(results)
        
        if res_df.empty:
            return res_df
            
        # Apply Corrections
        if correction.lower() == "bonferroni":
            res_df['Adjusted_p_value'] = np.minimum(res_df['Raw_p_value'] * len(res_df), 1.0)
        else:
            res_df['Adjusted_p_value'] = res_df['Raw_p_value']
            
        return res_df

    def save_confidence_intervals_plot(self, output_path: str = "ci_plot.png"):
        """Generates the 95% CI plot and explicitly saves it as an image asset."""
        mean_diff = self.g2.mean() - self.g1.mean()
        se_diff = np.sqrt(self.g1.var()/len(self.g1) + self.g2.var()/len(self.g2))
        ci_lower, ci_upper = stats.norm.interval(0.95, loc=mean_diff, scale=se_diff)
        
        import matplotlib
        matplotlib.use('Agg')
        
        plt.figure(figsize=(8, 2.5))
        plt.errorbar(mean_diff, 0, xerr=[[mean_diff - ci_lower], [ci_upper - mean_diff]], 
                     fmt='o', color='royalblue', elinewidth=3, capsize=5, label='95% Confidence Interval')
        plt.axvline(0, color='red', linestyle='--', alpha=0.6, label='No Effect Line')
        plt.title("95% Confidence Interval for Mean Difference")
        plt.xlabel("Uplift Delta")
        plt.yticks([])
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


# --- CLI EXECUTOR PIPELINE ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced A/B Testing Engine CLI Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to experimental CSV data")
    parser.add_argument("--variant_col", type=str, default="variant", help="Column name for target groups")
    parser.add_argument("--metric_col", type=str, required=True, help="Column name for quantitative metric")
    parser.add_argument("--segment_col", type=str, default=None, help="Optional column name for segmented analysis evaluation")
    
    args = parser.parse_args()
    
    try:
        raw_df = pd.read_csv(args.data)
        engine = ABTestEngine(data=raw_df, variant_col=args.variant_col, metric_col=args.metric_col)
        
        print("\n==============================================")
        print("EXPERIMENTAL INFERENCE PIPELINE OUTPUT")
        print("==============================================")
        
        # 1. Traffic Check
        srm = engine.run_srm_check()
        print(f"SRM Integrity: {srm['status']} (p-value: {srm['p_value']:.5f})")
        
        # 2. Distribution Check
        norm = engine.check_normality()
        print(f"Distribution Metric: {norm['recommended_test']}")
        
        # 3. Main Statistical Core Output
        results = engine.run_hypothesis_test()
        print(f"Applied Math Framework: {results['applied_test']}")
        print(f"Calculated p-value: {results['p_value']:.5f}")
        print(f"Calculated Effect Size (Cohen's d): {results['cohens_d']:.4f}")
        print(f"Absolute Uplift Delta: {results['mean_difference']:.4f}")
        
        # 4. Save visualization plot to file system
        plot_filename = "ab_test_confidence_intervals.png"
        engine.save_confidence_intervals_plot(output_path=plot_filename)
        print(f"Visual Diagnostics: Saved 95% CI plot asset to -> ./{plot_filename}")
        
        # 5. Segmented Reporting (If requested via CLI)
        if args.segment_col:
            print("\n----------------------------------------------")
            print(f"SEGMENTED ANALYSIS BREAKDOWN (Col: {args.segment_col})")
            print("----------------------------------------------")
            seg_df = engine.run_segmented_testing(segment_col=args.segment_col, correction="bonferroni")
            if seg_df.empty:
                print("No segments met the criteria of >5 records per variant group.")
            else:
                print(seg_df.to_string(index=False))
                
        print("==============================================\n")
        
    except Exception as e:
        print(f"Execution pipeline failed: {e}")