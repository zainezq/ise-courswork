import re
import pandas as pd

# File paths
baseline_file = "../output_base.txt"
proposed_file = "../output_proposed.txt"

# Function to parse the input files and extract metrics
def parse_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    
    pattern = re.compile(
        r"System:\s*(?P<system>[\w\d-]+),\s*Dataset:\s*(?P<dataset>[\w\d.-]+).*?"
        r"Average MAPE:\s*(?P<mape>\d+\.\d+).*?"
        r"Average MAE:\s*(?P<mae>\d+\.\d+).*?"
        r"Average RMSE:\s*(?P<rmse>\d+\.\d+).*?"
        r"Average R²:\s*(?P<r2>\d+\.\d+)",
        re.DOTALL
    )

    data = []
    for match in pattern.finditer(content):
        data.append({
            "System": match.group("system"),
            "MAPE": float(match.group("mape")),
            "MAE": float(match.group("mae")),
            "RMSE": float(match.group("rmse")),
            "R²": float(match.group("r2")),
        })
    
    return pd.DataFrame(data)

# Parse both files
df_baseline = parse_file(baseline_file)
df_proposed = parse_file(proposed_file)

# Group by system and compute the mean
df_baseline_avg = df_baseline.groupby("System", as_index=False).mean()
df_proposed_avg = df_proposed.groupby("System", as_index=False).mean()

# Merge data based on system
df_comparison = df_baseline_avg.merge(df_proposed_avg, on="System", suffixes=("_Baseline", "_Proposed"))

# Calculate improvement
df_comparison["MAPE_Improvement"] = ((df_comparison["MAPE_Baseline"] - df_comparison["MAPE_Proposed"]) / df_comparison["MAPE_Baseline"]) * 100
df_comparison["MAE_Improvement"] = ((df_comparison["MAE_Baseline"] - df_comparison["MAE_Proposed"]) / df_comparison["MAE_Baseline"]) * 100
df_comparison["RMSE_Improvement"] = ((df_comparison["RMSE_Baseline"] - df_comparison["RMSE_Proposed"]) / df_comparison["RMSE_Baseline"]) * 100
df_comparison.rename(columns={"R²_Baseline": "R2_Baseline", "R²_Proposed": "R2_Proposed", "R²_Improvement": "R2_Improvement"}, inplace=True)
df_comparison["R2_Improvement"] = ((df_comparison["R2_Proposed"] - df_comparison["R2_Baseline"]) / df_comparison["R2_Baseline"]) * 100

# Select only 9 unique systems
df_comparison = df_comparison.head(9)

# Function to generate LaTeX table
def generate_latex_table(df, metric, caption, label):
    latex_code = (
        r"\begin{table}[h]" "\n"
        r"    \centering" "\n"
        r"    \begin{tabular}{|c|c|c|c|}" "\n"
        r"        \hline" "\n"
        f"        System & Baseline {metric} & Proposed {metric} & Improvement (\\%) \\\\" "\n"
        r"        \hline" "\n"
    )
    
    for _, row in df.iterrows():
        latex_code += f"        {row['System']} & {row[metric + '_Baseline']:.2f} & {row[metric + '_Proposed']:.2f} & {row[metric + '_Improvement']:.2f} \\\\\n"
    
    latex_code += (
        r"        \hline" "\n"
        r"    \end{tabular}" "\n"
        f"    \\caption{{{caption}}}" "\n"
        f"    \\label{{{label}}}" "\n"
        r"\end{table}"
    )
    
    return latex_code

# Generate LaTeX tables
latex_mape = generate_latex_table(df_comparison, "MAPE", "Comparison of MAPE between Linear Regression and Proposed Model", "tab:mape")
latex_mae = generate_latex_table(df_comparison, "MAE", "Comparison of MAE between Linear Regression and Proposed Model", "tab:mae")
latex_rmse = generate_latex_table(df_comparison, "RMSE", "Comparison of RMSE between Linear Regression and Proposed Model", "tab:rmse")
latex_r2 = generate_latex_table(df_comparison, "R2", "Comparison of R² Score between Linear Regression and Proposed Model", "tab:r2")

# Print LaTeX tables
print(latex_mape)
print("\n" + "-" * 80 + "\n")
print(latex_mae)
print("\n" + "-" * 80 + "\n")
print(latex_rmse)
print("\n" + "-" * 80 + "\n")
print(latex_r2)
