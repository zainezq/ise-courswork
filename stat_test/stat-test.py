import pandas as pd
import scipy.stats as stats

# Load and process data
def load_results(file_path):
    data = []
    with open(file_path, 'r') as file:
        system, dataset = None, None
        for line in file:
            if line.startswith("> System"):
                parts = line.strip().split(", ")
                system = parts[0].split(": ")[1]
                dataset = parts[1].split(": ")[1]
            elif line.startswith("Average"):
                metric, value = line.split(": ")
                data.append([system, dataset, metric.strip(), float(value.strip())])
    return pd.DataFrame(data, columns=["System", "Dataset", "Metric", "Value"])

# Load both baseline and new results
baseline_df = load_results("../output_base.txt")  
new_results_df = load_results("../output_proposed.txt") 

# Pivoting to have metrics as columns
baseline_pivot = baseline_df.pivot(index=["System", "Dataset"], columns="Metric", values="Value").reset_index()
new_results_pivot = new_results_df.pivot(index=["System", "Dataset"], columns="Metric", values="Value").reset_index()

# Merging datasets for comparison
comparison_df = pd.merge(baseline_pivot, new_results_pivot, on=["System", "Dataset"], suffixes=("_baseline", "_new"))

# Statistical Tests
metrics = ["Average MAPE", "Average MAE", "Average RMSE", "Average R²"]
stat_results = []

for metric in metrics:
    baseline_values = comparison_df[f"{metric}_baseline"]
    new_values = comparison_df[f"{metric}_new"]
    
    # Perform Shapiro-Wilk test for normality
    shapiro_baseline = stats.shapiro(baseline_values)[1]
    shapiro_new = stats.shapiro(new_values)[1]
    
    # If data is normally distributed, use paired t-test; otherwise, use Wilcoxon test
    if shapiro_baseline > 0.05 and shapiro_new > 0.05:
        test_stat, p_value = stats.ttest_rel(baseline_values, new_values)
        test_type = "Paired t-test"
    else:
        test_stat, p_value = stats.wilcoxon(baseline_values, new_values)
        test_type = "Wilcoxon Signed-Rank Test"
    
    stat_results.append([metric, test_type, test_stat, p_value])

# Convert results to DataFrame
stat_df = pd.DataFrame(stat_results, columns=["Metric", "Test Used", "Statistic", "p-value"])

# Generate LaTeX table
latex_table = stat_df.to_latex(index=False, column_format="|l|l|c|c|", escape=False)


# Convert p-value to scientific notation for better readability
stat_df["p-value"] = stat_df["p-value"].apply(lambda x: "{:.2e}".format(x))

# Generate the LaTeX table
latex_table = stat_df.to_latex(index=False, column_format="|l|l|c|c|", escape=False)
print(latex_table)
# Save the LaTeX table to a file
with open("statistical_results.tex", "w") as f:
    f.write(latex_table)