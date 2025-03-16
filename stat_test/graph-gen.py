import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths
baseline_file = "../output_base.txt"
proposed_file = "../results1.txt"

# Function to parse the input files and extract metrics
def parse_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    
    pattern = re.compile(
        r"System:\s*(?P<system>[\w\d-]+),\s*Dataset:\s*(?P<dataset>[\w\d.-]+).*?"
        r"Average MAPE:\s*(?P<mape>\d+\.\d+).*?"
        r"Average MAE:\s*(?P<mae>\d+\.\d+).*?"
        r"Average RMSE:\s*(?P<rmse>\d+\.\d+)",
        re.DOTALL
    )

    data = []
    for match in pattern.finditer(content):
        data.append({
            "System": match.group("system"),
            "MAPE": float(match.group("mape")),
            "MAE": float(match.group("mae")),
            "RMSE": float(match.group("rmse")),
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

# Select only 9 unique systems
df_comparison = df_comparison.head(9)
# Plotting all three metrics using line plots instead of bar charts

# Split the dataset into two: one with H2 and one without H2
df_with_h2 = df_comparison[df_comparison["System"] == "h2"]
df_without_h2 = df_comparison[df_comparison["System"] != "h2"]

# Function to plot the comparison graph
def plot_comparison(df, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df["System"]))

    # Plot lines for MAPE, MAE, and RMSE
    ax.plot(x, df["MAPE_Baseline"], marker='o', linestyle='-', color='blue', label="Baseline MAPE")
    ax.plot(x, df["MAE_Baseline"], marker='s', linestyle='-', color='orange', label="Baseline MAE")
    ax.plot(x, df["RMSE_Baseline"], marker='^', linestyle='-', color='red', label="Baseline RMSE")

    ax.plot(x, df["MAPE_Proposed"], marker='o', linestyle='--', color='cyan', label="Proposed MAPE")
    ax.plot(x, df["MAE_Proposed"], marker='s', linestyle='--', color='yellow', label="Proposed MAE")
    ax.plot(x, df["RMSE_Proposed"], marker='^', linestyle='--', color='pink', label="Proposed RMSE")

    # Labels and title
    ax.set_xlabel("System")
    ax.set_ylabel("Error Metrics")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df["System"], rotation=45, ha="right")
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

# Plot graphs separately
plot_comparison(df_without_h2, "Comparison of MAPE, MAE, and RMSE Without H2")
plot_comparison(df_with_h2, "Comparison of MAPE, MAE, and RMSE for H2")

