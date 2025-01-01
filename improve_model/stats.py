import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ensure the figures directory exists
figures_dir = "./figures/new"
os.makedirs(figures_dir, exist_ok=True)

# Function to plot feature distributions individually and combined
def plot_distributions(data, features, title_prefix, combined_fig_name=None):
    # Individual plots
    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.histplot(data, x=feature, hue="attack", kde=True, bins=30, palette="gray", alpha=0.7)
        plt.title(f"{title_prefix} Distribution: {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend(title="Attack", labels=["No Attack", "Attack"])
        plt.tight_layout()
        save_path = os.path.join(figures_dir, f"{feature}_distribution.jpeg")
        plt.savefig(save_path, dpi=300, format="jpeg")
        plt.close()
    
    # Combined plot
    fig, axes = plt.subplots(len(features), 1, figsize=(8, 5 * len(features)))
    for i, feature in enumerate(features):
        sns.histplot(data, x=feature, hue="attack", kde=True, bins=30, palette="gray", alpha=0.7, ax=axes[i])
        axes[i].set_title(f"{title_prefix} Distribution: {feature}")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Frequency")
        axes[i].legend(title="Attack", labels=["No Attack", "Attack"])
    plt.tight_layout()
    if combined_fig_name:
        plt.savefig(combined_fig_name, dpi=300, format="jpeg")
    plt.close()

# Function to plot boxplots for features individually and combined
def plot_boxplots(data, features, title_prefix, combined_fig_name=None):
    # Individual plots
    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=data, x="attack", y=feature, palette="gray", notch=True)
        plt.title(f"{title_prefix} Boxplot: {feature}")
        plt.xlabel("Attack (0: No Attack, 1: Attack)")
        plt.ylabel(feature)
        plt.tight_layout()
        save_path = os.path.join(figures_dir, f"{feature}_boxplot.jpeg")
        plt.savefig(save_path, dpi=300, format="jpeg")
        plt.close()
    
    # Combined plot
    fig, axes = plt.subplots(len(features), 1, figsize=(8, 5 * len(features)))
    for i, feature in enumerate(features):
        sns.boxplot(data=data, x="attack", y=feature, palette="gray", notch=True, ax=axes[i])
        axes[i].set_title(f"{title_prefix} Boxplot: {feature}")
        axes[i].set_xlabel("Attack (0: No Attack, 1: Attack)")
        axes[i].set_ylabel(feature)
    plt.tight_layout()
    if combined_fig_name:
        plt.savefig(combined_fig_name, dpi=300, format="jpeg")
    plt.close()

# Key features for visualization
key_features = ["cpu_sec_total", "cpu_sec_idle", "resident_memory_total", "virtual_memory_total", "open_fds"]

# Load the combined dataset
combined_data = pd.read_csv("./data/final_data_combined.csv")

# Plot feature distributions
plot_distributions(
    combined_data, 
    key_features, 
    "Feature", 
    combined_fig_name=os.path.join(figures_dir, "combined_feature_distributions.jpeg")
)

# Plot boxplots for key features
plot_boxplots(
    combined_data, 
    key_features, 
    "Feature", 
    combined_fig_name=os.path.join(figures_dir, "combined_feature_boxplots.jpeg")
)
