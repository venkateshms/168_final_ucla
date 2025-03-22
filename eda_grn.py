import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap


plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.3)
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Create custom color palettes
colors = ["#4B88A2", "#252839", "#D72638", "#F49D37", "#3F7D20"]
custom_palette = sns.color_palette(colors)
sns.set_palette(custom_palette)

# Create a custom colormap for heatmaps
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#4B88A2", "#F49D37", "#D72638"])

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Perform exploratory data analysis on GRN data')
    parser.add_argument('--input', type=str, default='SCING_GRN.csv',
                       help='Path to the input GRN CSV file')
    parser.add_argument('--output-dir', type=str, default='eda_plots',
                       help='Directory to save visualization outputs')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading GRN data from {args.input}...")
    
    # Load data
    grn_data = pd.read_csv(args.input)
    print(f"GRN data shape: {grn_data.shape}")
    print("GRN data columns:", grn_data.columns.tolist())
    print("GRN data sample:")
    print(grn_data.head())

    # Basic statistics
    print("\n--- GRN Statistics ---")
    print(grn_data.describe())

    # Analyze weight distributions
    plt.figure()
    ax = sns.histplot(grn_data['importance'], bins=50, kde=True, color=colors[0], 
                     line_kws={'linewidth': 3, 'color': colors[2]})
    plt.title('Distribution of Edge Weights in GRN', pad=20)
    plt.xlabel('Importance (Weight)', labelpad=15)
    plt.ylabel('Frequency', labelpad=15)
    # Add grid but only on y-axis
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add a text box with statistics
    textstr = f"Mean: {grn_data['importance'].mean():.2f}\nMedian: {grn_data['importance'].median():.2f}\nMax: {grn_data['importance'].max():.2f}"
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=props, fontsize=12, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'grn_weight_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Analyze log-transformed weight distributions
    plt.figure()
    ax = sns.histplot(np.log10(grn_data['importance']), bins=50, kde=True, color=colors[1],
                     line_kws={'linewidth': 3, 'color': colors[3]})
    plt.title('Log10 Distribution of Edge Weights in GRN', pad=20)
    plt.xlabel('Log10(Importance)', labelpad=15)
    plt.ylabel('Frequency', labelpad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add a text box with statistics
    log_weights = np.log10(grn_data['importance'])
    textstr = f"Mean: {log_weights.mean():.2f}\nMedian: {log_weights.median():.2f}\nStd Dev: {log_weights.std():.2f}"
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=props, fontsize=12, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'grn_log_weight_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Count unique TFs and targets
    grn_unique_tfs = grn_data['TF'].nunique()
    grn_unique_targets = grn_data['target'].nunique()

    print(f"\nGRN has {grn_unique_tfs} unique TFs and {grn_unique_targets} unique targets")

    # Analyze TF connectivity (number of targets per TF)
    grn_tf_counts = grn_data['TF'].value_counts()

    plt.figure()
    ax = sns.histplot(grn_tf_counts, bins=50, kde=True, color=colors[4],
                     line_kws={'linewidth': 3, 'color': colors[0]})
    plt.title('Distribution of Number of Targets per TF in GRN', pad=20)
    plt.xlabel('Number of Targets', labelpad=15)
    plt.ylabel('Number of TFs', labelpad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add a text box with statistics
    textstr = f"Mean: {grn_tf_counts.mean():.2f}\nMedian: {grn_tf_counts.median():.2f}\nMax: {grn_tf_counts.max()}"
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=props, fontsize=12, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'grn_tf_connectivity.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Top TFs by number of targets
    print("\nTop 10 TFs with most targets in GRN:")
    top_tfs = grn_tf_counts.head(10)
    print(top_tfs)

    # Create a bar plot for top TFs
    plt.figure()
    ax = sns.barplot(x=top_tfs.index, y=top_tfs.values, palette=sns.color_palette("viridis", len(top_tfs)))
    plt.title('Top 10 TFs with Most Targets in GRN', pad=20)
    plt.xlabel('Transcription Factor', labelpad=15)
    plt.ylabel('Number of Target Genes', labelpad=15)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(top_tfs.values):
        plt.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'grn_top_tfs.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Weight threshold analysis for GRN
    percentiles = [50, 75, 90, 95, 99]
    thresholds = np.percentile(grn_data['importance'], percentiles)

    print("\nGRN Weight Thresholds:")
    for p, t in zip(percentiles, thresholds):
        count = sum(grn_data['importance'] > t)
        percent = count / len(grn_data) * 100
        print(f"Percentile {p}%: {t:.2f} - Keeps {count} edges ({percent:.2f}% of total)")

    # Create a plot showing the number of edges retained at different thresholds
    percentiles_fine = np.arange(0, 100, 1)
    thresholds_fine = np.percentile(grn_data['importance'], percentiles_fine)
    edges_retained = [sum(grn_data['importance'] > t) for t in thresholds_fine]
    percent_retained = [count / len(grn_data) * 100 for count in edges_retained]

    plt.figure()
    ax1 = plt.gca()
    ax1.plot(percentiles_fine, edges_retained, linewidth=3, color=colors[2])
    ax1.set_xlabel('Percentile Threshold', labelpad=15)
    ax1.set_ylabel('Number of Edges Retained', color=colors[2], labelpad=15)
    ax1.tick_params(axis='y', labelcolor=colors[2])

    # Add a second y-axis for percentage
    ax2 = ax1.twinx()
    ax2.plot(percentiles_fine, percent_retained, linewidth=3, linestyle='--', color=colors[0])
    ax2.set_ylabel('Percentage of Edges Retained', color=colors[0], labelpad=15)
    ax2.tick_params(axis='y', labelcolor=colors[0])

    plt.title('Edge Retention vs. Weight Percentile Threshold (GRN)', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add vertical lines at key percentiles
    for p, t in zip(percentiles, thresholds):
        plt.axvline(x=p, color='gray', linestyle=':', alpha=0.7)
        count = sum(grn_data['importance'] > t)
        plt.text(p+0.5, edges_retained[percentiles_fine.tolist().index(p)], 
                 f"{p}%: {count} edges", 
                 verticalalignment='bottom', 
                 fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'grn_edge_retention.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Analyze target connectivity (number of TFs regulating each target)
    grn_target_counts = grn_data['target'].value_counts()

    plt.figure()
    ax = sns.histplot(grn_target_counts, bins=50, kde=True, color=colors[3],
                     line_kws={'linewidth': 3, 'color': colors[1]})
    plt.title('Distribution of Number of Regulating TFs per Target in GRN', pad=20)
    plt.xlabel('Number of Regulating TFs', labelpad=15)
    plt.ylabel('Number of Targets', labelpad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add a text box with statistics
    textstr = f"Mean: {grn_target_counts.mean():.2f}\nMedian: {grn_target_counts.median():.2f}\nMax: {grn_target_counts.max()}"
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=props, fontsize=12, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'grn_target_connectivity.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Top targets by number of regulating TFs
    print("\nTop 10 targets with most regulating TFs in GRN:")
    top_targets = grn_target_counts.head(10)
    print(top_targets)

    # Create a bar plot for top targets
    plt.figure()
    ax = sns.barplot(x=top_targets.index, y=top_targets.values, palette=sns.color_palette("magma", len(top_targets)))
    plt.title('Top 10 Targets with Most Regulating TFs in GRN', pad=20)
    plt.xlabel('Target Gene', labelpad=15)
    plt.ylabel('Number of Regulating TFs', labelpad=15)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(top_targets.values):
        plt.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'grn_top_targets.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Network density analysis
    total_possible_edges = grn_unique_tfs * grn_unique_targets
    actual_edges = len(grn_data)
    network_density = actual_edges / total_possible_edges

    print(f"\nNetwork Analysis:")
    print(f"Total possible edges (TFs × targets): {total_possible_edges}")
    print(f"Actual edges in network: {actual_edges}")
# Load the data
print("Loading SCING GRN data...")
scing_path = '/Users/madhavanvenkatesh/Desktop/168_project/SCING_GRN.csv'

# Load SCING data
scing_grn = pd.read_csv(scing_path)
print(f"SCING GRN shape: {scing_grn.shape}")
print("SCING GRN columns:", scing_grn.columns.tolist())
print("SCING GRN sample:")
print(scing_grn.head())

# Basic statistics
print("\n--- SCING GRN Statistics ---")
print(scing_grn.describe())

# Create output directory for plots
os.makedirs('eda_plots', exist_ok=True)

# Analyze weight distributions
plt.figure()
ax = sns.histplot(scing_grn['importance'], bins=50, kde=True, color=colors[0], 
                 line_kws={'linewidth': 3, 'color': colors[2]})
plt.title('Distribution of Edge Weights in SCING GRN', pad=20)
plt.xlabel('Importance (Weight)', labelpad=15)
plt.ylabel('Frequency', labelpad=15)
# Add grid but only on y-axis
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Add a text box with statistics
textstr = f"Mean: {scing_grn['importance'].mean():.2f}\nMedian: {scing_grn['importance'].median():.2f}\nMax: {scing_grn['importance'].max():.2f}"
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
            bbox=props, fontsize=12, verticalalignment='top')
plt.tight_layout()
plt.savefig('eda_plots/scing_weight_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyze log-transformed weight distributions
plt.figure()
ax = sns.histplot(np.log10(scing_grn['importance']), bins=50, kde=True, color=colors[1],
                 line_kws={'linewidth': 3, 'color': colors[3]})
plt.title('Log10 Distribution of Edge Weights in SCING GRN', pad=20)
plt.xlabel('Log10(Importance)', labelpad=15)
plt.ylabel('Frequency', labelpad=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Add a text box with statistics
log_weights = np.log10(scing_grn['importance'])
textstr = f"Mean: {log_weights.mean():.2f}\nMedian: {log_weights.median():.2f}\nStd Dev: {log_weights.std():.2f}"
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
            bbox=props, fontsize=12, verticalalignment='top')
plt.tight_layout()
plt.savefig('eda_plots/scing_log_weight_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Count unique TFs and targets
scing_unique_tfs = scing_grn['TF'].nunique()
scing_unique_targets = scing_grn['target'].nunique()

print(f"\nSCING GRN has {scing_unique_tfs} unique TFs and {scing_unique_targets} unique targets")

# Analyze TF connectivity (number of targets per TF)
scing_tf_counts = scing_grn['TF'].value_counts()

plt.figure()
ax = sns.histplot(scing_tf_counts, bins=50, kde=True, color=colors[4],
                 line_kws={'linewidth': 3, 'color': colors[0]})
plt.title('Distribution of Number of Targets per TF in SCING GRN', pad=20)
plt.xlabel('Number of Targets', labelpad=15)
plt.ylabel('Number of TFs', labelpad=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Add a text box with statistics
textstr = f"Mean: {scing_tf_counts.mean():.2f}\nMedian: {scing_tf_counts.median():.2f}\nMax: {scing_tf_counts.max()}"
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
            bbox=props, fontsize=12, verticalalignment='top')
plt.tight_layout()
plt.savefig('eda_plots/scing_tf_connectivity.png', dpi=300, bbox_inches='tight')
plt.show()

# Top TFs by number of targets
print("\nTop 10 TFs with most targets in SCING GRN:")
top_tfs = scing_tf_counts.head(10)
print(top_tfs)

# Create a bar plot for top TFs
plt.figure()
ax = sns.barplot(x=top_tfs.index, y=top_tfs.values, palette=sns.color_palette("viridis", len(top_tfs)))
plt.title('Top 10 TFs with Most Targets in SCING GRN', pad=20)
plt.xlabel('Transcription Factor', labelpad=15)
plt.ylabel('Number of Target Genes', labelpad=15)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(top_tfs.values):
    plt.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/scing_top_tfs.png', dpi=300, bbox_inches='tight')
plt.show()

# Weight threshold analysis for SCING GRN
percentiles = [50, 75, 90, 95, 99]
thresholds = np.percentile(scing_grn['importance'], percentiles)

print("\nSCING GRN Weight Thresholds:")
for p, t in zip(percentiles, thresholds):
    count = sum(scing_grn['importance'] > t)
    percent = count / len(scing_grn) * 100
    print(f"Percentile {p}%: {t:.2f} - Keeps {count} edges ({percent:.2f}% of total)")

# Create a plot showing the number of edges retained at different thresholds
percentiles_fine = np.arange(0, 100, 1)
thresholds_fine = np.percentile(scing_grn['importance'], percentiles_fine)
edges_retained = [sum(scing_grn['importance'] > t) for t in thresholds_fine]
percent_retained = [count / len(scing_grn) * 100 for count in edges_retained]

plt.figure()
ax1 = plt.gca()
ax1.plot(percentiles_fine, edges_retained, linewidth=3, color=colors[2])
ax1.set_xlabel('Percentile Threshold', labelpad=15)
ax1.set_ylabel('Number of Edges Retained', color=colors[2], labelpad=15)
ax1.tick_params(axis='y', labelcolor=colors[2])

# Add a second y-axis for percentage
ax2 = ax1.twinx()
ax2.plot(percentiles_fine, percent_retained, linewidth=3, linestyle='--', color=colors[0])
ax2.set_ylabel('Percentage of Edges Retained', color=colors[0], labelpad=15)
ax2.tick_params(axis='y', labelcolor=colors[0])

plt.title('Edge Retention vs. Weight Percentile Threshold (SCING GRN)', pad=20)
plt.grid(True, linestyle='--', alpha=0.7)

# Add vertical lines at key percentiles
for p, t in zip(percentiles, thresholds):
    plt.axvline(x=p, color='gray', linestyle=':', alpha=0.7)
    count = sum(scing_grn['importance'] > t)
    plt.text(p+0.5, edges_retained[percentiles_fine.tolist().index(p)], 
             f"{p}%: {count} edges", 
             verticalalignment='bottom', 
             fontsize=10)

plt.tight_layout()
plt.savefig('eda_plots/scing_edge_retention.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyze target connectivity (number of TFs regulating each target)
scing_target_counts = scing_grn['target'].value_counts()

plt.figure()
ax = sns.histplot(scing_target_counts, bins=50, kde=True, color=colors[3],
                 line_kws={'linewidth': 3, 'color': colors[1]})
plt.title('Distribution of Number of Regulating TFs per Target in SCING GRN', pad=20)
plt.xlabel('Number of Regulating TFs', labelpad=15)
plt.ylabel('Number of Targets', labelpad=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Add a text box with statistics
textstr = f"Mean: {scing_target_counts.mean():.2f}\nMedian: {scing_target_counts.median():.2f}\nMax: {scing_target_counts.max()}"
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
            bbox=props, fontsize=12, verticalalignment='top')
plt.tight_layout()
plt.savefig('eda_plots/scing_target_connectivity.png', dpi=300, bbox_inches='tight')
plt.show()

# Top targets by number of regulating TFs
print("\nTop 10 targets with most regulating TFs in SCING GRN:")
top_targets = scing_target_counts.head(10)
print(top_targets)

# Create a bar plot for top targets
plt.figure()
ax = sns.barplot(x=top_targets.index, y=top_targets.values, palette=sns.color_palette("magma", len(top_targets)))
plt.title('Top 10 Targets with Most Regulating TFs in SCING GRN', pad=20)
plt.xlabel('Target Gene', labelpad=15)
plt.ylabel('Number of Regulating TFs', labelpad=15)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(top_targets.values):
    plt.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/scing_top_targets.png', dpi=300, bbox_inches='tight')
plt.show()

# Network density analysis
total_possible_edges = scing_unique_tfs * scing_unique_targets
actual_edges = len(scing_grn)
network_density = actual_edges / total_possible_edges

print(f"\nNetwork Analysis:")
print(f"Total possible edges (TFs × targets): {total_possible_edges}")
print(f"Actual edges in network: {actual_edges}")
print(f"Network density: {network_density:.6f}")

# Create a pie chart showing network density
plt.figure(figsize=(10, 10))
labels = ['Present Edges', 'Missing Edges']
sizes = [actual_edges, total_possible_edges - actual_edges]
explode = (0.1, 0)  # explode the 1st slice
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.4f%%',
        shadow=True, startangle=140, colors=[colors[0], colors[2]])
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('SCING GRN Network Density', fontsize=20, pad=20, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/scing_network_density.png', dpi=300, bbox_inches='tight')
plt.show()

