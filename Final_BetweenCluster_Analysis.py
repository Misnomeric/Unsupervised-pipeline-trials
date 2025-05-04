# Insert any required libraries here
import pandas as pd
import hdbscan
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import fisher_exact


#===STEP 1: LOAD DATASET AND DROP RESISTANCE PHENOTYPE COLUMN===
df = pd.read_csv("C:/Users/anvay/OneDrive/Desktop/Study Stuff MBB/ARP/Datasets/final_beta_lactamase_dataset.csv")
X = df.drop(columns=['Resistance Phenotype'])

# Fit PCA without limiting n_components
pca = PCA()
X_pca = pca.fit_transform(X)

# Plot cumulative variance
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot: Optimal Number of PCA Components')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#===STEP 2: PCA REDUCTION===
pca = PCA(n_components=0.95, random_state=1580) 
X_pca = pca.fit_transform(X)


#===STEP 3: HDBSCAN CLUSTERING USING EUCLIDEAN DISTANCE===
clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=10)
cluster_labels = clusterer.fit_predict(X_pca)
df['Cluster'] = cluster_labels


#===STEP 4: t-SNE VISUALISATION OF PCA-REDUCED HDBSCAN CLUSTERING===
tsne = TSNE(random_state=1580)
tsne_result = tsne.fit_transform(X_pca)

tsne_df = pd.DataFrame(tsne_result, columns=['tSNE-1', 'tSNE-2'])
tsne_df['Cluster'] = cluster_labels.astype(str)  # Convert to string for discrete coloring

# Generate unique colors for each cluster
num_clusters = tsne_df['Cluster'].nunique()
palette = sns.color_palette("hls", num_clusters)

# Plot the t-SNE scatterplot
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=tsne_df[tsne_df['Cluster'] != '-1'],
    x='tSNE-1', y='tSNE-2',
    hue='Cluster',
    palette=palette,
    s=60,
    edgecolor='black',
    legend='full'
)

# Plot noise points in gray
plt.scatter(
    tsne_df[tsne_df['Cluster'] == '-1']['tSNE-1'],
    tsne_df[tsne_df['Cluster'] == '-1']['tSNE-2'],
    color='lightgray',
    s=50,
    label='Noise'
)

# === Add text labels to each cluster at its centroid ===
for cluster_label, group in tsne_df[tsne_df['Cluster'] != '-1'].groupby('Cluster'):
    x_mean = group['tSNE-1'].mean()
    y_mean = group['tSNE-2'].mean()
    plt.text(
        x_mean, y_mean,
        cluster_label,
        fontsize=8,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.6)
    )

# Final touches
plt.title("t-SNE of HDBSCAN Clustering (Labeled Clusters)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.tight_layout()
plt.show()

#===STEP 5: OVERLAYING RESISTANCE PHENOTYPE AGAINST CLUSTERS
# Cross-tabulation of Resistance Phenotype by Cluster (normalized)
resistance_vs_cluster = pd.crosstab(df['Cluster'], df['Resistance Phenotype'], normalize='index')

# Print the table rounded for clarity
print("Resistance Phenotype Distribution per Cluster:")
print(resistance_vs_cluster.round(3))

# Step 1: Compute resistance distribution per cluster
res_summary = pd.crosstab(df['Cluster'], df['Resistance Phenotype'], normalize='index')

# Step 2: Identify pure clusters
fully_susceptible_clusters = res_summary[res_summary[0] == 1.0].index.tolist()
fully_resistant_clusters = res_summary[res_summary[1] == 1.0].index.tolist()

# Step 3: Print results
print("Fully Susceptible Clusters:", fully_susceptible_clusters)
print("Fully Resistant Clusters:", fully_resistant_clusters)

# === STEP 6: Table of Most Variably Present Genes in Clusters 6 vs 24 ===

# Filter data for clusters 6 and 24
df_6_24 = df[df['Cluster'].isin([6, 24])].copy()

# Separate the two clusters
cluster_6 = df_6_24[df_6_24['Cluster'] == 6]
cluster_24 = df_6_24[df_6_24['Cluster'] == 24]

# Identify gene columns
non_gene_cols = ['Resistance Phenotype', 'Cluster']
gene_cols = [col for col in df.columns if col not in non_gene_cols]

# Calculate presence proportions and their absolute difference
variability_data = []
for gene in gene_cols:
    prop_6 = cluster_6[gene].mean()
    prop_24 = cluster_24[gene].mean()
    diff = abs(prop_6 - prop_24)
    variability_data.append({
        'Gene': gene,
        'Cluster 6 Proportion': round(prop_6, 2),
        'Cluster 24 Proportion': round(prop_24, 2),
        'Absolute Difference': round(diff, 2)
    })

# Create a DataFrame and sort
top_variable_genes_df = pd.DataFrame(variability_data).sort_values(by='Absolute Difference', ascending=False).reset_index(drop=True)

# Select top 15
top_variable_genes_df = top_variable_genes_df.head(15)
top_variable_genes_df.to_csv("C:/Users/anvay/OneDrive/Desktop/Study Stuff MBB/ARP/Datasets/CLUSTERS6AND24.csv")


#===STEP 7: BARPLOT OF RESISTANCE ENRICHMENT IN CLUSTERS
# Cross-tabulate resistance distribution per cluster
res_summary = pd.crosstab(df['Cluster'], df['Resistance Phenotype'], normalize='index')

# Sort clusters by proportion of resistant samples (column 1)
res_summary_sorted = res_summary.sort_values(by=1, ascending=False)

# Flip the column order so 'Resistant' (1) is plotted first
res_summary_plot = res_summary_sorted[[1, 0]]

# Plot with custom y-axis ticks and horizontal grid lines
ax = res_summary_plot.plot(
    kind='bar',
    stacked=True,
    color={1: 'salmon', 0: 'skyblue'},
    figsize=(12, 6)
)

# Customize y-axis ticks
yticks = np.arange(0, 1.1, 0.1)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{y:.1f}" for y in yticks])

# Add horizontal dotted lines
for y in yticks:
    ax.axhline(y=y, color='gray', linestyle=':', linewidth=0.8)

# Titles and labels
plt.title("Resistance Composition per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Proportion")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#step 9: DISPLAYING ALL MIXED CLUSTERS
# === Resistance composition per cluster ===
res_summary = pd.crosstab(df['Cluster'], df['Resistance Phenotype'], normalize='index')

# === Identify and sort mixed clusters ===
mixed_clusters = res_summary[(res_summary[0] > 0) & (res_summary[1] > 0)].copy()
mixed_clusters['Resistant Proportion'] = mixed_clusters[1]
mixed_clusters_sorted = mixed_clusters.sort_values(by='Resistant Proportion', ascending=False)

# === Display ===
print("Mixed Clusters (sorted by resistance proportion):")
print(mixed_clusters_sorted)


# === STEP 8: HEATMAP OF FULLY RESISTANT AND FULLY SUSCEPTIBLE CLUSTERS

#Subset data from those clusters only
df_combined = df[df['Cluster'].isin(fully_resistant_clusters + fully_susceptible_clusters)].copy()
df_res_combined = df_combined[df_combined['Resistance Phenotype'] == 1]
df_sus_combined = df_combined[df_combined['Resistance Phenotype'] == 0]

#Identify gene columns 
non_gene_cols = ['Resistance Phenotype', 'Cluster']
gene_cols = [col for col in df.columns if col not in non_gene_cols]

#Filter out genes appearing less than 6 times 
df_group = pd.concat([df_res_combined, df_sus_combined])
gene_cols = [gene for gene in gene_cols if df_group[gene].sum() >= 6]

#Compute enrichment within combined clusters only
res_enrichment = {}
sus_enrichment = {}
retained_genes = []

for gene in gene_cols:
    res_count = df_res_combined[gene].sum()
    sus_count = df_sus_combined[gene].sum()
    res_prop = res_count / len(df_res_combined) if len(df_res_combined) > 0 else 0
    sus_prop = sus_count / len(df_sus_combined) if len(df_sus_combined) > 0 else 0

    if res_prop > sus_prop or (sus_count == 0 and res_count > 0):
        retained_genes.append(gene)
        res_enrichment[gene] = round(res_prop, 2)
        sus_enrichment[gene] = round(sus_prop, 2)

#Plotting heatmap
heatmap_df = pd.DataFrame({
    'Resistant': res_enrichment,
    'Susceptible': sus_enrichment
}).T[retained_genes]

plt.figure(figsize=(len(retained_genes) * 0.4 + 2, 4))
sns.heatmap(heatmap_df, cmap='YlOrRd', annot=True, fmt=".2f", cbar=True)
plt.title("Gene Enrichment in Resistant vs Susceptible Samples\n(From Fully Resistant & Fully Susceptible Clusters Only, Min 6 Appearances)")
plt.xlabel("Gene")
plt.ylabel("Phenotype")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# === STEP 9: Compare 90–100% Resistant Clusters vs Fully Susceptible Clusters ===

# 1. Identify clusters
ninety_clusters = res_summary[(res_summary[1] > 0.9) & (res_summary[1] < 1.0)].index.tolist()
fully_susceptible_clusters = res_summary[res_summary[0] == 1.0].index.tolist()

# 2. Subset samples
df_nineties = df[df['Cluster'].isin(ninety_clusters)].copy()
df_sus = df[df['Cluster'].isin(fully_susceptible_clusters)].copy()

# 3. Gene columns
non_gene_cols = ['Resistance Phenotype', 'Cluster']
gene_cols = [col for col in df.columns if col not in non_gene_cols]

# 4. Remove genes that appear less than 6 times
df_group = pd.concat([df_nineties, df_sus])
gene_cols = [gene for gene in gene_cols if df_group[gene].sum() >= 6]

# 5. Enrichment calculation
nineties_enrichment = {}
sus_enrichment = {}
retained_genes = []

for gene in gene_cols:
    prop_nineties = df_nineties[gene].sum() / len(df_nineties) if len(df_nineties) > 0 else 0
    prop_sus = df_sus[gene].sum() / len(df_sus) if len(df_sus) > 0 else 0

    if prop_nineties > prop_sus or (prop_nineties > 0 and prop_sus == 0):
        retained_genes.append(gene)
        nineties_enrichment[gene] = round(prop_nineties, 2)
        sus_enrichment[gene] = round(prop_sus, 2)

# 6. Build heatmap
heatmap_df = pd.DataFrame({
    'Resistant': nineties_enrichment,
    'Susceptible': sus_enrichment
}).T[retained_genes]

# 6. Plot
plt.figure(figsize=(len(retained_genes) * 0.4 + 2, 4))
sns.heatmap(heatmap_df, cmap='YlOrRd', annot=True, fmt=".2f", cbar=True)
plt.title("Gene Enrichment in 90–96% Resistant vs Fully Susceptible Clusters\n(Min 6 Appearances)")
plt.xlabel("Gene")
plt.ylabel("Cluster Group")
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.25)
plt.show()


# === STEP 10: Compare Clusters 1&2, Cluster 23, and Other 90–100% Resistant Clusters ===

# Define cluster groups explicitly
clusters_1_2 = [1, 2]
cluster_23 = [23]
all_ninety_clusters = res_summary[res_summary[1] > 0.9].index.tolist()
other_ninety_clusters = [c for c in all_ninety_clusters if c not in clusters_1_2 + cluster_23]

# Subset resistant samples for each group
df_1_2_res = df[(df['Cluster'].isin(clusters_1_2)) & (df['Resistance Phenotype'] == 1)].copy()
df_23_res = df[(df['Cluster'].isin(cluster_23)) & (df['Resistance Phenotype'] == 1)].copy()
df_other_res = df[(df['Cluster'].isin(other_ninety_clusters)) & (df['Resistance Phenotype'] == 1)].copy()

# Identify gene columns explicitly
non_gene_cols_final = ['Resistance Phenotype', 'Cluster']
gene_cols_final = [col for col in df.columns if col not in non_gene_cols_final]

# Filter genes appearing at least 6 times across all three groups combined
df_combined_all = pd.concat([df_1_2_res, df_23_res, df_other_res])
gene_cols_filtered_final = [gene for gene in gene_cols_final if df_combined_all[gene].sum() >= 6]

# Calculate proportions for each gene in each cluster group
final_comparison_results = []
for gene in gene_cols_filtered_final:
    prop_1_2 = round(df_1_2_res[gene].mean(), 2)
    prop_23 = round(df_23_res[gene].mean(), 2)
    prop_other = round(df_other_res[gene].mean(), 2)
    final_comparison_results.append({
        'Gene': gene,
        'Proportion in Clusters 1 & 2': prop_1_2,
        'Proportion in Cluster 23': prop_23,
        'Proportion in Other 90–100% Clusters': prop_other,
    })

# Create the final dataframe sorted by the largest variation across the groups
final_clusters_comparison_df = pd.DataFrame(final_comparison_results)

# Calculate maximum absolute difference for sorting clarity
final_clusters_comparison_df['Max Absolute Difference'] = final_clusters_comparison_df[[
    'Proportion in Clusters 1 & 2',
    'Proportion in Cluster 23',
    'Proportion in Other 90–100% Clusters'
]].apply(lambda row: max(row) - min(row), axis=1)

# Sort by max absolute difference
final_clusters_comparison_df.sort_values(by='Max Absolute Difference', ascending=False, inplace=True)

# Drop the helper sorting column for clarity
final_clusters_comparison_df.drop(columns=['Max Absolute Difference'], inplace=True)

# Save the dataframe to CSV
final_clusters_comparison_df.to_csv("C:/Users/anvay/OneDrive/Desktop/Study Stuff MBB/ARP/Datasets/clusters1_2_23_vs_rest.csv", index=False)



# === STEP 11: Heatmap of 60–90% Resistant Clusters vs Fully Susceptible Clusters ===

# 1. Identify target cluster groups
sixty_to_ninety_clusters = res_summary[(res_summary[1] >= 0.6) & (res_summary[1] <= 0.9)].index.tolist()
fully_susceptible_clusters = res_summary[res_summary[0] == 1.0].index.tolist()

# 2. Subset data
df_sixty90 = df[df['Cluster'].isin(sixty_to_ninety_clusters)].copy()
df_sus = df[df['Cluster'].isin(fully_susceptible_clusters)].copy()

# 3. Identify gene columns
non_gene_cols = ['Resistance Phenotype', 'Cluster']
gene_cols = [col for col in df.columns if col not in non_gene_cols]

# 4. Removing genes that appear less than 6 times
df_group = pd.concat([df_sixty90, df_sus])
gene_cols = [gene for gene in gene_cols if df_group[gene].sum() >= 6]

# 5. Compare enrichment
sixty90_enrichment = {}
sus_enrichment = {}
retained_genes = []

for gene in gene_cols:
    prop_sixty90 = df_sixty90[gene].sum() / len(df_sixty90) if len(df_sixty90) > 0 else 0
    prop_sus = df_sus[gene].sum() / len(df_sus) if len(df_sus) > 0 else 0

    if prop_sixty90 > prop_sus or (prop_sixty90 > 0 and prop_sus == 0):
        retained_genes.append(gene)
        sixty90_enrichment[gene] = round(prop_sixty90, 2)
        sus_enrichment[gene] = round(prop_sus, 2)

# 6. Build heatmap
import matplotlib.pyplot as plt
import seaborn as sns

heatmap_df = pd.DataFrame({
    'Resistant': sixty90_enrichment,
    'Susceptible': sus_enrichment
}).T[retained_genes]

# 7. Plot heatmap
plt.figure(figsize=(len(retained_genes) * 0.4 + 2, 4))
sns.heatmap(heatmap_df, cmap='YlOrRd', annot=True, fmt=".2f", cbar=True)
plt.title("Gene Enrichment in 60–90% Resistant vs Fully Susceptible Clusters\n(Min 6 Appearances)")
plt.xlabel("Gene")
plt.ylabel("Cluster Group")
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.25)
plt.show()


# === STEP 12: Compare Cluster 0 vs Other 60–90% Resistant Clusters (Resistant Samples Only) ===

# 1. Identify target clusters
cluster0_res_only = [0]
sixty90_res_clusters = res_summary[(res_summary[1] >= 0.6) & (res_summary[1] <= 0.9)].index.tolist()
sixty90_others_res = [c for c in sixty90_res_clusters if c != 0]

# 2. Subset only RESISTANT samples from these clusters
df_cluster0_res = df[(df['Cluster'].isin(cluster0_res_only)) & (df['Resistance Phenotype'] == 1)].copy()
df_sixty90_others_res = df[(df['Cluster'].isin(sixty90_others_res)) & (df['Resistance Phenotype'] == 1)].copy()

# 3. Identify gene columns
non_gene_cols_c0 = ['Resistance Phenotype', 'Cluster']
gene_cols_c0 = [col for col in df.columns if col not in non_gene_cols_c0]

# 4. Filter genes appearing in at least 6 samples across both groups
df_c0_combined_res = pd.concat([df_cluster0_res, df_sixty90_others_res])
gene_cols_c0 = [g for g in gene_cols_c0 if df_c0_combined_res[g].sum() >= 6]

# 5. Calculate proportions and absolute differences
diff_table_c0 = []
for gene in gene_cols_c0:
    prop_c0 = df_cluster0_res[gene].mean()
    prop_other = df_sixty90_others_res[gene].mean()
    diff_table_c0.append({
        'Gene': gene,
        'Proportion in Cluster 0 (Resistant)': round(prop_c0, 2),
        'Proportion in Other 60–90% Clusters (Resistant)': round(prop_other, 2),
        'Absolute Difference': round(abs(prop_c0 - prop_other), 2)
    })

# 6. Build sorted dataframe
cluster0_vs_sixty90_res_df = pd.DataFrame(diff_table_c0).sort_values(by='Absolute Difference', ascending=False)
cluster0_vs_sixty90_res_df.to_csv("C:/Users/anvay/OneDrive/Desktop/Study Stuff MBB/ARP/Datasets/0vsother6090.csv")

# === STEP 13: Compare <60% Resistant Clusters vs Fully Susceptible Clusters ===
# 0. Create a noise-free copy of the data for this heatmap only
df_no_noise = df[df['Cluster'] != -1].copy()

# 1. Use already computed res_summary to get clusters
subsixty_clusters = res_summary[(res_summary[1] < 0.6)].index.tolist()
fully_susceptible_clusters = res_summary[res_summary[0] == 1.0].index.tolist()

# 2. Subset from df_no_noise instead of df
df_subsixty_combined = df_no_noise[df_no_noise['Cluster'].isin(subsixty_clusters + fully_susceptible_clusters)].copy()

# 3. Split by phenotype
df_res_subsixty = df_subsixty_combined[df_subsixty_combined['Resistance Phenotype'] == 1]
df_sus_subsixty = df_subsixty_combined[df_subsixty_combined['Resistance Phenotype'] == 0]

# 4. Identify gene columns
non_gene_cols = ['Resistance Phenotype', 'Cluster']
gene_cols = [col for col in df.columns if col not in non_gene_cols]

# 6. Compute enrichment
res_enrichment = {}
sus_enrichment = {}
retained_genes = []

for gene in gene_cols:
    res_prop = df_res_subsixty[gene].sum() / len(df_res_subsixty) if len(df_res_subsixty) > 0 else 0
    sus_prop = df_sus_subsixty[gene].sum() / len(df_sus_subsixty) if len(df_sus_subsixty) > 0 else 0

    if res_prop > sus_prop or (res_prop > 0 and sus_prop == 0):
        retained_genes.append(gene)
        res_enrichment[gene] = round(res_prop, 2)
        sus_enrichment[gene] = round(sus_prop, 2)

# 7. Build heatmap DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

heatmap_df = pd.DataFrame({
    'Resistant': res_enrichment,
    'Susceptible': sus_enrichment
}).T[retained_genes]

# 8. Plot heatmap
plt.figure(figsize=(len(retained_genes) * 0.4 + 2, 4))
sns.heatmap(heatmap_df, cmap='YlOrRd', annot=True, fmt=".2f", cbar=True)
plt.title("Gene Enrichment in <60% Resistant vs Fully Susceptible Clusters\n(Min 6 Appearances, Noise Excluded)")
plt.xlabel("Gene")
plt.ylabel("Cluster Group")
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.25)
plt.show()


# FISHER'S EXACT TEST ON MOST PREVALENT GENES IN CLUSTER GROUPS
# Combining results of previous cluster enrichments
first_gene_set=['KPC-1','KPC-3','OXA-9','SHV-106']
second_gene_set=['SHV-134','SHV-182','TEM-122']
third_gene_set=['SHV-11']
fourth_gene_set=['OXA-232','SHV-107','SHV-178','SHV-187','SHV-33','SHV-36','SHV-76']

# Combine all sets and remove duplicates
most_prevalent_genes = list(set(first_gene_set + second_gene_set + third_gene_set + fourth_gene_set))
# Run Fisher's exact test
fisher_results = []

for gene in most_prevalent_genes:
    if gene in df.columns:
        gene_present = df[df[gene] == 1]
        gene_absent = df[df[gene] == 0]

        a = gene_present['Resistance Phenotype'].sum()
        b = len(gene_present) - a
        c = gene_absent['Resistance Phenotype'].sum()
        d = len(gene_absent) - c

        if (a + b > 0) and (c + d > 0):
            table = [[a, b], [c, d]]
            oddsratio, pvalue = fisher_exact(table)
            log2_or = np.log2(oddsratio) if oddsratio > 0 else np.nan
            neg_log10_p = -np.log10(pvalue) if pvalue > 0 else np.nan
            fisher_results.append([gene, oddsratio, log2_or, pvalue, neg_log10_p])

# Convert to DataFrame
fisher_df = pd.DataFrame(fisher_results, columns=[
    'Gene', 'Odds Ratio', 'log2(Odds Ratio)', 'P-Value', '-log10(P-Value)'
])

# Sort and display
fisher_df = fisher_df.sort_values(by='P-Value').reset_index(drop=True)
fisher_df.to_csv("C:/Users/anvay/OneDrive/Desktop/Study Stuff MBB/ARP/Datasets/fisher_test_prevalent.csv")


# FINDING POSSIBLE RARE GENES in noise "cluster"

# 1. Subset noise and fully susceptible cluster samples
df_noise_res_only = df[(df['Cluster'] == -1) & (df['Resistance Phenotype'] == 1)].copy()
df_fully_sus = df[df['Cluster'].isin(fully_susceptible_clusters) & (df['Resistance Phenotype'] == 0)].copy()

# 2. Identify gene columns
noise_vs_sus_non_genes = ['Resistance Phenotype', 'Cluster']
noise_vs_sus_gene_cols = [col for col in df.columns if col not in noise_vs_sus_non_genes]

# 3. Filter genes that appear < 5 times across both groups
df_noise_vs_sus_combined = pd.concat([df_noise_res_only, df_fully_sus])
noise_vs_sus_gene_cols = [gene for gene in noise_vs_sus_gene_cols if df_noise_vs_sus_combined[gene].sum() >= 5]

# 4. Compute enrichment
noise_enrichment_dict = {}
sus_enrichment_dict = {}
retained_genes_noise_vs_sus = []

for gene in noise_vs_sus_gene_cols:
    prop_noise = df_noise_res_only[gene].sum() / len(df_noise_res_only) if len(df_noise_res_only) > 0 else 0
    prop_sus = df_fully_sus[gene].sum() / len(df_fully_sus) if len(df_fully_sus) > 0 else 0

    if prop_noise > prop_sus or (prop_noise > 0 and prop_sus == 0):
        retained_genes_noise_vs_sus.append(gene)
        noise_enrichment_dict[gene] = round(prop_noise, 2)
        sus_enrichment_dict[gene] = round(prop_sus, 2)

# 5. Build and plot heatmap
if retained_genes_noise_vs_sus:
    noise_vs_sus_heatmap_df = pd.DataFrame({
        'Noise (Resistant)': noise_enrichment_dict,
        'Susceptible': sus_enrichment_dict
    }).T[retained_genes_noise_vs_sus]

    plt.figure(figsize=(len(retained_genes_noise_vs_sus) * 0.4 + 2, 4))
    sns.heatmap(noise_vs_sus_heatmap_df, cmap='YlOrRd', annot=True, fmt=".2f", cbar=True)
    plt.title("Gene Enrichment in Noise (Resistant) vs Fully Susceptible Clusters\n(Min 5 Appearances)")
    plt.xlabel("Gene")
    plt.ylabel("Phenotype")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
else:
    print("No enriched genes found in noise vs fully susceptible comparison (min 5 appearances).")
    
#FISHER'S EXACT TEST FOR RARE GENES IN NOISE
rare_genes=['OXA-232','CARB-3','CMY-59','FOX-5','NDM-1','OXA-10','TEM-150','VEB-1','VEB-5','OXA-48','OXA-9']

# Run Fisher's exact test for rare genes
rare_fisher_results = []

for gene in rare_genes:
    if gene in df.columns:
        gene_present = df[df[gene] == 1]
        gene_absent = df[df[gene] == 0]

        a = gene_present['Resistance Phenotype'].sum()
        b = len(gene_present) - a
        c = gene_absent['Resistance Phenotype'].sum()
        d = len(gene_absent) - c

        if (a + b > 0) and (c + d > 0):
            table = [[a, b], [c, d]]
            oddsratio, pvalue = fisher_exact(table)
            log2_or = np.log2(oddsratio) if oddsratio > 0 else np.nan
            neg_log10_p = -np.log10(pvalue) if pvalue > 0 else np.nan
            rare_fisher_results.append([gene, oddsratio, log2_or, pvalue, neg_log10_p])

# Convert to DataFrame and sort by descending odds ratio
rare_fisher_df_sorted = pd.DataFrame(rare_fisher_results, columns=[
    'Gene', 'Odds Ratio', 'log2(Odds Ratio)', 'P-Value', '-log10(P-Value)'
]).sort_values(by='Odds Ratio', ascending=False).reset_index(drop=True)

rare_fisher_df_sorted.to_csv("C:/Users/anvay/OneDrive/Desktop/Study Stuff MBB/ARP/Datasets/RARE_FISHER_RESULTS.csv")


# === STEP 8: HEATMAP OF FULLY RESISTANT AND FULLY SUSCEPTIBLE CLUSTERS (using absolute difference) ===

# Subset data from those clusters only
df_combined = df[df['Cluster'].isin(fully_resistant_clusters + fully_susceptible_clusters)].copy()
df_res_combined = df_combined[df_combined['Resistance Phenotype'] == 1]
df_sus_combined = df_combined[df_combined['Resistance Phenotype'] == 0]

# Identify gene columns
non_gene_cols = ['Resistance Phenotype', 'Cluster']
gene_cols = [col for col in df.columns if col not in non_gene_cols]

# Filter out genes appearing less than 6 times
df_group = pd.concat([df_res_combined, df_sus_combined])
gene_cols = [gene for gene in gene_cols if df_group[gene].sum() >= 6]

# Compute absolute difference in proportions to find variably present genes
res_enrichment = {}
sus_enrichment = {}
retained_genes = []

for gene in gene_cols:
    res_prop = df_res_combined[gene].mean()
    sus_prop = df_sus_combined[gene].mean()
    abs_diff = abs(res_prop - sus_prop)

    if abs_diff > 0:  # Only retain genes with actual variability
        retained_genes.append(gene)
        res_enrichment[gene] = round(res_prop, 2)
        sus_enrichment[gene] = round(sus_prop, 2)

# Create heatmap DataFrame
heatmap_df = pd.DataFrame({
    'Resistant': res_enrichment,
    'Susceptible': sus_enrichment
}).T[retained_genes]

# Plot heatmap
plt.figure(figsize=(len(retained_genes) * 0.4 + 2, 4))
sns.heatmap(heatmap_df, cmap='YlOrRd', annot=True, fmt=".2f", cbar=True)
plt.title("Gene Enrichment in Resistant vs Susceptible Samples\n(From Fully Resistant & Fully Susceptible Clusters Only, Min 6 Appearances, By Absolute Difference)")
plt.xlabel("Gene")
plt.ylabel("Phenotype")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()