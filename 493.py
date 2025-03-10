import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate 500 dummy customers
num_customers = 500

# Create synthetic customer data
customer_data = pd.DataFrame({
    'CustomerID': np.arange(1, num_customers + 1),
    'Age': np.random.randint(18, 70, num_customers),
    'AnnualIncome': np.random.randint(20000, 120000, num_customers),
    'SpendingScore': np.random.randint(1, 100, num_customers),
    'AccountTenure': np.random.randint(1, 20, num_customers)
})

# Display first few rows
print(customer_data.head())

# Select features for clustering
features = ['Age', 'AnnualIncome', 'SpendingScore', 'AccountTenure']

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_data[features])

# Convert back to DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=features)

# Try different K values and compute inertia (within-cluster variance)
inertia = []
K_range = range(2, 11)  # Checking clusters from 2 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal K')
plt.show()

# Train K-Means with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_df)

# Compute silhouette score
sil_score = silhouette_score(scaled_df, customer_data['Cluster'])
print(f'Silhouette Score: {sil_score:.3f}')

# Group by cluster and compute mean values
cluster_summary = customer_data.groupby('Cluster')[['Age', 'AnnualIncome', 'SpendingScore', 'AccountTenure']].mean()
print(cluster_summary)

# Save the original customer data
customer_data.to_csv("customer_data.csv", index=False)

# Save the scaled data used for clustering
scaled_df.to_csv("scaled_customer_data.csv", index=False)

# Save the cluster summary (average values per cluster)
cluster_summary.to_csv("customer_cluster_summary.csv", index=True)

print("All DataFrames have been saved as CSV files!")

