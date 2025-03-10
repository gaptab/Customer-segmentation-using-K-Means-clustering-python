# Customer-segmentation-using-K-Means-clustering-python

![alt text](https://github.com/gaptab/Customer-segmentation-using-K-Means-clustering-python/blob/main/elbow_method.png)

**Step 1: Generating Data**

We create a synthetic dataset with 500 customers.

Each customer has Age, Annual Income, Spending Score, and Account Tenure (years of relationship with the company).

This data simulates real-world customer behavior, where spending score represents how frequently and how much a customer spends.

**Step 2: Data Preprocessing**

Since machine learning models perform better with normalized data, we standardize numerical features so that all values are on a similar scale.

This ensures that one feature (e.g., income) does not dominate others (e.g., age) due to larger numerical values.

**Step 3: Determining the Optimal Number of Clusters (K-Means Clustering)**

We use the Elbow Method to find the best number of clusters (K).

This method calculates how much variance is explained by different values of K.

The best K is found where adding more clusters does not significantly reduce variance, forming an "elbow" in the graph.

**Step 4: Applying K-Means Clustering**

Once the optimal number of clusters is found (let’s say K=4), we apply K-Means clustering to divide customers into four distinct groups.

Each customer is assigned to one of these clusters based on their spending behavior, income, and tenure.

**Step 5: Evaluating the Model**

We use the Silhouette Score, a metric that measures how well-separated the clusters are.

A high silhouette score means customers within the same group are similar, while different groups are well-separated.

**Step 6: Understanding the Customer Segments**

We calculate average age, income, spending score, and tenure for each cluster to interpret customer behavior.

Example insights:

Cluster 0 → High-income customers with moderate spending.

Cluster 1 → Young, high-spending customers.

Cluster 2 → Low-income customers with low spending.

Cluster 3 → Long-term customers with stable income and spending.

**Step 7: Business Insights & Recommendations**

Personalized Offers: High-spending customers should get exclusive loyalty programs.

Targeted Promotions: Low-income customers might respond well to discounts.

Retention Strategies: Long-term customers should receive special rewards to maintain loyalty.
