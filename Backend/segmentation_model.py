import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import os

# Load the dataset
df = pd.read_csv('customer_segmentation_data.csv')

# Select the features we want to use for clustering
features = ['age', 'gender', 'income', 'spending_score', 'membership_years', 
            'purchase_frequency', 'preferred_category', 'last_purchase_amount']
X = df[features].copy() # Create a copy to avoid modifying the original df

# Preprocessing
# 1. Encode 'gender'
le = LabelEncoder()
X['gender_encoded'] = le.fit_transform(X['gender'])

# 2. One-Hot Encode 'preferred_category'
X = pd.get_dummies(X, columns=['preferred_category'], prefix='', prefix_sep='')

# 3. Drop the original 'gender' and 'preferred_category' columns
X.drop('gender', axis=1, inplace=True)

# 4. Scale all the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use the Elbow Method to find the best k
inertia = []
k_range = range(2, 11) # Testing k from 2 to 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# # Plot the Elbow Curve
# plt.figure(figsize=(10, 6))
# plt.plot(k_range, inertia, marker='o', linestyle='--')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
# plt.title('Elbow Method for Optimal k')
# plt.xticks(k_range)
# plt.grid(True)
# plt.show()

# Let's choose k=4 based on the elbow method
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Add the cluster labels back to the original DataFrame
df['segment'] = kmeans.labels_

# Group the data by segment and see the average values for each feature
segment_analysis = df.groupby('segment').mean(numeric_only=True)
# print(segment_analysis)

# # Also, see the distribution of categories in each segment
# print("\nPreferred Category by Segment:")
# print(pd.crosstab(df['segment'], df['preferred_category']))


# Segment name mapping (customize as needed)
# Segment name mapping with descriptions
segment_names = {
    0: "High income, high spending score, low frequency -> High-Value Occasional Shoppers",
    1: "Low income, low spending score -> Budget-Conscious Customers",
    2: "Medium income, very high frequency -> Loyal Regulars",
    3: "High income, high spending score, high frequency -> Champions"
}

def predict_segment(input_dict):
    # Create a DataFrame from input
    input_df = pd.DataFrame([input_dict])

    # Encode gender
    input_df['gender_encoded'] = le.transform(input_df['gender'])

    # One-hot encode preferred_category
    input_df = pd.get_dummies(input_df, columns=['preferred_category'], prefix='', prefix_sep='')

    # Drop original gender
    input_df.drop('gender', axis=1, inplace=True)

    # Add any missing columns (from X) as zeros
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure columns are in the same order as X
    input_df = input_df[X.columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict segment
    segment_label = kmeans.predict(input_scaled)[0]
    segment_name = segment_names.get(segment_label, f"Segment {segment_label}")

    return segment_label, segment_name

# Example usage:
# input_features = {
#     'age': 35,
#     'gender': 'Male',
#     'income': 90000,
#     'spending_score': 85,
#     'membership_years': 2,
#     'purchase_frequency': 3,
#     'preferred_category': 'Electronics',
#     'last_purchase_amount': 1200
# }
# label, name = predict_segment(input_features)
# print(f"Predicted Segment: {label}, {name}".encode('utf-8', errors='replace').decode())


# Bundle everything needed for prediction
segmentation_artifacts = {
    "kmeans": kmeans,
    "scaler": scaler,
    "le": le,
    "columns": list(X.columns),
    "segment_names": segment_names
}

# Ensure the model directory exists (this works even if it already exists)
os.makedirs("Backend/model", exist_ok=True)

# print(os.path.abspath("model/model.pkl"))
# Save to model.pkl
with open("Backend/model/model.pkl", "wb") as f:
    pickle.dump(segmentation_artifacts, f)

print(f"Model saved to {os.path.abspath('Backend/model/model.pkl')}")