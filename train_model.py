import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib

# 1. Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="student_behavior"
)

# 2. Read student data
df = pd.read_sql("SELECT * FROM students", db)
db.close()

# 3. Select ML features
features = [
    "attendance",
    "study_hours",
    "assignment_score",
    "quiz_score",
    "participation",
    "lms_activity",
    "late_submissions",
    "previous_performance",
    "screen_time"
]

X = df[features]

# 4. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Elbow Method
inertias = []

for k in range(2, 6):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

# Elbow graph
plt.plot(range(2, 6), inertias, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# 6. Silhouette Scores
print("\nSilhouette Scores:")

for k in range(2, 6):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K = {k}, Score = {score:.2f}")

# 7. Final K-Means model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# 8. Display clusters
print("\nStudent Clusters:")
print(df[["student_id", "cluster"]].to_string(index=False))

# 9. Cluster profiles
print("\nCluster Profiles:")
print(df.groupby("cluster")[features].mean().round(2))

# 10. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA graph
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["cluster"],
    s=80
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Student Behaviour Clusters using PCA")
plt.show()

# 10. DBSCAN clustering
dbscan = DBSCAN(eps=1.5, min_samples=2)
dbscan_labels = dbscan.fit_predict(X_scaled)

df["dbscan_cluster"] = dbscan_labels

print("\nDBSCAN Clusters:")
print(df[["student_id", "dbscan_cluster"]].to_string(index=False))

# Count DBSCAN clusters
print("\nNumber of DBSCAN clusters:",
      len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0))

print("Number of noise points:",
      list(dbscan_labels).count(-1))

# 11. Save models
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca_model.pkl")

print("\nModels saved successfully!")