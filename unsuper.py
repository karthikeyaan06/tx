#pip install numpy pandas scikit-learn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Sample text data (no labels needed for unsupervised learning)
texts = [
    "I love this phone",
    "This camera takes great photos",
    "The battery life is terrible",
    "Amazing screen quality",
    "Very bad performance",
    "Battery dies quickly",
    "Excellent camera resolution",
    "Horrible user experience",
]

# Step 1: Convert text to TF-IDF feature vectors
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)

# Step 2: Apply KMeans clustering
k = 2  # Number of clusters
model = KMeans(n_clusters=k, random_state=42)
model.fit(X)

# Step 3: Output cluster assignments
for i, text in enumerate(texts):
    print(f"Text: {text} -> Cluster: {model.labels_[i]}")

terms = vectorizer.get_feature_names_out()
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

print("\nTop terms per cluster:")
for i in range(model.n_clusters):
    print(f"Cluster {i}: ", end='')
    print(", ".join(terms[ind] for ind in order_centroids[i, :5]))

