#pip install numpy pandas scikit-learn 
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
documents = [
    "I love this product, it's amazing!",
    "This is the worst experience ever.",
    "Absolutely fantastic, I highly recommend it.",
    "I am not happy with this purchase.",
    "Great value for money, really satisfied.",
    "Terrible quality, do not buy this.",
    "Best decision I made, totally worth it.",
    "Disappointed, it did not meet my expectations."
]
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
n_components = 2  # Number of topics or concepts to extract
lsa = TruncatedSVD(n_components=n_components, random_state=42)
X_lsa = lsa.fit_transform(X)
terms = vectorizer.get_feature_names_out()
for i, topic in enumerate(lsa.components_):
    print(f"Topic {i}: ", [terms[j] for j in topic.argsort()[-5:]])
df = pd.DataFrame(X_lsa, columns=[f"Concept {i}" for i in range(n_components)])
df["Original Text"] = documents
print(df)
