#pip install numpy pandas scikit-learn
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
documents = [
    "Machine learning is amazing",
    "Deep learning is a subset of ML",
    "LDA is useful for topic modeling"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
n_topics = 2  # Number of topics
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx}:", [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]])
