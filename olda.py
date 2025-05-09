#pip install numpy pandas scikit-learn
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
documents = [
    "Machine learning is evolving",
    "AI research is expanding",
    "Online LDA adapts to new topics"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
n_topics = 2  # Number of topics
olda = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=42)
olda.fit(X)
for idx, topic in enumerate(olda.components_):
    print(f"Topic {idx}:", [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]])
