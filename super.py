#pip install numpy pandas scikit-learn 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample text data and labels
texts = [
    "I love this product",        # positive
    "This is an amazing book",   # positive
    "I hate this movie",         # negative
    "This is the worst",         # negative
    "I like this",               # positive
    "Terrible experience",       # negative
]
labels = ['positive', 'positive', 'negative', 'negative', 'positive', 'negative']

# Step 1: Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Step 3: Train a classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
# Print predictions alongside true labels
for text, true_label, pred_label in zip(vectorizer.inverse_transform(X_test), y_test, y_pred):
    print(f"Text: {' '.join(text)} | True: {true_label} | Predicted: {pred_label}")
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
