from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = ["I love this!", "This is amazing!", "I hate it", "This is awful", "So good", "So bad"]
labels = [1, 1, 0, 0, 1, 0]  # 1=positive, 0=negative

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

sample = ["I really like this product!"]
print("Prediction:", "Positive" if model.predict(vectorizer.transform(sample))[0] else "Negative")