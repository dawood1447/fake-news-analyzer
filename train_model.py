import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Loading data...")
df = pd.read_csv('dataset/news.csv')

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 🚀 TWEAK: Aggressive n-grams (1,2) and max_df to filter out noise
print("Vectorizing text with N-Grams...")
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# 🚀 TWEAK: Max iterations pushed to 1000 for stronger boundary confidence
print("Training PassiveAggressiveClassifier...")
pac_model = PassiveAggressiveClassifier(max_iter=1000, C=0.5, random_state=42)
pac_model.fit(tfidf_train, y_train)

y_pred = pac_model.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Model Training Complete! Accuracy: {round(score*100, 2)}%')

with open('model.pkl', 'wb') as model_file:
    pickle.dump(pac_model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(tfidf_vectorizer, vec_file)
    
print("Ultimate model and vectorizer saved!")