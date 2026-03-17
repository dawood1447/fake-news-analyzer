import pickle
import numpy as np
from textblob import TextBlob

def load_model_and_predict(text):
    try:
        # Load the saved brain (vectorizer and model)
        with open('vectorizer.pkl', 'rb') as vec_file:
            vectorizer = pickle.load(vec_file)
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
            
        vectorized_text = vectorizer.transform([text])
        prediction = model.predict(vectorized_text)[0]
        
        # 1. Exact Probabilities via Sigmoid Curve
        decision = model.decision_function(vectorized_text)[0]
        prob = 1 / (1 + np.exp(-decision))
        
        # Map probabilities correctly based on how the model ordered the classes
        if model.classes_[1] == 'REAL':
            real_prob = round(prob * 100, 1)
            fake_prob = round((1 - prob) * 100, 1)
        else:
            fake_prob = round(prob * 100, 1)
            real_prob = round((1 - prob) * 100, 1)
        
        # 2. Smarter Explainability (Real vs Fake Indicators)
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]
        text_vector = vectorized_text.toarray()[0]
        
        # Find all words in the text that the model actually knows
        present_words_indices = text_vector.nonzero()[0]
        word_impact = {feature_names[idx]: coefs[idx] * text_vector[idx] for idx in present_words_indices}
        
        # Separate words based on which direction they push the model's math
        if model.classes_[1] == 'REAL':
            real_words = {w: imp for w, imp in word_impact.items() if imp > 0}
            fake_words = {w: imp for w, imp in word_impact.items() if imp < 0}
        else:
            fake_words = {w: imp for w, imp in word_impact.items() if imp > 0}
            real_words = {w: imp for w, imp in word_impact.items() if imp < 0}
            
        # Grab the top 10 strongest words for both sides to feed the UI highlighter
        top_real = [w for w, imp in sorted(real_words.items(), key=lambda x: abs(x[1]), reverse=True)[:10]]
        top_fake = [w for w, imp in sorted(fake_words.items(), key=lambda x: abs(x[1]), reverse=True)[:10]]
        
        # 3. Sentiment Analysis & Word Count
        polarity = TextBlob(text).sentiment.polarity
        abs_polarity = abs(polarity)
        
        if abs_polarity > 0.3:
            sentiment_desc = "High Emotional Tone 🔴 (Often indicates bias or sensationalism)"
        elif abs_polarity > 0.1:
            sentiment_desc = "Moderate Tone 🟡 (Slightly opinionated)"
        else:
            sentiment_desc = "Neutral/Factual Tone 🟢 (Characteristic of standard reporting)"

        return {
            "prediction": prediction,
            "real_prob": real_prob,
            "fake_prob": fake_prob,
            "top_real_words": top_real,
            "top_fake_words": top_fake,
            "sentiment": sentiment_desc,
            "length": len(text.split())
        }
    except Exception as e:
        return None