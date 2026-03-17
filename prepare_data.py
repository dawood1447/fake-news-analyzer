import pandas as pd
from sklearn.utils import shuffle
import re

print("Loading Kaggle datasets...")
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

print("Cleaning data to remove 'Reuters' bias...")
# This regex removes the "CITY (Reuters) - " opening from the real news text
# so the model can't use it as a cheat code.
true_df['text'] = true_df['text'].str.replace(r'^.*?\(Reuters\)\s*-\s*', '', regex=True)

# Add labels
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

print("Merging datasets...")
combined_df = pd.concat([true_df, fake_df])
final_df = combined_df[['text', 'label']]

print("Shuffling data...")
final_df = shuffle(final_df, random_state=42)

# Save the cleaned data
final_df.to_csv('dataset/news.csv', index=False)
print("Success! The unbiased dataset is now saved in dataset/news.csv")