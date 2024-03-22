import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

stemmed_words = ['shit', 'shit', 'lol lol shit','asifhaof','lol','shit']
# Wrap stemmed_words in a list to represent a single document
# stemmed_words = [' '.join(['shit', 'shit', 'lol','asifhaof','lol','shit'])]

# Convert the list of terms into a document-term matrix
doc_term_matrix = vectorizer.fit_transform(stemmed_words)
print(doc_term_matrix)

# Convert the document-term matrix into an array
doc_term_array = doc_term_matrix.toarray()
print(doc_term_array)

# Compute the frequency of terms
term_frequencies = np.sum(doc_term_array, axis=0)
print(term_frequencies)

# Compute the length of the list of term frequencies
list_length = np.linalg.norm(term_frequencies)
print(list_length)