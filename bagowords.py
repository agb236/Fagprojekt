# exercise 3.1.4
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from similarity import similarity

# Load the textDocs.txt as a long string into raw_file:
with open('/Users/agb/Downloads/tweetstrans.txt', 'r') as f:
    raw_file = f.read()
    
# raw_file contains sentences seperated by newline characters, 
# so we split by '\n':
corpus = raw_file.split('\n')

# corpus is now list of "documents" (sentences), but some of them are empty, 
# because textDocs.txt has a lot of empty lines, we filter/remove them:
corpus = list(filter(None, corpus))

# Display the result
print('Document-term matrix analysis')
print()
print('Corpus (5 documents/sentences):')
print(np.asmatrix(corpus[0:4]))
print()

# To automatically obtain the bag of words representation, we use sklearn's
# feature_extraction.text module, which has a function CountVectorizer.
# We make a CounterVectorizer:
vectorizer = CountVectorizer(token_pattern=r'\b[^\d\W]+\b')   

# The object vectorizer can now be used to first 'fit' the vectorizer to the
# corpus, and the subsequently transform the data. We start by fitting:
vectorizer.fit(corpus)


# The vectorizer has now determined the unique terms (or tokens) in the corpus
# and we can extract them using:
attributeNames = vectorizer.get_feature_names_out()
print('Found terms:')
print(attributeNames)
print(attributeNames.shape)
print(attributeNames[2745])
print()

# The next step is to count how many times each term is found in each document,
# which we do using the transform function:
X = vectorizer.transform(corpus)
N,M = X.shape
print('Number of documents (data objects, N):\t %i' % N)
print('Number of terms (attributes, M):\t %i' % M )
print()
print('Document-term matrix:')
print(X.toarray())
print(X.toarray().shape)
print()

# Load and process the stop words in a similar manner:
with open('/Users/agb/Desktop/Machine_Learning/02450Toolbox_Python/Data/stopWords.txt', 'r') as f:
    raw_file = f.read()
stopwords = raw_file.split('\n')

# When making the CountVectorizer, we now input the stop words:
vectorizer = CountVectorizer(token_pattern=r'\b[^\d\W]+\b', 
                             stop_words=stopwords)    
# Determine the terms in the corpus
vectorizer.fit(corpus)
# ... and count the frequency of each term within a document:
X = vectorizer.transform(corpus)
attributeNames = vectorizer.get_feature_names_out()
N,M = X.shape

# Display the result
print('Document-term matrix analysis (using stop words)')
print()
print('Number of documents (data objects, N):\t %i' % N)
print('No. terms after removing stop words (attributes, M):\t %i' % M )
print()
print('Found terms (no stop words):')
print(attributeNames)
print()
print('Document-term matrix:')
print(X.toarray())
print()

X = X.toarray()

# Query vector
q = np.asarray(X[1,:])
# notice, that you could get the query vector using the vectorizer, too:
#q = vectorizer.transform(['matrix rank solv'])
#q = np.asarray(q.toarray())
# or use any other string:
#q = vectorizer.transform(['Can I Google how to fix my problem?'])
#q = np.asarray(q.toarray())

# Method 2 (one line of code with no iterations - faster)
sim = (q @ X.T).T / (np.sqrt(np.power(X,2).sum(axis=1)) * np.sqrt(np.power(q,2).sum()))

# Method 3 (use the "similarity" function)
sim = similarity(X, q, 'cos')


# Display the result
print('Query vector:\n {0}\n'.format(q))
print('Similarity results:\n {0}'.format(sim))



