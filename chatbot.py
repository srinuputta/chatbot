import nltk
import random
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only first time)
nltk.download('punkt')
nltk.download('wordnet')

# Sample chatbot corpus
corpus = """
Hello! I am your AI chatbot. How can I help you today?
I can answer basic questions about Python, NLTK, and AI.
Python is a programming language.
NLTK is a Natural Language Toolkit for text processing.
AI stands for Artificial Intelligence.
Machine Learning is a subset of AI.
Goodbye and have a nice day!
"""

# Preprocessing
lemmer = WordNetLemmatizer()
sent_tokens = nltk.sent_tokenize(corpus.lower())  # sentence tokens

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(str.maketrans('', '', string.punctuation))))

# Greeting inputs and responses
GREETING_INPUTS = ("hello", "hi", "hey", "greetings")
GREETING_RESPONSES = ["Hello!", "Hi there!", "Hey!", "Greetings!"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generate chatbot response
def chatbot_response(user_input):
    user_input = user_input.lower()
    sent_tokens.append(user_input)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = vectorizer.fit_transform(sent_tokens)

    # Cosine similarity
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    score = flat[-1]

    sent_tokens.pop()  # remove last input for next round

    if score == 0:
        return "I'm sorry, I do not understand."
    else:
        return sent_tokens[idx]

# Chat loop
print("Chatbot: Hello! I am your chatbot. Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['bye', 'exit', 'quit']:
        print("Chatbot: Goodbye! Have a nice day 😊")
        break
    elif greeting(user_input):
        print("Chatbot:", greeting(user_input))
    else:
        print("Chatbot:", chatbot_response(user_input))
