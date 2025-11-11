# Tokenization and simple term frequency calculation
from collections import Counter
import math

corpus_of_documents = [
    "Practicing yoga daily can significantly reduce stress and improve physical flexibility.",
    "Many people join yoga classes to enhance mindfulness and achieve a balanced lifestyle.",
    "Corporate offices often organize yoga workshops to promote employee wellness.",
    "She opened a savings account at the local bank to manage her monthly expenses.",
    "Digital banking apps now make it easier than ever to transfer funds and pay bills.",
    "The bank is innovating by introducing AI-powered customer service chatbots.",
    "The software industry is rapidly evolving with the adoption of cloud computing and AI.",
    "Many developers attend software engineering conferences to stay updated on emerging trends.",
    "Agile methodology has become a standard practice in modern software companies."
]

#cosine similarity calculation
def cosine_similarity(query, document):
    query_tokens = query.lower().split()
    document_tokens = document.lower().split()

    #create counters for query and document
    query_counter = Counter(query_tokens)
    doc_counter = Counter(document_tokens)

    # Calculate dot product
    dot_product = sum(query_counter[token] * doc_counter[token] for token in query_counter.keys() & doc_counter.keys())
    
    # Calculate magnitudes
    query_magnitude = math.sqrt(sum(query_counter[token] ** 2 for token in query_counter))
    doc_magnitude = math.sqrt(sum(doc_counter[token] ** 2 for token in doc_counter))
    
    #calculate cosine similarity
    similarity = dot_product / (query_magnitude * doc_magnitude) if query_magnitude and doc_magnitude else 0.0
    return similarity


def return_response(query, corpus):
    similarities = []
    for doc in corpus:
        sim_score = cosine_similarity(query, doc)
        similarities.append(sim_score)
    return corpus_of_documents[similarities.index(max(similarities))]


import requests
import json

user_input = "How does yoga help us?"
relevant_document = return_response(user_input, corpus_of_documents)

full_response = []

prompt = """
 You are a bot that provides information about yoga and its benefits.
 You answer in concise and informative manner.
 Answer only in 30 words.
 The user query is : {user_input}
 Compile a detailed response based on the user query.
"""
url = "http://localhost:11434/api/generate"  # Ollama API endpoint

data = {
    "model": "gemma3:1b",
    "prompt": prompt.format(user_input=user_input, relevant_document=relevant_document),
}

headers = {"Content-Type": "application/json" }

response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)

try:
    for line in response.iter_lines():
        #filter out keep-alive new lines
        if line:
            decoded_line = json.loads(line.decode('utf-8'))
            #print(decoded_line)
            full_response.append(decoded_line['response'])
            
finally:
    response.close()

print("\nOllama Response:")
print(''.join(full_response))