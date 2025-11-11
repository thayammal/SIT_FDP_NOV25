# Tokenization and simple term frequency calculation
from collections import Counter
import math

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
#query = "Take a yoga class to improve flexibility and reduce stress"
query = "I need help with opening a bank account for my savings"

relevant_doc = return_response(query, corpus_of_documents)

print("\nCosine Similarity between query and document:")
similarity_score = cosine_similarity(query, relevant_doc)
print(similarity_score)


print("\nMost relevant document for the query:")
print(relevant_doc)
#  corpus_of_documents = {
#     "Practicing yoga daily can significantly reduce stress and improve physical flexibility.",
#     "Many people join yoga classes to enhance mindfulness and achieve a balanced lifestyle.",
#     "Corporate offices often organize yoga workshops to promote employee wellness.",
#     "She opened a savings account at the local bank to manage her monthly expenses.",
#     "Digital banking apps now make it easier than ever to transfer funds and pay bills.",
#     "The bank is innovating by introducing AI-powered customer service chatbots.",
#     "The software industry is rapidly evolving with the adoption of cloud computing and AI.",
#     "Many developers attend software engineering conferences to stay updated on emerging trends.",
#     "Agile methodology has become a standard practice in modern software companies."
# }

# user_query = "I am a AI scientist and i live in India"

# document = "India is a country where great AI scientist live"

# # Tokenization and simple term frequency calculation
# from collections import Counter
# import math

# query_tokens = user_query.lower().split()
# print("Query Tokens:", query_tokens)

# document_tokens = document.lower().split()
# print("Document Tokens:", document_tokens)

# # Calculate term frequencies
# query_counter = Counter(query_tokens)
# print("Query Counter:", query_counter)

# doc_counter = Counter(document_tokens)
# print("Document Counter:", doc_counter)

# # Using key tokens from the query to display their values
# for token in query_counter.keys():
#     print(token)

# lst = []
# for token in query_counter.keys():
#     lst.append(doc_counter[token])



# # Tokenization and simple term frequency calculation
# print("\nUsing intersection to find common tokens:")
# print("query_counter keys:", query_counter)
# print("doc_counter keys:", doc_counter)

# for tokens in query_counter.keys() & doc_counter.keys():
#     print(tokens)

