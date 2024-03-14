import numpy as np
import pandas as pd
import time
import requests
# import torch
# import pickle

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from config import HUGGINGFACEHUB_API_TOKEN, DATAFRAME_NAME, API_ST_URL, GENERATOR_ID
from processing import replace_rulebase
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# from deep_translator import GoogleTranslator

# For translation purpose
# indo_translator = GoogleTranslator(source='en', target='id')

# Smaller model: sentence-transformers/all-MiniLM-L6-v2
# Need GPU to make it fast for local model
# sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# with open('ristek_title_embeddings.pkl', 'rb') as f:
#     title_embeddings = pickle.load(f)
    
# with open('ristek_content_embeddings.pkl', 'rb') as f:
#     content_embeddings = pickle.load(f)

df = pd.read_excel(DATAFRAME_NAME)
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

template = """Question: {question}
Instructions: Your name is Ristek-GPT who are a helpful AI Assistant, your job is to answer the question so that it is easily understood by the user in English Language.
Context: {context}
Answer: """

prompt = PromptTemplate.from_template(template)

llm = HuggingFaceEndpoint(
    repo_id=GENERATOR_ID,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

# def encode_texts_local(texts):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     sbert_model.to(device)
#     embeddings = sbert_model.encode(texts, convert_to_tensor=True, device=device)
#     return embeddings.cpu().numpy()
    
# def search_documents_local(query, df, top_n=1):
#     query_embedding = encode_texts_local([query])[0]

#     title_similarities = cosine_similarity([query_embedding], title_embeddings)[0]
#     content_similarities = cosine_similarity([query_embedding], content_embeddings)[0]

#     title_weight, content_weight = 0.8, 0.2
#     total_similarities = title_weight * title_similarities + content_weight * content_similarities

#     top_indices = np.argsort(total_similarities)[::-1][:top_n]
#     top_documents = [(df.iloc[i]['title'], df.iloc[i]['content'], total_similarities[i]) for i in top_indices]

#     return top_documents

# We couldn't preprocess the dataframe if we're using API
# If lots of user use it, we're blocked
def query_api(payload):
	response = requests.post(API_ST_URL, headers=headers, json=payload)
	return response.json()

def query_llm_chain(question, context="No context provided. Use the context from your database"):
    response = llm_chain.invoke({"context": context, "question": question})
    return response['text']

def search_documents_api(query, df, top_n=1):
    # Content could be considered for search engine or not

    titles = df['title'].tolist()
    # contents = df['content'].tolist()

    title_similarities = query_api({
        "inputs": {
            "source_sentence": query,
            "sentences": titles
        },
        "options": {"wait_for_model": True}
    })

    # print(title_similarities)
    # content_similarities = query_api({
    #     "inputs": {
    #         "source_sentence": query,
    #         "sentences": contents
    #     },
    #     "options": {"wait_for_model": True}
    # })

    # title_weight, content_weight = 0.8, 0.2
    # total_similarities = [title_weight * t + content_weight * c for t, c in zip(title_similarities, content_similarities)]

    total_similarities = title_similarities

    top_indices = np.argsort(total_similarities)[::-1][:top_n]
    top_documents = [(df.iloc[i]['title'], df.iloc[i]['content'], total_similarities[i]) for i in top_indices]

    return top_documents

def filter_documents_by_threshold(top_documents, threshold):
    filtered_documents = [(title, content, score) for title, content, score in top_documents if score >= threshold]
    return filtered_documents

def demo_rag_qna(query, threshold=0.6, chatbot=False):
    # Specific replacement for ristek name
    query = replace_rulebase(query)

    # Search engine phase
    start_time = time.time()
    # top_documents, encoding_time = search_documents_local(query, df)
    top_documents = search_documents_api(query, df)
    relevant_documents = filter_documents_by_threshold(top_documents, threshold)
    end_time = time.time()
    search_engine_time = end_time - start_time

    print("Top similarity:", top_documents[0][2])
    print()

    start_time = time.time()
    concat_context = ""
    if chatbot:
        if len(relevant_documents)!=0:
            for i, (title, content, similarity) in enumerate(top_documents, start=1):
                concat_context += (content+"\n")
            # print("Context:\n", concat_context)
            # return indo_translator.translate(query_llm_chain(query, context=concat_context+"\nAnswer it based on this context."))
            result = query_llm_chain(query, context=concat_context+"\nAnswer it based on this context.")
            end_time = time.time()
            chatbot_time = end_time - start_time
            return result, search_engine_time, chatbot_time
        else:
            # return indo_translator.translate(query_llm_chain(query))
            result = query_llm_chain(query)
            end_time = time.time()
            chatbot_time = end_time - start_time
            return result, search_engine_time, chatbot_time
    else:
        end_time = time.time()
        chatbot_time = end_time - start_time
        if top_documents[0][2] < threshold:
            return 'There are no answers found in the database.', search_engine_time, chatbot_time
        for i, (title, content, similarity) in enumerate(relevant_documents, start=1):
            concat_context += (content+"\n")
        # print("Context:\n", concat_context)
        return concat_context, search_engine_time, chatbot_time


if __name__ == "__main__":
    query = "Could you explain to me about RISTEK Fasilkom UI in the most enganging way?"
    print(demo_rag_qna(query, threshold=0.6, chatbot=False))
    print(demo_rag_qna(query, threshold=0.6, chatbot=True))
    print(demo_rag_qna(query, threshold=0.9, chatbot=True))