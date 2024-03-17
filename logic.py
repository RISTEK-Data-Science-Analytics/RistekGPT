import numpy as np
import pandas as pd
import time
import requests
import torch
import pickle
import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from config import DATAFRAME_NAME, API_ST_URL, GENERATOR_ID, \
                    TITLE_EMBEDDING, CONTENT_EMBEDDING, ST_URL, instruction
from processing import replace_rulebase
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from streaming import QueueStreamingCallbackHandler, start_llm_generation, stream_tokens
# from deep_translator import GoogleTranslator

# For translation purpose
# indo_translator = GoogleTranslator(source='en', target='id')

HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
df = pd.read_excel(DATAFRAME_NAME)
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

template = """Question: {question}
Instruction: {instruction}
Context: {context}
Answer: """

sbert_model = SentenceTransformer(ST_URL)
sbert_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sbert_model.to(device)

with open(TITLE_EMBEDDING, 'rb') as f:
    title_embeddings = pickle.load(f)
    
with open(CONTENT_EMBEDDING, 'rb') as f:
    content_embeddings = pickle.load(f)

prompt = PromptTemplate.from_template(template)

llm = HuggingFaceEndpoint(
    repo_id=GENERATOR_ID,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    streaming=True,
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

# We couldn't preprocess the dataframe if we're using API
# If lots of user use it, we're blocked
def query_api(payload):
	response = requests.post(API_ST_URL, headers=headers, json=payload)
	return response.json()

def query_llm_chain(question, instruction, streaming=False, context="No context provided. Use the context from your database"):
    if streaming:    
        full_prompt = prompt.format(question=question, instruction=instruction, context=context)
        callback_handler = QueueStreamingCallbackHandler()
        queue = start_llm_generation(llm, full_prompt, callback_handler)
        for token in stream_tokens(queue):
            # print(token)
            yield token
    else:
        response = llm_chain.invoke({"context": context, "instruction":instruction, "question": question})
        yield response['text']

def search_documents_api(query, df, top_n=1, title_weight=1):
    # Content could be considered for search engine or not

    titles = df['title'].tolist()
    contents = df['content'].tolist()
    content_weight = 1 - title_weight

    title_similarities = query_api({
        "inputs": {
            "source_sentence": query,
            "sentences": titles
        },
        "options": {"wait_for_model": True}
    })
    total_similarities = title_similarities

    if content_weight!=0:
        content_similarities = query_api({
            "inputs": {
                "source_sentence": query,
                "sentences": contents
            },
            "options": {"wait_for_model": True}
        })

        total_similarities = [title_weight * t + content_weight * c for t, c in zip(title_similarities, content_similarities)]

    top_indices = np.argsort(total_similarities)[::-1][:top_n]
    top_documents = [(df.iloc[i]['title'], df.iloc[i]['content'], total_similarities[i]) for i in top_indices]

    return top_documents

def encode_texts(texts):
    embeddings = sbert_model.encode(texts, convert_to_tensor=True, device=device)
    return embeddings.cpu().numpy()

def search_documents_local(query, df, top_n=1, title_weight=1):
    query_embedding = encode_texts([query])[0]

    content_weight = 1 - title_weight
    title_similarities = cosine_similarity([query_embedding], title_embeddings)[0]
    total_similarities = title_similarities

    if content_weight!=0:
        content_similarities = cosine_similarity([query_embedding], content_embeddings)[0]
        total_similarities = title_weight * title_similarities + content_weight * content_similarities

    top_indices = np.argsort(total_similarities)[::-1][:top_n]
    top_documents = [(df.iloc[i]['title'], df.iloc[i]['content'], total_similarities[i]) for i in top_indices]

    return top_documents

def filter_documents_by_threshold(top_documents, threshold):
    filtered_documents = [(title, content, score) for title, content, score in top_documents if score >= threshold]
    return filtered_documents

def demo_rag_qna(query, threshold=0.6, chatbot=False, use_streaming=False):
    # Specific replacement for preprocess purpose
    query = replace_rulebase(query)

    # Search engine phase
    search_engine_start_time = time.time()
    top_documents = search_documents_local(query, df)
    # top_documents = search_documents_api(query, df)

    relevant_documents = filter_documents_by_threshold(top_documents, threshold)
    search_engine_end_time = time.time()
    search_engine_time = search_engine_end_time - search_engine_start_time

    print("Top similarity:", top_documents[0][2])
    print()

    chatbot_start_time = time.time()
    concat_context = ""
    if chatbot:
        if len(relevant_documents)!=0:
            for i, (title, content, similarity) in enumerate(top_documents, start=1):
                concat_context += (content+"\n")
            # indo_translator.translate(query_llm_chain(query, instruction, streaming=use_streaming, context=concat_context+"\nAnswer it based on this context."))
    
            for result in query_llm_chain(query, instruction, streaming=use_streaming, context=concat_context+"\nAnswer it based on this context."):
                yield result
            chatbot_end_time = time.time()
            chatbot_time = chatbot_end_time - chatbot_start_time
            yield {"type": "metrics", "search_engine_time": search_engine_time, "chatbot_time": chatbot_time}
        else:
            # indo_translator.translate(query_llm_chain(query, instruction, streaming=use_streaming))
            for result in query_llm_chain(query, instruction, streaming=use_streaming):
                # print(result)
                yield result

            chatbot_end_time = time.time()
            chatbot_time = chatbot_end_time - chatbot_start_time
            yield {"type": "metrics", "search_engine_time": search_engine_time, "chatbot_time": chatbot_time}
    else:
        chatbot_end_time = time.time()
        chatbot_time = chatbot_end_time - chatbot_start_time
        if top_documents[0][2] < threshold:
            yield 'There are no answers found in the database.'
            yield {"type": "metrics", "search_engine_time": search_engine_time, "chatbot_time": chatbot_time}
        else:
            for i, (title, content, similarity) in enumerate(relevant_documents, start=1):
                concat_context += (content+"\n")
            yield concat_context
            yield {"type": "metrics", "search_engine_time": search_engine_time, "chatbot_time": chatbot_time}


if __name__ == "__main__":
    query = "Could you explain to me about RISTEK Fasilkom UI in the most enganging way?"
    print(demo_rag_qna(query, threshold=0.6, chatbot=False))
    print(demo_rag_qna(query, threshold=0.6, chatbot=True))
    print(demo_rag_qna(query, threshold=0.9, chatbot=True))