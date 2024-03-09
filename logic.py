import numpy as np
import pandas as pd
import torch
import pickle

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# from deep_translator import GoogleTranslator
from config import HUGGINGFACEHUB_API_TOKEN

df = pd.read_excel("RISTEk.xlsx")
sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# indo_translator = GoogleTranslator(source='en', target='id')
template = """Question: {question}
Instructions: Answer the question so that it is easily understood by the user in English Language. Additional information: RISTEK is sometimes pronounced as RISTEK Fasilkom UI or vice versa.
Context: {context}
Answer: """

prompt = PromptTemplate.from_template(template)
repo_id = "google/gemma-7b-it"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

def query_llm_chain(question, context="No context"):
    response = llm_chain.invoke({"context": context, "question": question})
    return response['text']

def encode_texts(texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sbert_model.to(device)
    embeddings = sbert_model.encode(texts, convert_to_tensor=True, device=device)
    return embeddings.cpu().numpy()
    
def search_documents(query, df, top_n=1):
    with open('ristek_title_embeddings.pkl', 'rb') as f:
        title_embeddings = pickle.load(f)
    
    with open('ristek_content_embeddings.pkl', 'rb') as f:
        content_embeddings = pickle.load(f)
    
    query_embedding = encode_texts([query])[0]

    title_similarities = cosine_similarity([query_embedding], title_embeddings)[0]
    content_similarities = cosine_similarity([query_embedding], content_embeddings)[0]

    title_weight, content_weight = 0.8, 0.2
    total_similarities = title_weight * title_similarities + content_weight * content_similarities

    top_indices = np.argsort(total_similarities)[::-1][:top_n]
    top_documents = [(df.iloc[i]['title'], df.iloc[i]['content'], total_similarities[i]) for i in top_indices]

    return top_documents

def filter_documents_by_threshold(top_documents, threshold):
    filtered_documents = [(title, content, score) for title, content, score in top_documents if score >= threshold]
    return filtered_documents

def demo_rag_qna(query, threshold=0.6, chatbot=False):
    top_documents = search_documents(query, df)
    relevant_documents = filter_documents_by_threshold(top_documents, threshold)

    print("Top similarity:", top_documents[0][2])
    print()

    concat_context = ""
    if chatbot:
        if len(relevant_documents)!=0:
            for i, (title, content, similarity) in enumerate(top_documents, start=1):
                concat_context += (content+"\n")
            # print("Context:\n", concat_context)
            # return indo_translator.translate(query_llm_chain(query, context=concat_context+"\nAnswer it based on this context."))
            return query_llm_chain(query, context=concat_context+"\nAnswer it based on this context.")
        else:
            # return indo_translator.translate(query_llm_chain(query))
            return query_llm_chain(query)
    else:
        if top_documents[0][2] < threshold:
            return 'There are no answers found in the database.'
        for i, (title, content, similarity) in enumerate(relevant_documents, start=1):
            concat_context += (content+"\n")
        # print("Context:\n", concat_context)
        return concat_context


if __name__ == "__main__":
    query = "Could you explain to me about RISTEK Fasilkom UI in the most enganging way?"
    print(demo_rag_qna(query, threshold=0.6, chatbot=False))
    print(demo_rag_qna(query, threshold=0.6, chatbot=True))
    print(demo_rag_qna(query, threshold=0.9, chatbot=True))