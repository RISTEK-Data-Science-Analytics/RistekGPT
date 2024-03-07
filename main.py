import os
import streamlit as st
import time

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

HUGGINGFACEHUB_API_TOKEN = "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"
# Assuming the HUGGINGFACEHUB_API_TOKEN is set in your environment for security reasons.
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize your LLM template, endpoint, and chain outside the main function to avoid reinitialization on each rerun.
template = """Question: {question}

Answer: """
prompt = PromptTemplate.from_template(template)
repo_id = "google/gemma-7b-it"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

def query_llm_chain(question):
    return llm_chain.invoke(question)['text']

def main():
    st.title("RistekGPT")

    question = st.text_input("Ask a question:", "")

    # "Submit" button.
    submit_button = st.button("Submit")

    if submit_button:
        response_container = st.empty()  # Container to hold and update the response.

        response = ""
        with st.spinner("Thinking..."):
            response = query_llm_chain(question)

        formatted_response = ""  # Initialize an empty string to accumulate the response.
        for i in range(len(response)):
            time.sleep(0.002)  # Adjust sleep time to simulate typing speed.
            formatted_response += response[i]  # Append the next character to the accumulated response.
            response_container.markdown(formatted_response, unsafe_allow_html=True)  # Render the response as Markdown.

    # Footer indicating "Powered by Gemma".
    st.markdown("---")
    st.markdown("<div style='text-align: center;'>Powered by Gemma</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
