import streamlit as st
from logic import demo_rag_qna
import time
from config import GENERATOR_NAME

# Hide the github logo
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

def main():
    st.title(GENERATOR_NAME)

    question = st.text_input("Ask a question:", "")

    with_chatbot_button = st.button("With Chatbot")
    without_chatbot_button = st.button("Without Chatbot")

    response_container = st.empty()  # Container to hold and update the response.

    if with_chatbot_button or without_chatbot_button:
        chatbot_mode = with_chatbot_button  # If the "With Chatbot" button is pressed, set chatbot_mode to True, otherwise False.

        formatted_response = ""
        with st.spinner("Thinking..."):
            for part in demo_rag_qna(question, threshold=0.75, chatbot=chatbot_mode, use_streaming=True):
                if isinstance(part, dict) and "type" in part and part["type"] == "metrics":
                    # The part is the metrics dict
                    search_engine_time = part.get("search_engine_time", 0)
                    chatbot_time = part.get("chatbot_time", 0)

                    # Display the timing information
                    st.write("Search Engine Time:", round(search_engine_time, 2), "seconds")
                    st.write("Chatbot Time:", round(chatbot_time, 2), "seconds")
                else:
                    # The part is a content token
                    formatted_response += part
                    response_container.markdown(formatted_response, unsafe_allow_html=True)

    # Footer indicating "Powered by Gemma".
    st.markdown("---")
    st.markdown("<div style='text-align: center;'>Powered by Gemma</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
