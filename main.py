import streamlit as st
from logic import demo_rag_qna

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
    st.title("Ristek-GPT")

    question = st.text_input("Ask a question:", "")

    with_chatbot_button = st.button("With Chatbot")
    without_chatbot_button = st.button("Without Chatbot")

    response_container = st.empty()  # Container to hold and update the response.

    if with_chatbot_button or without_chatbot_button:
        chatbot_mode = with_chatbot_button  # If the "With Chatbot" button is pressed, set chatbot_mode to True, otherwise False.

        response = ""
        with st.spinner("Thinking..."):
            # Adjust the chatbot parameter based on the button pressed.
            response, search_engine_time, chatbot_time = demo_rag_qna(question, threshold=0.6, chatbot=chatbot_mode)

        formatted_response = ""  # Initialize an empty string to accumulate the response.
        for i in range(len(response)):
            # time.sleep(0.001)  # Adjust sleep time to simulate typing speed.
            formatted_response += response[i]  # Append the next character to the accumulated response.
            response_container.markdown(formatted_response, unsafe_allow_html=True)  # Render the response as Markdown.
    
        # Display chatbot and search engine time
        st.write("------")
        st.write("Search Engine Time:", round(search_engine_time,2), "seconds")
        st.write("Chatbot Time:", round(chatbot_time,2), "seconds")

    # Footer indicating "Powered by Gemma".
    st.markdown("---")
    st.markdown("<div style='text-align: center;'>Powered by Gemma</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
