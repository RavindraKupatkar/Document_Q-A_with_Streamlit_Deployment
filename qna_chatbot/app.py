import streamlit as st
import os
from dotenv import load_dotenv
from chatbot import DocumentAssistant

# Load environment variables
load_dotenv()

def initialize_session_state():
    """
    Initialize or reset session state variables
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your document assistant. Ask me anything about the loaded documents."}
        ]
    
    if "assistant" not in st.session_state:
        st.session_state.assistant = DocumentAssistant()

def display_chat_history():
    """
    Display existing chat messages
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    """
    Process user input and generate response
    """
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching through documents..."):
                try:
                    # Query documents and get response
                    response = st.session_state.assistant.query_documents(prompt)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")

def main():
    """
    Main Streamlit application
    """
    # Page Configuration
    st.set_page_config(
        page_title="Document Q&A Assistant", 
        page_icon=":book:", 
        layout="wide"
    )
    
    # Title and Description
    st.title("📚 Document Intelligence Assistant")
    st.write("Ask questions about your uploaded documents. AI-powered insights at your fingertips.")
    
    # Sidebar for additional controls
    with st.sidebar:
        st.header("Document Management")
        
        # Document source path configuration
        source_path = st.text_input(
            "Document Source Path", 
            value=os.getenv('SOURCE_FILES_PATH', './Ravindra-Kupatkar-resume.pdf/')
        )
        
        # Refresh documents button
        if st.button("Refresh Document Index"):
            try:
                # Reinitialize assistant with new path
                st.session_state.assistant = DocumentAssistant(source_path)
                st.success("Document index refreshed successfully!")
            except Exception as e:
                st.error(f"Error refreshing documents: {e}")
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat history
    display_chat_history()
    
    # Handle user input
    handle_user_input()

if __name__ == "__main__":
    main()
