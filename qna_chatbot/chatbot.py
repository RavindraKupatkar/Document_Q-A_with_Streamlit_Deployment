import os
import openai
from typing import List  # Added import
from dotenv import load_dotenv

# Langchain imports
from langchain_community.document_loaders import (
    PyPDFLoader, 
    CSVLoader, 
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

class DocumentAssistant:
    """
    A class to manage document loading, processing, and querying
    """
    def __init__(self, source_path: str = None):
        """
        Initialize the DocumentAssistant
        
        :param source_path: Path to documents directory
        """
        self.source_path = source_path or os.getenv('SOURCE_FILES_PATH', './documents/')
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 300))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 90))
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
        )
        
        # Initialize vector store
        self.vectorstore = None
        
    def load_documents(self, file_types: List[str] = None) -> List:
        """
        Load documents from specified directory
        
        :param file_types: List of file extensions to load
        :return: List of loaded documents
        """
        file_types = file_types or ['.pdf', '.csv', '.txt']
        all_documents = []
        
        for filename in os.listdir(self.source_path):
            filepath = os.path.join(self.source_path, filename)
            
            try:
                if any(filename.endswith(ext) for ext in file_types):
                    if filename.endswith('.pdf'):
                        loader = PyPDFLoader(filepath)
                    elif filename.endswith('.csv'):
                        loader = CSVLoader(filepath)
                    else:
                        loader = TextLoader(filepath)
                    
                    documents = loader.load()
                    all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        return all_documents
    
    def chunk_documents(self, documents):
        """
        Split documents into smaller chunks
        
        :param documents: List of documents to chunk
        :return: Chunked documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)
    
    def create_vector_store(self):
        """
        Create a vector store from documents
        """
        documents = self.load_documents()
        chunked_docs = self.chunk_documents(documents)
        
        self.vectorstore = FAISS.from_documents(chunked_docs, self.embeddings)
    
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """
        Retrieve relevant context for a query
        
        :param query: User query
        :param k: Number of top documents to retrieve
        :return: Concatenated context from top documents
        """
        if not self.vectorstore:
            self.create_vector_store()
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        context_docs = retriever.invoke(query)
        
        return "\n\n".join([doc.page_content for doc in context_docs])
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate response using OpenAI API
        
        :param query: User query
        :param context: Retrieved context
        :return: AI-generated response
        """
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful AI assistant. Use the provided context to answer questions precisely and professionally."
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
        
        try:
            response = openai.chat.completions.create(
                model=os.getenv('CHAT_MODEL', 'gpt-3.5-turbo'),
                messages=messages,
                max_tokens=300
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I couldn't generate a response. Please try again."

    def query_documents(self, query: str) -> str:
        """
        Comprehensive method to retrieve context and generate response
        
        :param query: User query
        :return: Generated response
        """
        try:
            # Retrieve context
            context = self.retrieve_context(query)
            
            # Generate response
            return self.generate_response(query, context)
        except Exception as e:
            print(f"Query error: {e}")
            return "An error occurred while processing your query."

# Optional: For direct testing
def main():
    assistant = DocumentAssistant()
    
    while True:
        query = input("Ask a question (or 'quit' to exit): ")
        
        if query.lower() == 'quit':
            break
        
        response = assistant.query_documents(query)
        print("\nResponse:", response)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
