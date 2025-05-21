import os
import pickle
from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Define constants for paths (relative to rag_app directory)
# The full path from project root will be rag_app/vector_store/faiss_index and rag_app/uploads
VECTOR_STORE_PATH = "vector_store/faiss_index"
UPLOAD_DIRECTORY = "uploads"


class RAGHandler:
    def __init__(self, openai_api_key: str):
        """
        Initializes the RAGHandler.

        Args:
            openai_api_key: The API key for OpenAI services.
        """
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        self.vector_store: Optional[FAISS] = None
        self._load_vector_store()

    def _load_vector_store(self) -> None:
        """
        Loads an existing FAISS vector store from the defined path.
        """
        # Ensure the path is constructed relative to the rag_app directory
        full_vector_store_path = os.path.join(os.path.dirname(__file__), '..', VECTOR_STORE_PATH)
        if os.path.exists(full_vector_store_path):
            try:
                self.vector_store = FAISS.load_local(
                    full_vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True, # Required by FAISS for pickle loading
                )
                print(f"Vector store loaded successfully from {full_vector_store_path}")
            except Exception as e:
                print(f"Error loading vector store from {full_vector_store_path}: {e}")
                self.vector_store = None
        else:
            print(f"No existing vector store found at {full_vector_store_path}. A new one will be created if documents are processed.")

    def _save_vector_store(self) -> None:
        """
        Saves the current FAISS vector store to the defined path.
        """
        if self.vector_store is not None:
            # Ensure the path is constructed relative to the rag_app directory
            full_vector_store_path = os.path.join(os.path.dirname(__file__), '..', VECTOR_STORE_PATH)
            try:
                os.makedirs(os.path.dirname(full_vector_store_path), exist_ok=True)
                self.vector_store.save_local(full_vector_store_path)
                print(f"Vector store saved successfully to {full_vector_store_path}")
            except Exception as e:
                print(f"Error saving vector store to {full_vector_store_path}: {e}")

    def load_and_process_documents(self, file_paths: List[str]) -> None:
        """
        Loads documents from the given file paths, processes them, and updates the vector store.

        Args:
            file_paths: A list of absolute paths to the documents to be processed.
        """
        all_chunks = []
        for file_path in file_paths:
            print(f"Processing file: {file_path}...")
            try:
                if not os.path.exists(file_path):
                    print(f"Warning: File not found at {file_path}. Skipping.")
                    continue

                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif file_ext == ".txt":
                    loader = TextLoader(file_path)
                elif file_ext == ".docx":
                    loader = Docx2txtLoader(file_path)
                # TODO: Add UnstructuredFileLoader for other types if needed
                # elif file_ext in [".html", ".md", ".csv", ".xlsx", ".pptx"]:
                #     loader = UnstructuredFileLoader(file_path)
                else:
                    print(f"Unsupported file type: {file_ext} for file {file_path}. Skipping.")
                    continue

                documents = loader.load()
                chunks = self.text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
                print(f"Successfully loaded and split {file_path}. {len(chunks)} chunks created.")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        if not all_chunks:
            print("No documents were successfully processed. Vector store not updated.")
            return

        if self.vector_store is None:
            try:
                self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
                print("New vector store created with processed documents.")
            except Exception as e:
                print(f"Error creating new vector store: {e}")
                return # Do not proceed to save if creation failed
        else:
            try:
                self.vector_store.add_documents(all_chunks)
                print("Processed documents added to existing vector store.")
            except Exception as e:
                print(f"Error adding documents to existing vector store: {e}")
                return # Do not proceed to save if adding failed

        self._save_vector_store()
        print("Document processing complete. Vector store updated and saved.")

    def get_answer(self, query: str) -> Optional[str]:
        """
        Retrieves an answer from the RAG system based on the given query.

        Args:
            query: The question to ask the RAG system.

        Returns:
            The answer string, or None if an error occurs or the store is not initialized.
        """
        if self.vector_store is None:
            print("Vector store not initialized. Please upload documents first.")
            return None

        try:
            retriever = self.vector_store.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Can be "stuff", "map_reduce", "refine", "map_rerank"
                retriever=retriever,
                return_source_documents=True,  # Set to False if source documents are not needed
            )

            response = qa_chain.invoke({"query": query})
            
            # The actual result is usually in 'result'. 
            # The 'source_documents' key contains the retrieved documents.
            answer = response.get("result")
            # source_docs = response.get("source_documents")
            # if source_docs:
            #     print(f"\nRetrieved {len(source_docs)} source documents:")
            #     for i, doc in enumerate(source_docs):
            #         print(f"Document {i+1}:")
            #         print(f"  Page Content: {doc.page_content[:200]}...") # Print snippet
            #         print(f"  Metadata: {doc.metadata}")


            if answer:
                return str(answer)
            else:
                print("No answer found in the QA chain response.")
                return None

        except Exception as e:
            print(f"Error during QA process: {e}")
            return None


if __name__ == '__main__':
    # This is a placeholder for testing the RAGHandler
    # You would need to set your OPENAI_API_KEY environment variable
    # or pass it directly for this test to work.
    print("RAGHandler module loaded. This is a basic test section.")

    # Example of how to use (requires an API key and sample files):
    # Ensure you have an 'uploads' and 'vector_store' directory in 'rag_app'
    # And place some test files in 'rag_app/uploads'
    
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # if not OPENAI_API_KEY:
    #     print("OPENAI_API_KEY environment variable not set. Cannot run example.")
    # else:
    #     print(f"Found OpenAI API Key: {OPENAI_API_KEY[:5]}...") # Print only a part for security
    #     handler = RAGHandler(openai_api_key=OPENAI_API_KEY)

    #     # Create dummy files for testing if they don't exist
    #     sample_files_dir = os.path.join(os.path.dirname(__file__), '..', UPLOAD_DIRECTORY)
    #     os.makedirs(sample_files_dir, exist_ok=True)
        
    #     test_txt_path = os.path.join(sample_files_dir, "sample_test.txt")
    #     if not os.path.exists(test_txt_path):
    #         with open(test_txt_path, "w") as f:
    #             f.write("This is a test document. The capital of France is Paris. Mars is a planet in our solar system.")
    #         print(f"Created dummy file: {test_txt_path}")

    #     files_to_process = [test_txt_path]
        
    #     if files_to_process and os.path.exists(files_to_process[0]):
    #         print(f"Test files to process: {files_to_process}")
    #         handler.load_and_process_documents(files_to_process)
    #         print("Test processing complete.")

    #         if handler.vector_store:
    #             # Test the get_answer method
    #             query1 = "What is the capital of France?"
    #             print(f"\nQuerying: {query1}")
    #             answer1 = handler.get_answer(query1)
    #             print(f"Answer: {answer1}")

    #             query2 = "Tell me about Mars."
    #             print(f"\nQuerying: {query2}")
    #             answer2 = handler.get_answer(query2)
    #             print(f"Answer: {answer2}")
                
    #             query3 = "What is the meaning of life?" # A query not in the document
    #             print(f"\nQuerying: {query3}")
    #             answer3 = handler.get_answer(query3)
    #             print(f"Answer: {answer3}")
    #     else:
    #         print(f"Test file {test_txt_path} not found. Skipping processing example.")

    #     # Clean up dummy file
    #     # if os.path.exists(test_txt_path):
    #     #     os.remove(test_txt_path)
    #     #     print(f"Cleaned up dummy file: {test_txt_path}")

    # print("\nTo run the example, uncomment the code block above, ensure 'sample_test.txt' can be created/found in rag_app/uploads, and set your OPENAI_API_KEY.")
    pass
