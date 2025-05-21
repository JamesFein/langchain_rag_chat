import unittest
import os
import shutil
from unittest.mock import patch, MagicMock

# Adjust import path for RAGHandler
# This assumes that tests are run from the root of the project (e.g. `python -m unittest discover`)
# or that rag_app is in PYTHONPATH
try:
    from rag_app.app.rag_handler import RAGHandler, UPLOAD_DIRECTORY, VECTOR_STORE_PATH
except ImportError:
    # Fallback if running directly from tests directory or rag_app/app is not in path
    # This might require adjusting PYTHONPATH when running tests
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
    from rag_handler import RAGHandler, UPLOAD_DIRECTORY, VECTOR_STORE_PATH


class TestRAGHandler(unittest.TestCase):

    def setUp(self):
        """Set up test environment for RAGHandler."""
        self.test_dir = os.path.dirname(__file__) # rag_app/tests/
        self.base_app_dir = os.path.dirname(self.test_dir) # rag_app/

        # Use a dummy API key for tests
        self.dummy_openai_api_key = "test_api_key"
        os.environ["OPENAI_API_KEY"] = self.dummy_openai_api_key

        # Define test-specific paths relative to the base_app_dir (rag_app/)
        self.test_upload_dir_name = "test_uploads"
        self.test_vector_store_dir_name = "test_vector_store"
        
        self.test_upload_path = os.path.join(self.base_app_dir, self.test_upload_dir_name)
        self.test_vector_store_path = os.path.join(self.base_app_dir, self.test_vector_store_dir_name, "faiss_index")

        # Create test directories
        os.makedirs(self.test_upload_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.test_vector_store_path), exist_ok=True)

        # Instantiate RAGHandler
        self.rag_handler = RAGHandler(openai_api_key=self.dummy_openai_api_key)

        # Override RAGHandler's paths to use test-specific ones
        # We need to ensure RAGHandler uses these paths internally.
        # The current RAGHandler uses constants defined at module level for paths.
        # For robust testing, it's better if RAGHandler takes these as parameters
        # or if we can patch the constants *before* RAGHandler instance is created
        # or specifically when methods like _load_vector_store / _save_vector_store are called.
        
        # Let's patch the global constants directly in the rag_handler module for the scope of these tests
        self.upload_dir_patch = patch(f'{RAGHandler.__module__}.UPLOAD_DIRECTORY', self.test_upload_dir_name)
        self.vector_store_patch = patch(f'{RAGHandler.__module__}.VECTOR_STORE_PATH', os.path.join(self.test_vector_store_dir_name, "faiss_index"))
        
        self.mock_upload_dir = self.upload_dir_patch.start()
        self.mock_vector_store_path = self.vector_store_patch.start()

        # Re-initialize rag_handler or its path-dependent components if necessary,
        # or ensure that its methods construct paths dynamically using the patched constants.
        # RAGHandler's __init__ calls _load_vector_store, which uses the module-level constant.
        # So, we need to re-initialize after patching or make sure the instance uses the patched path.
        
        # For simplicity in this context, we'll assume the RAGHandler's methods will re-evaluate
        # os.path.join(os.path.dirname(__file__), '..', VECTOR_STORE_PATH) correctly due to the patch.
        # Let's reset vector_store to None to force it to use the new path on load/save.
        self.rag_handler.vector_store = None # Reset to ensure it uses patched path on next op
        # Manually ensure its internal paths reflect the test paths if methods don't re-evaluate
        # For example, if RAGHandler's constructor sets instance variables for these paths.
        # Looking at rag_handler.py, _load_vector_store and _save_vector_store use:
        # full_vector_store_path = os.path.join(os.path.dirname(__file__), '..', VECTOR_STORE_PATH)
        # This means the patch on VECTOR_STORE_PATH should work.
        # Similarly for UPLOAD_DIRECTORY used in main.py, but not directly in RAGHandler methods.

    def tearDown(self):
        """Clean up test environment."""
        # Stop the patches
        self.upload_dir_patch.stop()
        self.vector_store_patch.stop()

        if os.path.exists(self.test_upload_path):
            shutil.rmtree(self.test_upload_path)
        
        test_vector_store_base = os.path.join(self.base_app_dir, self.test_vector_store_dir_name)
        if os.path.exists(test_vector_store_base):
            shutil.rmtree(test_vector_store_base)
        
        del os.environ["OPENAI_API_KEY"]

    def test_load_and_process_documents_txt(self):
        """Test loading a .txt file and processing it into the vector store."""
        dummy_txt_filename = "test_doc.txt"
        # The path for file creation should be the *actual* test upload path
        dummy_txt_filepath = os.path.join(self.test_upload_path, dummy_txt_filename)

        with open(dummy_txt_filepath, "w") as f:
            f.write("This is a test document for RAGHandler.")

        # Ensure vector store is initially None or clear it
        self.rag_handler.vector_store = None
        if os.path.exists(self.test_vector_store_path):
             shutil.rmtree(self.test_vector_store_path) # Clean up from previous potential runs

        # Process the document
        # load_and_process_documents expects a list of absolute paths
        self.rag_handler.load_and_process_documents([dummy_txt_filepath])

        self.assertIsNotNone(self.rag_handler.vector_store, "Vector store should be initialized after processing.")
        
        # The vector store path used by RAGHandler should be the patched one.
        # _save_vector_store uses os.path.join(os.path.dirname(__file__), '..', VECTOR_STORE_PATH)
        # __file__ here is rag_handler.py, so '..' is rag_app/.
        # So the effective path is rag_app/test_vector_store/faiss_index
        expected_vector_store_file_path = os.path.join(self.base_app_dir, self.mock_vector_store_path, "index.faiss")
        self.assertTrue(os.path.exists(expected_vector_store_file_path),
                        f"FAISS index file should be created at {expected_vector_store_file_path}")

    @patch(f'{RAGHandler.__module__}.RetrievalQA') # Patch RetrievalQA where it's used
    def test_get_answer_with_vector_store(self, mock_retrieval_qa):
        """Test get_answer when the vector store is initialized and RetrievalQA is mocked."""
        # 1. Ensure vector store is initialized (e.g., by loading a dummy doc)
        dummy_txt_filename = "test_query_doc.txt"
        dummy_txt_filepath = os.path.join(self.test_upload_path, dummy_txt_filename)
        with open(dummy_txt_filepath, "w") as f:
            f.write("The capital of Testland is Testville.")
        
        self.rag_handler.vector_store = None # Reset
        if os.path.exists(self.test_vector_store_path):
             shutil.rmtree(self.test_vector_store_path)
        self.rag_handler.load_and_process_documents([dummy_txt_filepath])
        self.assertIsNotNone(self.rag_handler.vector_store, "Vector store should be initialized.")

        # 2. Mock RetrievalQA chain
        mock_qa_instance = MagicMock()
        mock_qa_instance.invoke.return_value = {"result": "Mocked answer: Testville is the capital."}
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_instance

        # 3. Call get_answer
        query = "What is the capital of Testland?"
        answer = self.rag_handler.get_answer(query)

        # 4. Assertions
        self.assertEqual(answer, "Mocked answer: Testville is the capital.")
        mock_retrieval_qa.from_chain_type.assert_called_once()
        mock_qa_instance.invoke.assert_called_once_with({"query": query})

    def test_get_answer_no_vector_store(self):
        """Test get_answer when the vector store is not initialized."""
        # Ensure vector_store is None and no physical store exists
        self.rag_handler.vector_store = None
        
        # The path used by _load_vector_store is rag_app/test_vector_store/faiss_index
        physical_store_path = os.path.join(self.base_app_dir, self.mock_vector_store_path)
        if os.path.exists(physical_store_path):
            shutil.rmtree(physical_store_path)

        # Attempt to load (which should fail or result in None)
        self.rag_handler._load_vector_store() # This will print "No existing vector store found..."
        self.assertIsNone(self.rag_handler.vector_store, "Vector store should be None.")
        
        answer = self.rag_handler.get_answer("Any query")
        self.assertIsNone(answer, "Answer should be None when vector store is not initialized.")

if __name__ == '__main__':
    unittest.main()
