import os
import streamlit as st
import logging
import sys

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="Timetable Management System",
    page_icon="📚",
    layout="wide",
)

# Then configure logging
def setup_logging():
    # Create logger
    logger = logging.getLogger("timetable_generator")
    logger.setLevel(logging.DEBUG)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    
    return logger

# Create logger instance
logger = setup_logging()

# Add HTTP request monitoring
try:
    # Optional: Monitor HTTP requests
    import http.client as http_client
    http_client.HTTPConnection.debuglevel = 1
    
    # You need to initialize logging, as shown above
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True
except ImportError:
    logger.warning("Failed to set up HTTP request debugging - requests package may not be installed")

# Now continue with the rest of the imports
from langchain_ollama import OllamaLLM  
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredMarkdownLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import pandas as pd
import tempfile
import json
import re
from typing import List, Dict, Optional, Any

# ----- Constants ----- #
DEFAULT_MODEL = "llama3"
AVAILABLE_OLLAMA_MODELS = ["gemma3:4b", "deepseek-r1:1.5b"]
AVAILABLE_GROQ_MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it", "llama3-70b-8192", "deepseek-r1-distill-llama-70b", "meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/Llama-Guard-4-12B", "mistral-saba-24b", "qwen-qwq-32b"]
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# Helper function for logging response generation
def log_response_generation(messages, response):
    """Log details about LLM response generation"""
    logger.info("=== LLM Request/Response Details ===")
    logger.info(f"Number of messages sent: {len(messages)}")
    
    for i, msg in enumerate(messages):
        logger.info(f"Message {i+1} role: {type(msg).__name__}")
        # Log first 100 chars of content
        content_preview = msg.content[:100] + ("..." if len(msg.content) > 100 else "")
        logger.info(f"Message {i+1} content preview: {content_preview}")
    
    # Log response length and preview
    response_length = len(response) if response else 0
    logger.info(f"Response length: {response_length} characters")
    if response:
        response_preview = response[:100] + ("..." if len(response) > 100 else "")
        logger.info(f"Response preview: {response_preview}")
    logger.info("=== End of LLM Details ===")

# ----- Helper Functions ----- #
def load_document(file, file_type):
    """Load document based on file type"""
    logger.info(f"Loading document of type: {file_type}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name
    
    if file_type == "md":
        loader = UnstructuredMarkdownLoader(tmp_path)
    elif file_type == "csv":
        loader = CSVLoader(tmp_path)
    else:
        # For txt files, use markdown loader
        loader = UnstructuredMarkdownLoader(tmp_path)
    
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")
    
    # Clean up temp file
    os.unlink(tmp_path)
    return documents


def create_retriever(documents, llm_model):
    """Create a retriever from documents"""
    logger.info(f"Creating retriever from {len(documents)} documents")
    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")

    # Check if embedding model is initialized, if not use HuggingFaceEmbeddings as fallback
    if st.session_state.embedding_model is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.warning("Embedding model not initialized. Using HuggingFace embeddings as fallback.")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        embeddings = st.session_state.embedding_model
        logger.info(f"Using Ollama embeddings model")

    # Create vector store and retriever
    logger.info("Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    logger.info("FAISS vectorstore created successfully")

    # Add compression for better retrieval
    logger.info("Adding LLM chain extractor for context compression")
    compressor = LLMChainExtractor.from_llm(llm_model)
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=compressor
    )
    logger.info("Retriever setup complete")

    return compression_retriever


def timetable_to_markdown(df):
    """Convert DataFrame to markdown table"""
    if df is None or df.empty:
        return "No timetable data available."
    return df.to_markdown(index=False)

def extract_timetable_updates(response_text, current_df):
    """Extract timetable updates from LLM response"""
    logger.info("Attempting to extract timetable updates from response")
    # First, check for CSV content
    csv_match = re.search(r'```csv\n(.*?)\n```', response_text, re.DOTALL)
    if csv_match:
        try:
            csv_content = csv_match.group(1)
            import io
            new_df = pd.read_csv(io.StringIO(csv_content))
            logger.info(f"Successfully extracted CSV table with {len(new_df)} rows")
            return new_df, True
        except Exception as e:
            logger.error(f"Failed to parse CSV: {str(e)}")
            st.error(f"Failed to parse CSV: {str(e)}")
    
    # Try to extract markdown table
    table_match = re.search(r'\|.*\|[\r\n]+\|[- |]+\|([\s\S]*?)(?:\n\n|\Z)', response_text)
    if table_match:
        try:
            table_content = table_match.group(0)
            import io
            # Add the header line back
            new_df = pd.read_table(io.StringIO(table_content), sep='|', skipinitialspace=True)
            # Clean up column names
            new_df = new_df.iloc[:, 1:-1]  # Remove first and last columns (empty)
            new_df.columns = [col.strip() for col in new_df.columns]
            logger.info(f"Successfully extracted markdown table with {len(new_df)} rows")
            return new_df, True
        except Exception as e:
            logger.error(f"Failed to parse markdown table: {str(e)}")
            st.error(f"Failed to parse markdown table: {str(e)}")
    
    logger.info("No valid timetable format found in response")
    return current_df, False

# ----- Main App Function ----- #
def main():
    logger.info("Starting Timetable Management System")

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rules_retriever" not in st.session_state:
        st.session_state.rules_retriever = None
    
    if "org_info_retriever" not in st.session_state:
        st.session_state.org_info_retriever = None
    
    if "timetable_df" not in st.session_state:
        st.session_state.timetable_df = None
    
    if "llm" not in st.session_state:
        st.session_state.llm = None
        
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None
    
    # Create sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # LLM Provider selection
        st.subheader("LLM Provider")
        llm_provider = st.radio(
            "Select LLM Provider",
            options=["Ollama (Local)", "Groq (Cloud)"],
            index=0,
            help="Choose between local Ollama models or cloud Groq models"
        )
        
        # Model selection based on provider
        if llm_provider == "Ollama (Local)":
            selected_model = st.selectbox(
                "Select Ollama Model", 
                options=AVAILABLE_OLLAMA_MODELS,
                index=0,
                help="Choose the local Ollama model to use"
            )
        else:  # Groq
            selected_model = st.selectbox(
                "Select Groq Model", 
                options=AVAILABLE_GROQ_MODELS,
                index=0,
                help="Choose the Groq cloud model to use"
            )

        # Initialize model button
        if st.button("Initialize/Change Model"):
            with st.spinner(f"Initializing {selected_model}..."):
                try:
                    logger.info(f"Initializing LLM model: {selected_model} with provider {llm_provider}")
                    
                    if llm_provider == "Ollama (Local)":
                        # Initialize Ollama model
                        st.session_state.llm = OllamaLLM(model=selected_model, num_thread=4, temperature=0.3)
                    else:
                        # Initialize Groq model
                        # No need to provide API key as it's already initialized in env
                        st.session_state.llm = ChatGroq(model_name=selected_model, temperature=0.3)
                    
                    logger.info(f"Model {selected_model} initialized successfully")
                    st.success(f"{selected_model} initialized successfully!")
                except Exception as e:
                    logger.error(f"Failed to initialize model: {str(e)}")
                    st.error(f"Failed to initialize model: {str(e)}")
                    if llm_provider == "Ollama (Local)":
                        st.info("Make sure Ollama is running locally and the selected model is available.")
                    else:
                        st.info("Make sure your Groq API key is properly set in the environment.")

        st.divider()
        
        # Add a section for embedding model
        st.subheader("Embedding Model")
        
        embedding_model_name = st.text_input(
            "Embedding Model Name", 
            value=DEFAULT_EMBEDDING_MODEL,
            help="Enter the name of an Ollama embedding model (e.g., nomic-embed-text)"
        )
        
        if st.button("Initialize Embedding Model"):
            with st.spinner(f"Initializing embedding model {embedding_model_name}..."):
                try:
                    logger.info(f"Initializing embedding model: {embedding_model_name}")
                    # Pull the embedding model first to make sure it's available
                    import subprocess
                    subprocess.run(["ollama", "pull", embedding_model_name], check=True)
                    
                    # Initialize the embedding model
                    st.session_state.embedding_model = OllamaEmbeddings(model=embedding_model_name)
                    logger.info(f"Embedding model {embedding_model_name} initialized successfully")
                    st.success(f"Embedding model {embedding_model_name} initialized successfully!")
                except Exception as e:
                    logger.error(f"Failed to initialize embedding model: {str(e)}")
                    st.error(f"Failed to initialize embedding model: {str(e)}")
                    st.info("Make sure Ollama is running locally and the embedding model is available.")
        
        st.divider()
        
        # Document uploaders
        st.subheader("Upload Documents")
        
        # Rules document
        rules_file = st.file_uploader(
            "Upload Rules Document", 
            type=["md", "txt"],
            help="Upload a markdown file containing the rules for timetable creation"
        )
        
        if rules_file and st.session_state.llm:
            if st.session_state.embedding_model is None:
                st.warning("⚠️ Please initialize the embedding model before processing documents")
            elif st.button("Process Rules"):
                with st.spinner("Processing rules document..."):
                    logger.info("Processing rules document")
                    documents = load_document(rules_file, rules_file.name.split('.')[-1])
                    st.session_state.rules_retriever = create_retriever(documents, st.session_state.llm)
                    logger.info("Rules document processed successfully")
                    st.success("Rules document processed!")
        
        # Organization info document
        org_info_file = st.file_uploader(
            "Upload Organization Information", 
            type=["md", "txt"],
            help="Upload a markdown file containing information about courses, faculty, etc."
        )
        
        if org_info_file and st.session_state.llm:
            if st.session_state.embedding_model is None:
                st.warning("⚠️ Please initialize the embedding model before processing documents")
            elif st.button("Process Organization Info"):
                with st.spinner("Processing organization info..."):
                    logger.info("Processing organization info document")
                    documents = load_document(org_info_file, org_info_file.name.split('.')[-1])
                    st.session_state.org_info_retriever = create_retriever(documents, st.session_state.llm)
                    logger.info("Organization info processed successfully")
                    st.success("Organization info processed!")
        
        # Timetable structure file
        timetable_file = st.file_uploader(
            "Upload Timetable Structure", 
            type=["csv"],
            help="Upload a CSV file containing the current timetable or template"
        )
        
        if timetable_file:
            if st.button("Load Timetable"):
                with st.spinner("Loading timetable..."):
                    try:
                        logger.info("Loading timetable from CSV")
                        df = pd.read_csv(timetable_file)
                        st.session_state.timetable_df = df
                        logger.info(f"Timetable loaded with {len(df)} rows and {len(df.columns)} columns")
                        st.success("Timetable loaded successfully!")
                    except Exception as e:
                        logger.error(f"Failed to load timetable: {str(e)}")
                        st.error(f"Failed to load timetable: {str(e)}")
        
        # System status
        st.subheader("System Status")
        status_cols = st.columns(2)
        with status_cols[0]:
            st.write("LLM:")
            st.write("Embeddings:")
            st.write("Rules:")
            st.write("Org Info:")
            st.write("Timetable:")
        
        with status_cols[1]:
            llm_status = "✅" if st.session_state.llm else "❌"
            embed_status = "✅" if st.session_state.embedding_model else "❌"
            rules_status = "✅" if st.session_state.rules_retriever else "❌"
            org_status = "✅" if st.session_state.org_info_retriever else "❌"
            tt_status = "✅" if st.session_state.timetable_df is not None else "❌"
            
            llm_name = selected_model if st.session_state.llm else "Not initialized"
            provider_indicator = f"({llm_provider.split()[0]})" if st.session_state.llm else ""
            
            st.write(f"{llm_status} {llm_name} {provider_indicator}")
            st.write(f"{embed_status} {embedding_model_name if st.session_state.embedding_model else 'Not initialized'}")
            st.write(f"{rules_status}")
            st.write(f"{org_status}")
            st.write(f"{tt_status}")
        
        # Clear all button
        if st.button("Reset System", type="primary"):
            logger.info("Resetting system state")
            for key in ["messages", "rules_retriever", "org_info_retriever", "timetable_df", "llm", "embedding_model"]:
                if key in st.session_state:
                    st.session_state[key] = None if key != "messages" else []
                    logger.info(f"Reset {key}")
            logger.info("System reset complete, triggering rerun")
            st.experimental_rerun()

    # Main content area
    st.title("📅 Interactive Timetable Management System")
    
    # Check if all components are ready
    all_ready = all([
        st.session_state.llm,
        st.session_state.embedding_model,
        st.session_state.rules_retriever,
        st.session_state.org_info_retriever,
        st.session_state.timetable_df is not None
    ])

    if not all_ready:
        st.warning("⚠️ Please initialize all components in the sidebar before proceeding.")
        
        # Show component that needs attention
        if not st.session_state.llm:
            st.info("➡️ Initialize the LLM model first")
        elif not st.session_state.embedding_model:
            st.info("➡️ Initialize the embedding model")
        elif not st.session_state.rules_retriever:
            st.info("➡️ Upload and process the rules document")
        elif not st.session_state.org_info_retriever:
            st.info("➡️ Upload and process the organization information")
        elif st.session_state.timetable_df is None:
            st.info("➡️ Upload and load the timetable structure")
    else:
        # Display the current timetable
        with st.expander("Current Timetable", expanded=True):
            st.dataframe(st.session_state.timetable_df, use_container_width=True)
            
            # Add download button
            csv = st.session_state.timetable_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Timetable as CSV",
                csv,
                "timetable.csv",
                "text/csv",
                key='download-csv'
            )
        
        # Chat interface
        st.subheader("💬 Timetable Assistant")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about creating or modifying the timetable..."):
            logger.info(f"User input: {prompt[:50]}...")
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response_placeholder.markdown("Thinking...")  # Show initial loading message
                
                try:
                    with st.spinner("Generating response..."):
                        # Get context from retrievers
                        if st.session_state.rules_retriever:
                            logger.info("Retrieving relevant rules")
                            rules_docs = st.session_state.rules_retriever.get_relevant_documents(prompt)
                            rules_context = "\n".join([doc.page_content for doc in rules_docs])
                            logger.info(f"Retrieved {len(rules_docs)} rule documents")
                        else:
                            rules_context = "No rules information available."
                            logger.warning("No rules retriever available")
                        
                        if st.session_state.org_info_retriever:
                            logger.info("Retrieving relevant organization info")
                            org_docs = st.session_state.org_info_retriever.get_relevant_documents(prompt)
                            org_context = "\n".join([doc.page_content for doc in org_docs])
                            logger.info(f"Retrieved {len(org_docs)} organization documents")
                        else:
                            org_context = "No organization information available."
                            logger.warning("No organization info retriever available")
                        
                        # Create system prompt
                        system_prompt = f"""
                        You are an AI assistant specialized in creating and modifying academic timetables.
                        Follow these rules and constraints when working with timetables:
                        
                        RULES AND CONSTRAINTS:
                        {rules_context}
                        
                        ORGANIZATION INFORMATION:
                        {org_context}
                        
                        When modifying a timetable:
                        1. Analyze the current timetable carefully
                        2. Make changes that adhere to all rules and constraints
                        3. Ensure no scheduling conflicts are created
                        4. Maintain proper course allocations based on faculty expertise
                        5. Consider room capacities and equipment needs
                        
                        Always explain your reasoning before providing the modified timetable.
                        """
                        
                        # Create timetable context
                        current_timetable = f"CURRENT TIMETABLE:\n{timetable_to_markdown(st.session_state.timetable_df)}"
                        
                        # Create messages
                        messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=f"{current_timetable}\n\nUser request: {prompt}")
                        ]
                        
                        # Log the request
                        logger.info("Starting streaming response from LLM")
                        
                        # Stream the response
                        response = ""
                        for chunk in st.session_state.llm.stream(messages):
                            if hasattr(chunk, 'content'):
                                response += chunk.content
                                # Update placeholder with current response
                                response_placeholder.markdown(response + "▌")  # Add cursor for better UX
                                
                                # Log every few chunks
                                if len(response) % 100 == 0:
                                    logger.debug(f"Received {len(response)} characters so far")
                        
                        # Final response display without cursor
                        response_placeholder.markdown(response)
                        logger.info("Completed streaming response")
                        
                        # Log detailed info about the interaction
                        log_response_generation(messages, response)
                        
                        # Check for timetable updates in the response
                        new_df, was_updated = extract_timetable_updates(response, st.session_state.timetable_df)
                        
                        if was_updated:
                            # Update the timetable
                            st.session_state.timetable_df = new_df
                            logger.info(f"Timetable updated with {len(new_df)} rows")
                            response += "\n\n*Timetable has been updated successfully!*"
                            response_placeholder.markdown(response)
                        
                        # Add assistant message to history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # If timetable was updated, show the new version
                        if was_updated:
                            st.success("Timetable updated successfully!")
                            st.dataframe(st.session_state.timetable_df, use_container_width=True)
                
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    logger.error(f"ERROR: {error_msg}")
                    logger.exception("Full exception details:")
                    response_placeholder.markdown(f"❌ {error_msg}")
                    st.error(error_msg)

# Run the app
if __name__ == "__main__":
    logger.info("Application starting")
    main()
