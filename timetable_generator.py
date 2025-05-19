import os
import streamlit as st
import logging
import sys

# First, add the import statement at the top of the file with other imports
from langchain.memory import ConversationBufferMemory

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
from langchain_google_genai import ChatGoogleGenerativeAI
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
AVAILABLE_GEMINI_MODELS = ["gemini-2.5-flash-preview-04-17", "gemini-2.0-flash", "gemini-2.0-flash-live-001"]
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# Add this function to your code (e.g., near the top with other helper functions)
def make_column_names_unique(df):
    """Ensure DataFrame column names are absolutely unique"""
    # Get current columns as a list
    current_cols = list(df.columns)
    
    # Create a new list for column names
    new_cols = []
    used_names = set()
    
    # Process each column
    for i, col_name in enumerate(current_cols):
        # Convert NaN to string
        if pd.isna(col_name):
            col_name = f"Column_{i}"
        else:
            col_name = str(col_name)
        
        # Ensure uniqueness
        base_name = col_name
        counter = 1
        while col_name in used_names:
            col_name = f"{base_name}_{counter}"
            counter += 1
        
        # Add to our tracking structures
        new_cols.append(col_name)
        used_names.add(col_name)
    
    # Set the new column names
    df_copy = df.copy()
    df_copy.columns = new_cols
    return df_copy

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
            
            # Use Python's built-in csv module instead of pandas for initial parsing
            import csv
            import io
            
            # Parse the CSV content
            csv_reader = csv.reader(io.StringIO(csv_content))
            rows = list(csv_reader)
            
            if not rows:
                logger.error("CSV content is empty")
                return current_df, False
            
            # Get headers and data rows
            headers = rows[0]
            data_rows = rows[1:]
            
            # Create a DataFrame with a simple numeric index for columns
            import numpy as np
            temp_df = pd.DataFrame(data_rows)
            
            # Now set the original headers back (with duplicates)
            # We need to ensure we have the right number of columns
            temp_headers = headers[:len(temp_df.columns)]
            # Add any missing headers if the parsed data has more columns than headers
            if len(temp_df.columns) > len(headers):
                temp_headers += [f"Column{i}" for i in range(len(headers), len(temp_df.columns))]
            
            temp_df.columns = temp_headers
                
            logger.info(f"Successfully extracted CSV table with {len(temp_df)} rows")
            temp_df = make_column_names_unique(temp_df)
            return temp_df, True
            
        except Exception as e:
            logger.error(f"Failed to parse CSV: {str(e)}")
            st.error(f"Failed to parse CSV: {str(e)}")
    
    # Try to extract markdown table
    table_match = re.search(r'\|.*\|[\r\n]+\|[- |]+\|([\s\S]*?)(?:\n\n|\Z)', response_text)
    if table_match:
        try:
            table_content = table_match.group(0)
            
            # Parse markdown table manually
            lines = table_content.strip().split('\n')
            if len(lines) < 3:  # Need at least header, separator, and one data row
                logger.error("Markdown table has insufficient lines")
                return current_df, False
            
            # Extract headers from the first line
            header_line = lines[0].strip()
            headers = [col.strip() for col in header_line.split('|')]
            headers = [col for col in headers if col]  # Remove empty strings
            
            # Skip the separator line (line[1])
            
            # Parse data rows
            data_rows = []
            for line in lines[2:]:
                if not line.strip():
                    continue
                cells = [cell.strip() for cell in line.split('|')]
                cells = [cell for cell in cells if cell != '']  # Remove empty cells from beginning/end
                data_rows.append(cells)
            
            # Create DataFrame manually
            # For columns, use numeric indices initially
            temp_df = pd.DataFrame(data_rows)
            
            # Set the original headers back (with duplicates)
            # We need to ensure we have the right number of columns
            temp_headers = headers[:len(temp_df.columns)]
            # Add any missing headers if the parsed data has more columns than headers
            if len(temp_df.columns) > len(headers):
                temp_headers += [f"Column{i}" for i in range(len(headers), len(temp_df.columns))]
            
            temp_df.columns = temp_headers
            temp_df = make_column_names_unique(temp_df)  # Make column names unique
            
            logger.info(f"Successfully extracted markdown table with {len(temp_df)} rows")
            
            return temp_df, True
            
        except Exception as e:
            logger.error(f"Failed to parse markdown table: {str(e)}")
            st.error(f"Failed to parse markdown table: {str(e)}")
    
    logger.info("No valid timetable format found in response")
    return current_df, False

def validate_dataframe(df):
    """Validates dataframe for use in the app, fixes issues if possible"""
    # Check for duplicate column names
    if df.columns.duplicated().any():
        # Create unique column names
        cols = list(df.columns)
        for i in range(len(cols)):
            if pd.isna(cols[i]):
                cols[i] = f"Column_{i}"
            elif cols[i] in cols[:i]:
                count = cols[:i].count(cols[i])
                cols[i] = f"{cols[i]}_{count+1}"
        
        df.columns = cols
        logger.info("Fixed duplicate column names")
    
    return df

# ----- Main App Function ----- #
# First, add the import statement at the top of the file with other imports
from langchain.memory import ConversationBufferMemory

# In the main() function, update the session state initialization
def main():
    logger.info("Starting Timetable Management System")

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Add ConversationBufferMemory to session state
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    
    # Rest of the existing initialization code
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
            options=["Ollama (Local)", "Groq (Cloud)", "Gemini (Cloud)"],
            index=0,
            help="Choose between local Ollama models, cloud Groq models, or Google Gemini models"
        )
        
        # Model selection based on provider
        if llm_provider == "Ollama (Local)":
            selected_model = st.selectbox(
                "Select Ollama Model", 
                options=AVAILABLE_OLLAMA_MODELS,
                index=0,
                help="Choose the local Ollama model to use"
            )
        elif llm_provider == "Groq (Cloud)":
            selected_model = st.selectbox(
                "Select Groq Model", 
                options=AVAILABLE_GROQ_MODELS,
                index=0,
                help="Choose the Groq cloud model to use"
            )
        else:  # Gemini
            selected_model = st.selectbox(
                "Select Gemini Model", 
                options=AVAILABLE_GEMINI_MODELS,
                index=0,
                help="Choose the Google Gemini model to use"
            )

        # Initialize model button
        if st.button("Initialize/Change Model"):
            with st.spinner(f"Initializing {selected_model}..."):
                try:
                    logger.info(f"Initializing LLM model: {selected_model} with provider {llm_provider}")
                    
                    if llm_provider == "Ollama (Local)":
                        # Initialize Ollama model
                        st.session_state.llm = OllamaLLM(model=selected_model, num_thread=4, temperature=0.3)
                    elif llm_provider == "Groq (Cloud)":
                        # Initialize Groq model
                        # No need to provide API key as it's already initialized in env
                        st.session_state.llm = ChatGroq(model_name=selected_model, temperature=0.3)
                    else:
                        # Initialize Gemini model
                        # No need to provide API key as it's already in env
                        st.session_state.llm = ChatGoogleGenerativeAI(model=selected_model, temperature=0.3)
                    
                    logger.info(f"Model {selected_model} initialized successfully")
                    st.success(f"{selected_model} initialized successfully!")
                except Exception as e:
                    logger.error(f"Failed to initialize model: {str(e)}")
                    st.error(f"Failed to initialize model: {str(e)}")
                    if llm_provider == "Ollama (Local)":
                        st.info("Make sure Ollama is running locally and the selected model is available.")
                    elif llm_provider == "Groq (Cloud)":
                        st.info("Make sure your Groq API key is properly set in the environment.")
                    else:
                        st.info("Make sure your Google API key is properly set in the environment.")

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
                        # Ensure column names are unique
                        df = make_column_names_unique(df)
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
            st.rerun()

        # Add this to the sidebar after the "Clear all" button
        st.sidebar.divider()
        st.sidebar.subheader("Memory Management")

        # Option to view the current memory
        if st.sidebar.button("View Conversation History"):
            memory_vars = st.session_state.memory.load_memory_variables({})
            if "history" in memory_vars and memory_vars["history"]:
                st.sidebar.json(memory_vars)
            else:
                st.sidebar.info("No conversation history available.")

        # Option to clear just the memory
        if st.sidebar.button("Clear Conversation Memory"):
            if "memory" in st.session_state:
                st.session_state.memory.clear()
                st.sidebar.success("Conversation memory cleared!")
                logger.info("Conversation memory cleared")

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
            # Before displaying the dataframe, ensure column names are unique
            # Right before displaying the dataframe (around line 554)
            if st.session_state.timetable_df is not None:
                # Make a copy with unique column names just for display
                display_df = make_column_names_unique(st.session_state.timetable_df)
                st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("No timetable data available.")
            
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
        
        # Now, modify the chat response generation part to use memory
        # In the main() function, update the session state initialization
        # In the chat response generation part:
        if prompt := st.chat_input("Ask about creating or modifying the timetable..."):
            logger.info(f"User input: {prompt[:50]}...")
            # Add user message to UI messages (for display purposes)
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
                    
                        # Create system prompt (keep your existing dynamic system prompt)
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
                    
                        # Load memory variables to get chat history
                        memory_variables = st.session_state.memory.load_memory_variables({})
                    
                        # Create messages
                        # Start with system message
                        messages = [SystemMessage(content=system_prompt)]
                    
                        # Add memory messages if they exist - for context
                        chat_history = memory_variables.get("history", [])
                        if chat_history:
                            messages.extend(chat_history)
                    
                        # Add current user query with timetable context
                        messages.append(HumanMessage(content=f"{current_timetable}\n\nUser request: {prompt}"))
                    
                        # Log the request
                        logger.info("Starting streaming response from LLM")
                    
                        # Stream the response (keep your existing streaming code)
                        response = ""
                        for chunk in st.session_state.llm.stream(messages):
                            if hasattr(chunk, 'content'):
                                response += chunk.content
                                # Update placeholder with current response
                                response_placeholder.markdown(response + "▌")
                            
                                # Log every few chunks
                                if len(response) % 100 == 0:
                                    logger.debug(f"Received {len(response)} characters so far")
                    
                        # Final response display without cursor
                        response_placeholder.markdown(response)
                        logger.info("Completed streaming response")
                    
                        # Log detailed info about the interaction
                        log_response_generation(messages, response)
                    
                        # Save the conversation to memory
                        st.session_state.memory.save_context({"input": prompt}, {"output": response})
                    
                        # Add assistant message to display history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    
                        # Check for timetable updates in the response
                        new_df, was_updated = extract_timetable_updates(response, st.session_state.timetable_df)
                    
                        # Continue with existing code for handling timetable updates
                except Exception as e:
                    # Keep your existing exception handling
                    pass

# Run the app
if __name__ == "__main__":
    logger.info("Application starting")
    main()
