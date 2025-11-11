import os
import tempfile
import streamlit as st
from datetime import timedelta
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from couchbase.n1ql import QueryScanConsistency
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions
from couchbase.exceptions import QueryIndexAlreadyExistsException, ScopeAlreadyExistsException, CollectionAlreadyExistsException

# Import CouchbaseQueryDocumentStore for GSI-based vector search with BHIVe support
from couchbase_haystack import (
    CouchbaseQueryDocumentStore,
    CouchbaseQueryEmbeddingRetriever,
    CouchbasePasswordAuthenticator,
    CouchbaseClusterOptions,
    QueryVectorSearchType,
    QueryVectorSearchSimilarity,
    CouchbaseQueryOptions
)

def check_environment_variable(variable_name):
    """Check if environment variable is set"""
    if variable_name not in os.environ:
        st.error(f"{variable_name} environment variable is not set. Please add it to the secrets.toml file")
        st.stop()

def create_scope_if_not_exists(collection_manager, scope_name):
    """Create scope if it doesn't exist"""
    try:
        scopes = collection_manager.get_all_scopes()
        scope_names = [scope.name for scope in scopes]
        
        if scope_name not in scope_names:
            collection_manager.create_scope(scope_name)
            st.info(f"Scope '{scope_name}' created successfully")
            return True
        return False
    except ScopeAlreadyExistsException:
        return False
    except Exception as e:
        st.warning(f"Could not create scope '{scope_name}': {str(e)}")
        return False

def create_collection_if_not_exists(collection_manager, scope_name, collection_name):
    """Create collection if it doesn't exist"""
    try:
        scopes = collection_manager.get_all_scopes()
        
        for scope in scopes:
            if scope.name == scope_name:
                collection_names = [collection.name for collection in scope.collections]
                
                if collection_name not in collection_names:
                    collection_manager.create_collection(scope_name=scope_name, collection_name=collection_name)
                    st.info(f"Collection '{collection_name}' created in scope '{scope_name}'")
                    return True
                return False
        
        st.error(f"Scope '{scope_name}' does not exist, cannot create collection")
        return False
    except CollectionAlreadyExistsException:
        return False
    except Exception as e:
        st.warning(f"Could not create collection '{collection_name}': {str(e)}")
        return False

def create_vector_index_if_not_exists(cluster, bucket_name, scope_name, collection_name, similarity="DOT", dimension=1536):
    """Create Hyperscale vector index if it doesn't exist"""
    index_name = f"idx_{collection_name}_vector"
    
    try:
        # Check if index exists by trying to query system:indexes
        check_query = f"""
        SELECT COUNT(*) as count FROM system:indexes 
        WHERE name = '{index_name}' 
        AND keyspace_id = '{collection_name}'
        AND bucket_id = '{bucket_name}'
        AND scope_id = '{scope_name}'
        """
        
        result = cluster.query(check_query)
        rows = list(result)
        
        if rows and rows[0].get('count', 0) > 0:
            st.success(f"Vector index '{index_name}' already exists!")
            return False  # Index already exists
        
        # Count documents in collection first
        count_query = f"SELECT COUNT(*) as doc_count FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`"
        count_result = cluster.query(count_query)
        count_rows = list(count_result)
        doc_count = count_rows[0].get('doc_count', 0) if count_rows else 0
        
        if doc_count == 0:
            st.error("No documents found in collection. Please upload a PDF first before creating the vector index.")
            return False
        
        # Create the Hyperscale vector index
        create_index_query = f"""
        CREATE VECTOR INDEX {index_name}
        ON `{collection_name}`(embedding VECTOR) 
        WITH {{
          "dimension": {dimension},
          "similarity": "{similarity}"
        }}
        """
        
        # Set query context to the bucket.scope, then run the create index
        cluster.bucket(bucket_name).scope(scope_name).query(create_index_query).execute()
        st.success(f"Vector index '{index_name}' created successfully!")
        st.info("Note: The vector index may take a few moments to become fully available")
        return True
        
    except QueryIndexAlreadyExistsException:
        st.info(f"Vector index '{index_name}' already exists")
        return False
    except Exception as e:
        error_msg = str(e)
        
        # If the error is about index already existing, that's fine
        if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
            st.info(f"Vector index '{index_name}' already exists")
            return False
        # If it's about no documents, warn but continue
        elif "no documents" in error_msg.lower() or "training" in error_msg.lower():
            st.warning(f"Vector index requires documents for training. Please upload a PDF first.")
            return False
        else:
            st.error(f"Could not create vector index: {error_msg}")
            st.info("You may need to create the vector index manually. See README for instructions.")
            return False

def setup_couchbase_resources(cluster_connection_string, username, password, bucket_name, scope_name, collection_name) -> Cluster:
    """Setup Couchbase resources: scope, collection, and optionally vector index"""
    try:
        # Connect to cluster for management operations
        auth = PasswordAuthenticator(username, password)
        cluster = Cluster(cluster_connection_string, ClusterOptions(auth))
        cluster.wait_until_ready(timedelta(seconds=10))
        
        bucket = cluster.bucket(bucket_name)
        collection_manager = bucket.collections()
        
        # Create scope if needed
        scope_created = create_scope_if_not_exists(collection_manager, scope_name)
        
        # Create collection if needed
        collection_created = create_collection_if_not_exists(collection_manager, scope_name, collection_name)
        
        # If we just created scope or collection, wait a bit for them to be ready
        if scope_created or collection_created:
            import time
            time.sleep(2)

        return cluster
        
    except Exception as e:
        st.error(f"Error during Couchbase setup: {str(e)}")
        st.info("Continuing with existing resources...")

def save_to_vector_store(uploaded_file, indexing_pipeline) -> Cluster:
    """Process the PDF & store it in Couchbase Vector Store"""
    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        result = indexing_pipeline.run({"converter": {"sources": [temp_file_path]}})
        
        st.info(f"PDF loaded into vector store: {result['writer']['documents_written']} documents indexed")
        
        # Create the scope and collection
        cluster = setup_couchbase_resources(
            cluster_connection_string=os.getenv("DB_CONN_STR"),
            username=os.getenv("DB_USERNAME"),
            password=os.getenv("DB_PASSWORD"),
            bucket_name=os.getenv("DB_BUCKET"),
            scope_name=os.getenv("DB_SCOPE"),
            collection_name=os.getenv("DB_COLLECTION"),
        )

        # Create the vector index
        create_vector_index_if_not_exists(
            cluster=cluster,
            bucket_name=os.getenv("DB_BUCKET"),
            scope_name=os.getenv("DB_SCOPE"),
            collection_name=os.getenv("DB_COLLECTION"),
        )
        
        return cluster

@st.cache_resource(show_spinner="Connecting to Vector Store")
def get_document_store():
    """Return the Couchbase document store using CouchbaseQueryDocumentStore."""
    return CouchbaseQueryDocumentStore(
        cluster_connection_string=Secret.from_env_var("DB_CONN_STR"),
        authenticator=CouchbasePasswordAuthenticator(
            username=Secret.from_env_var("DB_USERNAME"),
            password=Secret.from_env_var("DB_PASSWORD")
        ),
        cluster_options=CouchbaseClusterOptions(profile='wan_development'),
        bucket=os.getenv("DB_BUCKET"),
        scope=os.getenv("DB_SCOPE"),
        collection=os.getenv("DB_COLLECTION"),
        search_type=QueryVectorSearchType.ANN,
        similarity=QueryVectorSearchSimilarity.DOT,
        nprobes=10,
        query_options=CouchbaseQueryOptions(
            timeout=timedelta(seconds=60),
            scan_consistency=QueryScanConsistency.NOT_BOUNDED
        )
    )


if __name__ == "__main__":
    OPENAI_API_KEY = Secret.from_env_var("OPENAI_API_KEY")
    st.set_page_config(
        page_title="Chat with your PDF using Haystack, Couchbase & Gemini Pro",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    # Load and check environment variables
    env_vars = ["DB_CONN_STR", "DB_USERNAME", "DB_PASSWORD", "DB_BUCKET", "DB_SCOPE", "DB_COLLECTION", "OPENAI_API_KEY"]
    for var in env_vars:
        check_environment_variable(var)

    # Setup Couchbase resources (scope and collection only, index created after document upload)
    with st.spinner("Setting up Couchbase resources..."):
        setup_couchbase_resources(
            cluster_connection_string=os.getenv("DB_CONN_STR"),
            username=os.getenv("DB_USERNAME"),
            password=os.getenv("DB_PASSWORD"),
            bucket_name=os.getenv("DB_BUCKET"),
            scope_name=os.getenv("DB_SCOPE"),
            collection_name=os.getenv("DB_COLLECTION"),
        )

    # Initialize document store
    document_store = get_document_store()

    # Create indexing pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("converter", PyPDFToDocument())
    indexing_pipeline.add_component("cleaner", DocumentCleaner())
    indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=250, split_overlap=50))
    indexing_pipeline.add_component("embedder", OpenAIDocumentEmbedder())
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    indexing_pipeline.connect("converter.documents", "cleaner.documents")
    indexing_pipeline.connect("cleaner.documents", "splitter.documents")
    indexing_pipeline.connect("splitter.documents", "embedder.documents")
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    # Create RAG pipeline
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("query_embedder", OpenAITextEmbedder())
    rag_pipeline.add_component("retriever", CouchbaseQueryEmbeddingRetriever(document_store=document_store))
    rag_pipeline.add_component("prompt_builder", PromptBuilder(template="""
    You are a helpful bot. If you cannot answer based on the context provided, respond with a generic answer. Answer the question as truthfully as possible using the context below:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    Question: {{question}}
    """))
    rag_pipeline.add_component(
        "llm",
        OpenAIGenerator(
            api_key=OPENAI_API_KEY,
            model="gpt-4o",
        ),
    )
    rag_pipeline.add_component("answer_builder", AnswerBuilder())

    rag_pipeline.connect("query_embedder", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("llm.meta", "answer_builder.meta")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    # Frontend
    couchbase_logo = "https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png"

    st.title("Chat with PDF")
    st.markdown("Answers with [Couchbase logo](https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png) are generated using *RAG* while 🤖 are generated by pure *LLM (OpenAI)*")

    with st.sidebar:
        st.header("Upload your PDF")
        with st.form("upload pdf"):
            uploaded_file = st.file_uploader("Choose a PDF.", help="The document will be deleted after one hour of inactivity (TTL).", type="pdf")
            submitted = st.form_submit_button("Upload")
            if submitted:
                save_to_vector_store(uploaded_file, indexing_pipeline)

        st.subheader("How does it work?")
        st.markdown("""
            For each question, you will get two answers: 
            * one using RAG ([Couchbase logo](https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png))
            * one using pure LLM - OpenAI (🤖). 
            """)

        st.markdown("For RAG, we are using [Haystack](https://haystack.deepset.ai/), [Couchbase Vector Search](https://docs.couchbase.com/cloud/vector-index/hyperscale-vector-index.html) & [OpenAI](https://openai.com/). We fetch parts of the PDF relevant to the question using high-performance GSI vector search & add it as the context to the LLM. The LLM is instructed to answer based on the context from the Vector Store.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hi, I'm a chatbot who can chat with the PDF. How can I help you?", "avatar": "🤖"})

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a question based on the PDF"):
        st.chat_message("user").markdown(question)
        st.session_state.messages.append({"role": "user", "content": question, "avatar": "👤"})

        # Ensure vector index exists before first query (fallback safety check)
        if "index_check_done" not in st.session_state:
            with st.spinner("Ensuring vector index is ready..."):
                cluster = setup_couchbase_resources(
                    cluster_connection_string=os.getenv("DB_CONN_STR"),
                    username=os.getenv("DB_USERNAME"),
                    password=os.getenv("DB_PASSWORD"),
                    bucket_name=os.getenv("DB_BUCKET"),
                    scope_name=os.getenv("DB_SCOPE"),
                    collection_name=os.getenv("DB_COLLECTION"),
                )
                create_vector_index_if_not_exists(
                    cluster=cluster,
                    bucket_name=os.getenv("DB_BUCKET"),
                    scope_name=os.getenv("DB_SCOPE"),
                    collection_name=os.getenv("DB_COLLECTION"),
                )
                st.session_state.index_check_done = True

        # RAG response
        with st.chat_message("assistant", avatar=couchbase_logo):
            message_placeholder = st.empty()
            rag_result = rag_pipeline.run(
                {
                    "query_embedder": {"text": question},
                    "retriever": {"top_k": 3},
                    "prompt_builder": {"question": question},
                    "answer_builder": {"query": question},
                }
            )
            rag_response = rag_result["answer_builder"]["answers"][0].data
            message_placeholder.markdown(rag_response)
        st.session_state.messages.append({"role": "assistant", "content": rag_response, "avatar": couchbase_logo})

        # Pure LLM response
        with st.chat_message("ai", avatar="🤖"):
            message_placeholder_pure_llm = st.empty()
            pure_llm_result = rag_pipeline.run(
                {
                    "prompt_builder": {"question": question},
                    "llm": {},
                    "answer_builder": {"query": question},
                    "query_embedder": {"text": question}
                }
            )
            pure_llm_response = pure_llm_result["answer_builder"]["answers"][0].data
            message_placeholder_pure_llm.markdown(pure_llm_response)
        st.session_state.messages.append({"role": "assistant", "content": pure_llm_response, "avatar": "🤖"})