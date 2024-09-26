import os
import tempfile
import streamlit as st
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from couchbase_haystack import CouchbaseDocumentStore, CouchbaseEmbeddingRetriever, CouchbasePasswordAuthenticator, CouchbaseClusterOptions

def check_environment_variable(variable_name):
    """Check if environment variable is set"""
    if variable_name not in os.environ:
        st.error(f"{variable_name} environment variable is not set. Please add it to the secrets.toml file")
        st.stop()

def save_to_vector_store(uploaded_file, indexing_pipeline):
    """Process the PDF & store it in Couchbase Vector Store"""
    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        st.info(f"Temp directory created at: {temp_dir.name} jj {uploaded_file.name}")
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        result = indexing_pipeline.run({"converter": {"sources": [temp_file_path]}})
        
        st.info(f"PDF loaded into vector store: {result['writer']['documents_written']} documents indexed")

@st.cache_resource(show_spinner="Connecting to Vector Store")
def get_document_store():
    """Return the Couchbase document store"""
    return CouchbaseDocumentStore(
        cluster_connection_string=Secret.from_token(os.getenv("DB_CONN_STR")),
        authenticator=CouchbasePasswordAuthenticator(
            username=Secret.from_token(os.getenv("DB_USERNAME")),
            password=Secret.from_token(os.getenv("DB_PASSWORD"))
        ),
        cluster_options=CouchbaseClusterOptions(profile='wan_development'),
        bucket=os.getenv("DB_BUCKET"),
        scope=os.getenv("DB_SCOPE"),
        collection=os.getenv("DB_COLLECTION"),
        vector_search_index=os.getenv("INDEX_NAME"),
    )


if __name__ == "__main__":
    OPENAI_API_KEY = Secret.from_token(os.getenv("OPENAI_API_KEY"))
    st.set_page_config(
        page_title="Chat with your PDF using Haystack, Couchbase & Gemini Pro",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    # Load and check environment variables
    env_vars = ["DB_CONN_STR", "DB_USERNAME", "DB_PASSWORD", "DB_BUCKET", "DB_SCOPE", "DB_COLLECTION", "INDEX_NAME", "OPENAI_API_KEY"]
    for var in env_vars:
        check_environment_variable(var)

    # Initialize document store
    document_store = get_document_store()

    # Create indexing pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("converter", PyPDFToDocument())
    indexing_pipeline.add_component("cleaner", DocumentCleaner())
    indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=250, split_overlap=30))
    indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    indexing_pipeline.connect("converter.documents", "cleaner.documents")
    indexing_pipeline.connect("cleaner.documents", "splitter.documents")
    indexing_pipeline.connect("splitter.documents", "embedder.documents")
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    # Create RAG pipeline
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("query_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
    rag_pipeline.add_component("retriever", CouchbaseEmbeddingRetriever(document_store=document_store))
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
            model="gpt-4o-",
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
    st.markdown("Answers with [Couchbase logo](https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png) are generated using *RAG* while 🤖 are generated by pure *LLM (Gemini)*")

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
            * one using pure LLM - Gemini (🤖). 
            """)

        st.markdown("For RAG, we are using [Haystack](https://haystack.deepset.ai/), [Couchbase Vector Search](https://couchbase.com/) & [Gemini](https://gemini.google.com/). We fetch parts of the PDF relevant to the question using Vector search & add it as the context to the LLM. The LLM is instructed to answer based on the context from the Vector Store.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hi, I'm a chatbot who can chat with the PDF. How can I help you?", "avatar": "🤖"})

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a question based on the PDF"):
        st.chat_message("user").markdown(question)
        st.session_state.messages.append({"role": "user", "content": question, "avatar": "👤"})

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