## RAG Demo using Couchbase, Streamlit, Haystack, and OpenAI

This is a demo app built to chat with your custom PDFs using the **Couchbase Vector Index** to augment the OpenAI results in a Retrieval-Augmented-Generation (RAG) model.

This demo uses **`CouchbaseQueryDocumentStore`** with GSI vector indexes, which offers:

- **High-performance vector search at massive scale** (billions of documents)
- **BHIVe (Hyperscale Vector Index)** support for pure vector search
- **Composite Vector Index** support for filtered vector search
- **SQL++ queries** for efficient vector retrieval
- **Low memory footprint** and concurrent updates & searches

### How does it work?

You can upload your PDFs with custom data & ask questions about the data in the chat box.
For each question, you will get two answers:

- one using RAG (Couchbase logo)
- one using pure LLM - OpenAI (🤖).

The RAG pipeline utilizes Haystack, Couchbase GSI Vector Index (Hyperscale or Composite Index), and OpenAI models. It fetches relevant parts of the PDF using vector search & adds them as context for the language model.


### Setup and Installation

- #### Install dependencies:

  `pip install -r requirements.txt`

- #### Set the environment secrets

  Copy the `secrets.example.toml` file in `.streamlit` folder and rename it to `secrets.toml` and replace the placeholders with the actual values for your environment

```
    DB_CONN_STR = "<couchbase_cluster_connection_string>"
    DB_USERNAME = "<couchbase_username>"
    DB_PASSWORD = "<couchbase_password>"
    DB_BUCKET = "<bucket_name>"
    DB_SCOPE = "<scope_name>"
    DB_COLLECTION = "<collection_name>"
    OPENAI_API_KEY = "<openai_api_key>"
```

- #### Create the Vector Index

  This demo uses Couchbase new Vector Indexes (introduced in version 8.0). With this version you have two new options:

  **Option 1: Hyperscale Vector Index** - Recommended in general due to its scalability, and used in this demo
  
  BHIVe is optimized for pure vector search at scale. It's perfect for chatbots, RAG applications, and scenarios where you need fast vector similarity search on large datasets.

  **Option 2: Composite Vector Index**
  
  Composite indexes combine vector fields with other scalar fields, allowing you to apply filters before vector search. This is useful when you need to narrow down results based on metadata (e.g., date, category, user_id) before performing vector similarity search.

  Learn more about these 2 vector indexes and when to use one over the other, [here](https://docs.couchbase.com/cloud/vector-index/use-vector-indexes.html).

- #### Key Components

    - Streamlit: Provides the web interface
    - Haystack: Orchestrates the RAG pipeline
    - Couchbase: Serves as the high-performance vector store
    - OpenAI: Supplies embeddings and the language model

### Vector Index Creation

You need to create a Hyperscale vector index on your collection **after** loading some documents (required for index training). Choose between BHIVe or Composite Index based on your use case. Whichever vector index (Hyperscale or Composite) you choose won't affect the functionality of this demo, though performance differences may occur.

#### Option 1: Hyperscale Vector Index - Recommended

Hyperscale is a dedicated vector index optimized for pure vector search at massive scale. Use this for the best performance in RAG applications. Refer [here](https://docs.couchbase.com/cloud/vector-index/hyperscale-vector-index.html) for detailed instructions.

**Creating a Hyperscale Index using SQL++:**

You can create the index using the Couchbase Query Workbench or programmatically:

```sql
CREATE VECTOR INDEX idx_pdf_hyperscale
ON `bucket_name`.`scope_name`.`collection_name`(embedding VECTOR) 
WITH {
  "dimension": 1536,           
  "similarity": "DOT"         
};
```

**Index Parameters Explained:**
- `dimension`: Must match your embedding model (1536 for OpenAI ada-002/ada-003, 768 for sentence-transformers)
- `similarity`: Must match the similarity metric in `CouchbaseQueryDocumentStore`
  - `DOT`: Dot product (recommended for OpenAI embeddings)

#### Option 2: Composite Vector Index

Composite indexes combine vector fields with other scalar fields. This is useful when you need to filter documents by metadata before performing vector search.

**Creating a Composite Index using SQL++:**

```sql
CREATE INDEX idx_pdf_composite 
ON `bucket_name`.`scope_name`.`collection_name`(embedding VECTOR) 
USING GSI 
WITH {
  "dimension": 1536,
  "similarity": "DOT"
};
```

#### Important Notes

1. **Index Creation Timing**: Hyperscale and Composite vector indexes require training data. Create the index **after** you've loaded the documents into your collection.

2. **Similarity Metric**: The `similarity` parameter in the index **must match** the `similarity` parameter in your `CouchbaseQueryDocumentStore` configuration.

3. **Dimension**: Must match your embedding model's output dimensions.

#### Verifying Your Index

After creating the index, verify it exists:

```sql
SELECT * FROM system:indexes 
WHERE name='idx_pdf_composite';
```

- #### Run the application

  `streamlit run chat_with_pdf.py`

### Implementation Details

This demo uses the following key components:

1. **CouchbaseQueryDocumentStore**: GSI-based document store with vector support
   - Configured with `QueryVectorSearchType.ANN` for fast approximate nearest neighbor search
   - Uses `QueryVectorSearchSimilarity.DOT` for dot product similarity (recommended for OpenAI embeddings)
   - Supports both BHIVe and Composite indexes

2. **CouchbaseQueryEmbeddingRetriever**: High-performance retriever for GSI vector search
   - Uses SQL++ queries with `APPROX_VECTOR_DISTANCE()` function for ANN search
   - Retrieves top-k most similar documents based on embedding similarity

3. **OpenAI Embeddings**: 
   - `text-embedding-ada-002` model with 1536 dimensions
   - Generates embeddings for both documents and queries

For more details on implementation, refer to the extensive code comments in `chat_with_pdf.py`.

### Additional Resources

- [Couchbase Vector Index Documentation](https://docs.couchbase.com/cloud/vector-index/vectors-and-indexes-overview.html)
- [Haystack Documentation](https://docs.haystack.deepset.ai/)
- [couchbase-haystack GitHub Repository](https://github.com/Couchbase-Ecosystem/couchbase-haystack)