# RAG Demo using Couchbase, Streamlit, Haystack, and OpenAI

This is a demo app built to chat with your custom PDFs using **Couchbase Vector Search** to augment the OpenAI results in a Retrieval-Augmented-Generation (RAG) model.

## Three Implementation Options

### Option 1: Search Service (FTS) Vector Search (`chat_with_pdf_with_fts.py`)

Uses **`CouchbaseSearchDocumentStore`** with Full Text Search (FTS) vector indexes, which offers:

- **Flexible vector search** with FTS capabilities
- **Rich text search** combined with vector similarity
- **Complex filtering** using FTS queries
- **Compatible with Couchbase 7.6+**
- **Ideal for hybrid search** scenarios combining full-text and vector search

### Option 2: Hyperscale Vector Index (Default - `chat_with_pdf.py`)

Uses **`CouchbaseQueryDocumentStore`** with Hyperscale (BHIVe) vector index, which offers:

- **High-performance vector search at massive scale** (billions of documents)
- **Pure vector search** optimized for RAG applications
- **SQL++ queries** for efficient vector retrieval
- **Recommended for Couchbase 8.0+** for pure vector similarity search

### Option 3: Composite Vector Index (`chat_with_pdf.py`)

Uses **`CouchbaseQueryDocumentStore`** with Composite vector index, which offers:

- **Vector search with metadata filtering**
- **Combines vector fields with scalar fields** for pre-filtering
- **SQL++ queries** with efficient filtered vector retrieval
- **Best for filtered vector search** scenarios (e.g., filter by date, category, user_id)
- **Recommended for Couchbase 8.0+** when you need to filter before vector search

## How does it work?

You can upload your PDFs with custom data & ask questions about the data in the chat box.
For each question, you will get two answers:

- one using RAG (Couchbase logo)
- one using pure LLM - OpenAI (🤖).

The RAG pipeline utilizes Haystack, Couchbase Vector Search, and OpenAI models. It fetches relevant parts of the PDF using vector search & adds them as context for the language model.

## Which Option Should You Choose?

**Use Search Service / FTS (`chat_with_pdf_with_fts.py`) if:**
- You need compatibility with Couchbase 7.6+
- You want to combine vector search with rich full-text search capabilities
- Your use case requires complex FTS filtering and text queries
- You need hybrid search (combining keyword search with semantic search)
- You're already familiar with FTS indexes

**Use Hyperscale Vector Index (`chat_with_pdf.py`) if:**
- You're using Couchbase 8.0+
- You need maximum performance and scalability for pure vector search
- Your use case involves RAG, chatbots, or semantic search without complex filtering
- You want the ultra-low memory footprint and highest throughput
- You're working with billions of documents

**Use Composite Vector Index (`chat_with_pdf.py`) if:**
- You're using Couchbase 8.0+
- You need to filter documents by metadata before performing vector search
- Your use case involves filtered vector search (e.g., by date, category, user_id, status)
- You want to combine scalar field filtering with vector similarity search
- You need efficient pre-filtering before semantic search


## Setup and Installation

### Install dependencies

`pip install -r requirements.txt`

### Set the environment secrets

Copy the `secrets.example.toml` file in `.streamlit` folder and rename it to `secrets.toml` and replace the placeholders with the actual values for your environment

**For Hyperscale or Composite Vector Index (`chat_with_pdf.py`):**
```
DB_CONN_STR = "<couchbase_cluster_connection_string>"
DB_USERNAME = "<couchbase_username>"
DB_PASSWORD = "<couchbase_password>"
DB_BUCKET = "<bucket_name>"
DB_SCOPE = "<scope_name>"
DB_COLLECTION = "<collection_name>"
OPENAI_API_KEY = "<openai_api_key>"
```

**For Search Service / FTS (`chat_with_pdf_with_fts.py`):**

Add one additional environment variable to the above configuration:
```
INDEX_NAME = "<vector_capable_fts_index_name>"
```

### Create the Vector Index

Depending on which implementation you choose, you'll need to create the appropriate index:

**For Hyperscale or Composite Vector Index (`chat_with_pdf.py`):**

This demo uses Couchbase new Vector Indexes (introduced in version 8.0). Choose between:

- **Hyperscale Vector Index**: Optimized for pure vector search at scale. Perfect for RAG, chatbots, and scenarios needing fast vector similarity search on large datasets.

- **Composite Vector Index**: Combines vector fields with scalar fields, allowing you to apply metadata filters before vector search (e.g., date, category, user_id).

Learn more about these vector indexes [here](https://docs.couchbase.com/cloud/vector-index/use-vector-indexes.html).

**For Search Service / FTS (`chat_with_pdf_with_fts.py`):**

You'll need to create a Full Text Search index with vector capabilities. See the FTS index creation section below for detailed instructions.

### Key Components

- Streamlit: Provides the web interface
- Haystack: Orchestrates the RAG pipeline
- Couchbase: Serves as the high-performance vector store
- OpenAI: Supplies embeddings and the language model

## Vector Index Creation

### Hyperscale or Composite Vector Index (for `chat_with_pdf.py`)

You need to create a Hyperscale or Composite vector index on your collection **after** loading some documents (required for index training). Choose between BHIVe or Composite Index based on your use case. Whichever vector index (Hyperscale or Composite) you choose won't affect the functionality of this demo, though performance differences may occur.

**Option 1: Hyperscale Vector Index (Recommended)**

Hyperscale is a dedicated vector index optimized for pure vector search at massive scale. Use this for the best performance in RAG applications. Refer to the [Hyperscale Vector Index Guide](https://docs.couchbase.com/cloud/vector-index/hyperscale-vector-index.html) for detailed instructions.

Creating a Hyperscale Index using SQL++ (use Couchbase Query Workbench or programmatically):

```sql
CREATE VECTOR INDEX idx_pdf_hyperscale
ON `bucket_name`.`scope_name`.`collection_name`(embedding VECTOR) 
WITH {
  "dimension": 1536,           
  "similarity": "DOT"         
};
```

**Option 2: Composite Vector Index**

Composite indexes combine vector fields with other scalar fields. This is useful when you need to filter documents by metadata before performing vector search.

Creating a Composite Index using SQL++:

```sql
CREATE INDEX idx_pdf_composite 
ON `bucket_name`.`scope_name`.`collection_name`(embedding VECTOR) 
WITH {
  "dimension": 1536,
  "similarity": "DOT"
};
```

**Index Parameters:**
- `dimension`: Must match your embedding model (1536 for OpenAI ada-002/ada-003, 768 for sentence-transformers)
- `similarity`: Must match the similarity metric in `CouchbaseQueryDocumentStore`. Use `DOT` for dot product (recommended for OpenAI embeddings)

**Important Notes:**
1. **Index Creation Timing**: Hyperscale and Composite vector indexes require training data. Create the index **after** you've loaded the documents into your collection.
2. **Similarity Metric**: The `similarity` parameter in the index **must match** the `similarity` parameter in your `CouchbaseQueryDocumentStore` configuration.
3. **Dimension**: Must match your embedding model's output dimensions.

**Verifying Your Index:**

After creating the index, verify it exists:

```sql
SELECT * FROM system:indexes 
WHERE name="idx_pdf_hyperscale";  -- or idx_pdf_composite
```

### FTS Vector Index (for `chat_with_pdf_with_fts.py`)

For the FTS-based implementation, you need to create a Full Text Search index with vector capabilities. This index should be created **after** loading some documents into your collection.

**Creating an FTS Index with Vector Support**

You can create the index using the Couchbase UI or by importing the provided index definition.

Using Couchbase Capella:
1. Follow the import instructions [here](https://docs.couchbase.com/cloud/search/import-search-index.html)
2. Use the provided `sampleSearchIndex.json` file in this repository
3. Update the following values in the JSON before importing:
   - `sourceName`: Replace `haystack_bucket` with your bucket name
   - `types`: Replace `haystack_scope.haystack_collection` with your actual `scope_name.collection_name`
4. Import the file in Capella
5. Click on Create Index

Using Couchbase Server:
1. Navigate to Search -> Add Index -> Import
2. Copy the contents of `sampleSearchIndex.json` from this repository
3. Update the following values:
   - `sourceName`: Replace `haystack_bucket` with your bucket name
   - `types`: Replace `haystack_scope.haystack_collection` with your actual `scope_name.collection_name`
4. Paste the updated JSON in the Import screen
5. Click on Create Index

**FTS Index Definition**

The `sampleSearchIndex.json` file contains a pre-configured FTS index with vector capabilities. Key features:
- **Index Name**: `pdf_search` (customizable)
- **Vector Field**: `embedding` with 1536 dimensions
- **Similarity**: `dot_product` (optimized for OpenAI embeddings)
- **Text Field**: `content` for document text
- **Metadata**: Dynamic mapping for `meta` fields

## Run the Application

**For Hyperscale or Composite Vector Index:**
```
streamlit run chat_with_pdf.py
```

**For Search Service / FTS:**
```
streamlit run chat_with_pdf_with_fts.py
```

## Implementation Details

### Hyperscale and Composite Vector Index Implementation (`chat_with_pdf.py`)

This demo uses the following key components:

1. **CouchbaseQueryDocumentStore**: 
   - Configured with `QueryVectorSearchType.ANN` for fast approximate nearest neighbor search
   - Uses `QueryVectorSearchSimilarity.DOT` for dot product similarity (recommended for OpenAI embeddings)
   - Supports both **Hyperscale (BHIVe)** and **Composite** indexes
   - Leverages SQL++ for efficient vector retrieval
   - Same code works for both index types - just create the appropriate index

2. **CouchbaseQueryEmbeddingRetriever**: 
   - Uses SQL++ queries with `APPROX_VECTOR_DISTANCE()` function for ANN search
   - Retrieves top-k most similar documents based on embedding similarity
   - Optimized for low-latency, high-throughput vector search

3. **OpenAI Embeddings**: 
   - `text-embedding-ada-002` model with 1536 dimensions
   - Generates embeddings for both documents and queries

For more details on implementation, refer to the extensive code comments in `chat_with_pdf.py`.

### Search Service / FTS Implementation (`chat_with_pdf_with_fts.py`)

This alternative implementation uses:

1. **CouchbaseSearchDocumentStore**:
   - Uses Full Text Search service for vector indexing and retrieval
   - Compatible with Couchbase 7.6+ and 8.0+
   - Supports rich text search combined with vector similarity

2. **CouchbaseSearchEmbeddingRetriever**:
   - Leverages FTS vector search capabilities
   - Retrieves top-k most similar documents using FTS queries
   - Supports complex filtering with FTS query syntax

3. **OpenAI Embeddings**:
   - Same `text-embedding-ada-002` model with 1536 dimensions
   - Generates embeddings for both documents and queries

For more details on FTS implementation, refer to the code comments in `chat_with_pdf_with_fts.py`.

## Additional Resources

- [Couchbase Vector Index Documentation](https://docs.couchbase.com/cloud/vector-index/vectors-and-indexes-overview.html)
- [Haystack Documentation](https://docs.haystack.deepset.ai/)
- [couchbase-haystack GitHub Repository](https://github.com/Couchbase-Ecosystem/couchbase-haystack)