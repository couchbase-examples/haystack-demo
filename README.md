# RAG Demo using Couchbase, Streamlit, Haystack, and OpenAI

This is a demo app built to chat with your custom PDFs using **Couchbase Vector Search** to augment the OpenAI results in a Retrieval-Augmented-Generation (RAG) model.

## Three Implementation Options

### Option 1: Search Vector Index (`chat_with_pdf_with_search_vector_index.py`)

Uses **`CouchbaseSearchDocumentStore`** with Search vector indexes, which offers:

- **Flexible vector search** 
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

### Option 3: Composite Vector Index

Uses **`CouchbaseQueryDocumentStore`** with Composite vector index, which offers:

- **Vector search with metadata filtering**
- **Combines vector fields with scalar fields** for pre-filtering
- **SQL++ queries** with efficient filtered vector retrieval
- **Best for filtered vector search** scenarios (e.g., filter by date, category, user_id)
- **Recommended for Couchbase 8.0+** when you need to filter before vector search

This demo doesn't use Composite Vector index, but you can easily do so by just removing `VECTOR` from [this line](./chat_with_pdf.py#L109) and keeping the rest same. To learn more about how Composite Vector Indexes are made, you can refer [here](https://docs.couchbase.com/cloud/vector-index/composite-vector-index.html).

## How does it work?

You can upload your PDFs with custom data & ask questions about the data in the chat box.
For each question, you will get two answers:

- one using RAG (Couchbase logo)
- one using pure LLM - OpenAI (🤖).

The RAG pipeline utilizes Haystack, Couchbase Vector Search, and OpenAI models. It fetches relevant parts of the PDF using vector search & adds them as context for the language model.

## Quick Start

1. **Clone this repository**
   ```bash
   git clone <repository-url>
   cd haystack-demo
   ```

2. **Create a Python virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a Couchbase bucket** (via Couchbase UI/Capella) with the name "sample_bucket"

5. **Configure environment variables** (see Setup section below)

6. **Run the Streamlit app**
   ```bash
   # For Hyperscale Vector Index (default)
   streamlit run chat_with_pdf.py
   
   # OR for Search Vector Index
   streamlit run chat_with_pdf_with_search_vector_index.py
   ```

7. **Upload a PDF** - everything else is automatic!

The app automatically creates:
- Scopes and collections
- Vector indexes (after PDF upload for `chat_with_pdf.py`, or on startup for `chat_with_pdf_with_search_vector_index.py`)

## Which Option Should You Choose?
Couchbase Capella supports three types of vector indexes:

- **Hyperscale Vector Index** (`chat_with_pdf.py`) - Best for RAG/chatbot applications with pure semantic search and billions of documents
- **Composite Vector Index** - Best when you need to filter by metadata before vector search
- **Search Vector Index** (`chat_with_pdf_with_search_vector_index.py`) - Best for hybrid searches combining keywords, geospatial, and semantic search

> **For this PDF chat demo, we recommend Hyperscale Vector Index** for optimal performance in RAG applications.

Learn more about choosing the right vector index in the [official Couchbase vector index documentation](https://docs.couchbase.com/cloud/vector-index/use-vector-indexes.html).


## Setup and Installation

### Install dependencies

`pip install -r requirements.txt`

### Set the environment secrets

Copy the `secrets.example.toml` file in `.streamlit` folder and rename it to `secrets.toml` and replace the placeholders with the actual values for your environment

**For Hyperscale Vector Index (`chat_with_pdf.py`):**
```
DB_CONN_STR = "<couchbase_cluster_connection_string>"
DB_USERNAME = "<couchbase_username>"
DB_PASSWORD = "<couchbase_password>"
DB_BUCKET = "<bucket_name>"
DB_SCOPE = "<scope_name>"
DB_COLLECTION = "<collection_name>"
OPENAI_API_KEY = "<openai_api_key>"
```

**For Search Vector Index (`chat_with_pdf_with_search_vector_index.py`):**

Add one additional environment variable to the above configuration:
```
INDEX_NAME = "<vector_capable_search_vector_index_name>"
```

### Automatic Resource Setup

The application automatically handles resource creation in the following order:

**On Application Startup:**

1. Creates the scope if it doesn't exist
2. Creates the collection if it doesn't exist

**After PDF Upload (`chat_with_pdf.py`):**

3. Automatically creates the Hyperscale index after documents are loaded
4. Falls back to creating the index on first query if needed

**On Application Startup (`chat_with_pdf_with_search_vector_index.py`):**

3. Attempts to create the Search Vector index (can be created without documents)

**What You Need:**
- Your Couchbase **bucket must exist** with the name "sample_bucket"
- All other resources (scope, collection, indexes) are created automatically
- **No manual index creation required** - just upload your PDF and the index will be created

**Note**: For `chat_with_pdf.py`, the vector index is created automatically **after you upload your first PDF** because Hyperscale/Composite indexes require documents for training.

## Manual Vector Index Creation (Optional)

**⚠️ Manual creation is NOT required** - the app creates indexes automatically when you upload a PDF. This section is only for advanced users who want manual control.

### Hyperscale or Composite Vector Index

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

[Composite indexes](https://docs.couchbase.com/cloud/vector-index/composite-vector-index.html) combine vector fields with other scalar fields. This is useful when you need to filter documents by metadata before performing vector search.

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
WHERE name LIKE 'idx_%_vector';
```

### Search Vector Vector Index (for `chat_with_pdf_with_search_vector_index.py`)

**Automatic Creation**: The app attempts to create the Search Vector index automatically on startup using the `INDEX_NAME` from your configuration.

**Manual Creation** (if automatic creation fails): Create a Full Text Search index with vector capabilities.

**Creating an Search Vector Index with Vector Support**

If automatic creation fails, you can create the index using the Couchbase UI or by importing the provided index definition.

Using Couchbase Capella:
1. Follow the import instructions [here](https://docs.couchbase.com/cloud/search/import-search-index.html)
2. Use the provided `sampleSearchIndex.json` file in this repository
3. Update the following values in the JSON before importing:
   - `sourceName`: Replace `sample_bucket` with your bucket name
   - `types`: Replace `scope.coll` with your actual `scope_name.collection_name`
4. Import the file in Capella
5. Click on Create Index

Using Couchbase Server:
1. Navigate to Search -> Add Index -> Import
2. Use the provided `sampleSearchIndex.json` file in this repository
3. Update the following values in the JSON before importing:
   - `sourceName`: Replace `sample_bucket` with your bucket name
   - `types`: Replace `scope.coll` with your actual `scope_name.collection_name`
4. Paste the updated JSON in the Import screen
5. Click on Create Index

**Search Vector Index Definition**

The `sampleSearchIndex.json` file contains a pre-configured Search Vector index with vector capabilities. Key features:
- **Index Name**: `sample-index` (customizable)
- **Vector Field**: `embedding` with 1536 dimensions
- **Similarity**: `dot_product` (optimized for OpenAI embeddings)
- **Text Field**: `content` for document text
- **Metadata**: Dynamic mapping for `meta` fields

## Run the Application

**For Hyperscale or Composite Vector Index:**
```
streamlit run chat_with_pdf.py
```

**For Search Vector Index:**
```
streamlit run chat_with_pdf_with_search_vector_index.py
```

## Implementation Details

### Hyperscale Vector Index Implementation (`chat_with_pdf.py`)

This demo uses the following key components:

1. **CouchbaseQueryDocumentStore**: 
   - Configured with `QueryVectorSearchType.ANN` for fast approximate nearest neighbor search
   - Uses `QueryVectorSearchSimilarity.DOT` for dot product similarity (recommended for OpenAI embeddings)
   - Supports both **Hyperscale** and **Composite** vector indexes
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

### Search Vector Index Implementation (`chat_with_pdf_with_search_vector_index.py`)

This alternative implementation uses:

1. **CouchbaseSearchDocumentStore**:
   - Uses Full Text Search service for vector indexing and retrieval
   - Compatible with Couchbase 7.6+ and 8.0+
   - Supports rich text search combined with vector similarity

2. **CouchbaseSearchEmbeddingRetriever**:
   - Leverages Search vector search capabilities
   - Retrieves top-k most similar documents using FTS queries
   - Supports complex filtering with FTS query syntax

3. **OpenAI Embeddings**:
   - Same `text-embedding-ada-002` model with 1536 dimensions
   - Generates embeddings for both documents and queries

For more details on FTS implementation, refer to the code comments in `chat_with_pdf_with_search_vector_index.py`.

## Additional Resources

- [Couchbase Vector Index Documentation](https://docs.couchbase.com/cloud/vector-index/vectors-and-indexes-overview.html)
- [Haystack Documentation](https://docs.haystack.deepset.ai/docs/intro)
- [couchbase-haystack GitHub Repository](https://github.com/Couchbase-Ecosystem/couchbase-haystack)