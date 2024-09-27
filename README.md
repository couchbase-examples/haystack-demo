## RAG Demo using Couchbase, Streamlit, Haystack, and OpenAI

This is a demo app built to chat with your custom PDFs using the vector search capabilities of Couchbase to augment the OpenAI results in a Retrieval-Augmented-Generation (RAG) model.

### How does it work?

You can upload your PDFs with custom data & ask questions about the data in the chat box.
For each question, you will get two answers:

- one using RAG (Couchbase logo)
- one using pure LLM - Gemini Pro (🤖).

The RAG pipeline utilizes Haystack, Couchbase Vector Search, and a OpenAI model. It fetches relevant parts of the PDF using vector search and adds them as context for the language model.


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
    INDEX_NAME = "<vector_capable_fts_index_name>"
    OPENAI_API_KEY = "<openai_api_key>"
```

- #### Create the Search Index on Full Text Service

  We need to create the Search Index on the Full Text Service in Couchbase. For this demo, you can import the following index using the instructions.

  - [Couchbase Capella](https://docs.couchbase.com/cloud/search/import-search-index.html)

    - Copy the index definition to a new file index.json
    - Import the file in Capella using the instructions in the documentation.
    - Click on Create Index to create the index.

  - [Couchbase Server](https://docs.couchbase.com/server/current/search/import-search-index.html)

    - Click on Search -> Add Index -> Import
    - Copy the following Index definition in the Import screen
    - Click on Create Index to create the index.

- #### Key Components

    - Streamlit: Provides the web interface
    - Haystack: Orchestrates the RAG pipeline
    - Couchbase: Serves as the vector store
    - OpenAI: Supplies the language model

  #### Index Definition

  Here, we are creating the index `pdf_search` on the documents in the `haystack_collection` collection within the `haystack_scope` scope in the bucket `haystack_bucket`. The Vector field is set to `embeddings` with 1536 dimensions and the text field set to `text`. We are also indexing and storing all the fields under `metadata` in the document as a dynamic mapping to account for varying document structures. The similarity metric is set to `dot_product`. If there is a change in these parameters, please adapt the index accordingly.

  ```
    {
        "name": "pdf_search",
        "type": "fulltext-index",
        "sourceType": "gocbcore",
        "sourceName": "haystack_bucket",
        "planParams": {
            "indexPartitions": 1,
            "numReplicas": 0
        },
        "params": {
            "doc_config": {
                "docid_prefix_delim": "",
                "docid_regexp": "",
                "mode": "scope.collection.type_field",
                "type_field": "type"
            },
            "mapping": {
                "default_analyzer": "standard",
                "default_datetime_parser": "dateTimeOptional",
                "index_dynamic": true,
                "store_dynamic": true,
                "default_mapping": {
                    "dynamic": true,
                    "enabled": false
                },
                "types": {
                    "haystack_scope.haystack_collection": {
                        "dynamic": false,
                        "enabled": true,
                        "properties": {
                            "content": {
                                "enabled": true,
                                "fields": [
                                    {
                                        "docvalues": true,
                                        "include_in_all": false,
                                        "include_term_vectors": false,
                                        "index": true,
                                        "name": "content",
                                        "store": true,
                                        "type": "text"
                                    }
                                ]
                            },
                            "embedding": {
                                "enabled": true,
                                "dynamic": false,
                                "fields": [
                                    {
                                        "vector_index_optimized_for": "recall",
                                        "docvalues": true,
                                        "dims": 1536,
                                        "include_in_all": false,
                                        "include_term_vectors": false,
                                        "index": true,
                                        "name": "embedding",
                                        "similarity": "dot_product",
                                        "store": true,
                                        "type": "vector"
                                    }
                                ]
                            },
                            "dataframe": {
                                "enabled": true,
                                "fields": [
                                    {
                                        "docvalues": true,
                                        "include_in_all": false,
                                        "include_term_vectors": false,
                                        "index": true,
                                        "name": "dataframe",
                                        "store": true,
                                        "analyzer": "keyword",
                                        "type": "text"
                                    }
                                ]
                            },
                            "meta": {
                                "dynamic": true,
                                "enabled": true,
                                "properties": {
                                    "name": {
                                        "enabled": true,
                                        "fields": [
                                            {
                                                "docvalues": true,
                                                "include_in_all": false,
                                                "include_term_vectors": false,
                                                "index": true,
                                                "name": "name",
                                                "store": true,
                                                "analyzer": "keyword",
                                                "type": "text"
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

```

- #### Run the application

  `streamlit run chat_with_pdf.py`

For more details on implementation, refer to the code comments in chat_with_pdf.py.