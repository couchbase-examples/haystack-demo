"""Couchbase setup shared by both Streamlit apps.

Both demos (`chat_with_pdf_with_query_vector_index.py` and
`chat_with_pdf_with_search_vector_index.py`) need the same steps before they
can store documents: check the required environment variables, connect to the
cluster, and create the scope and collection if they are missing. Each app
then creates its own kind of vector index.
"""

import os
import time
from datetime import timedelta

import streamlit as st
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import CollectionAlreadyExistsException, ScopeAlreadyExistsException
from couchbase.options import ClusterOptions


def check_environment_variable(variable_name):
    """Check if environment variable is set"""
    if variable_name not in os.environ:
        st.error(f"{variable_name} environment variable is not set. Please add it to the secrets.toml file")
        st.stop()


def create_scope_if_not_exists(collection_manager, scope_name):
    """Create scope if it doesn't exist. Returns True if it was created."""
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
    """Create collection if it doesn't exist. Returns True if it was created."""
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


def setup_couchbase_resources(cluster_connection_string, username, password, bucket_name, scope_name, collection_name):
    """Connect to the cluster and create the scope and collection if needed.

    Returns the connected Cluster, or None if setup failed (the error is shown
    in the app and the demo carries on with whatever already exists).
    """
    try:
        auth = PasswordAuthenticator(username, password)
        cluster = Cluster(cluster_connection_string, ClusterOptions(auth))
        cluster.wait_until_ready(timedelta(seconds=10))

        collection_manager = cluster.bucket(bucket_name).collections()

        scope_created = create_scope_if_not_exists(collection_manager, scope_name)
        collection_created = create_collection_if_not_exists(collection_manager, scope_name, collection_name)

        # Newly created scopes/collections take a moment to become usable
        if scope_created or collection_created:
            time.sleep(2)

        return cluster

    except Exception as e:
        st.error(f"Error during Couchbase setup: {str(e)}")
        st.info("Continuing with existing resources...")
        return None
