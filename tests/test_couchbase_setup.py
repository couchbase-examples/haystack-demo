"""Unit tests for couchbase_setup.py. They use fakes and need no Couchbase cluster.

Run from the repository root:
    python -m unittest discover -s tests
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import couchbase_setup  # noqa: E402
from couchbase.exceptions import CollectionAlreadyExistsException, ScopeAlreadyExistsException  # noqa: E402


class FakeCollectionManager:
    """In-memory stand-in for a bucket's collection manager."""

    def __init__(self, scopes=None):
        # {scope_name: [collection_name, ...]}
        self.scopes = {name: list(colls) for name, colls in (scopes or {}).items()}
        self.created_scopes = []
        self.created_collections = []

    def get_all_scopes(self):
        return [
            SimpleNamespace(name=name, collections=[SimpleNamespace(name=c) for c in colls])
            for name, colls in self.scopes.items()
        ]

    def create_scope(self, scope_name):
        self.scopes[scope_name] = []
        self.created_scopes.append(scope_name)

    def create_collection(self, scope_name, collection_name):
        self.scopes[scope_name].append(collection_name)
        self.created_collections.append((scope_name, collection_name))


class SetupTestCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(couchbase_setup, "st")
        self.st = patcher.start()
        self.addCleanup(patcher.stop)


class CreateScopeTests(SetupTestCase):
    def test_creates_missing_scope(self):
        manager = FakeCollectionManager({"_default": ["_default"]})
        self.assertTrue(couchbase_setup.create_scope_if_not_exists(manager, "haystack"))
        self.assertEqual(manager.created_scopes, ["haystack"])

    def test_existing_scope_is_left_alone(self):
        manager = FakeCollectionManager({"haystack": []})
        self.assertFalse(couchbase_setup.create_scope_if_not_exists(manager, "haystack"))
        self.assertEqual(manager.created_scopes, [])

    def test_already_exists_race_is_not_an_error(self):
        manager = FakeCollectionManager()
        manager.create_scope = mock.Mock(side_effect=ScopeAlreadyExistsException())
        self.assertFalse(couchbase_setup.create_scope_if_not_exists(manager, "haystack"))
        self.st.warning.assert_not_called()

    def test_other_errors_are_reported_as_warnings(self):
        manager = FakeCollectionManager()
        manager.get_all_scopes = mock.Mock(side_effect=RuntimeError("boom"))
        self.assertFalse(couchbase_setup.create_scope_if_not_exists(manager, "haystack"))
        self.st.warning.assert_called_once()


class CreateCollectionTests(SetupTestCase):
    def test_creates_missing_collection(self):
        manager = FakeCollectionManager({"haystack": []})
        self.assertTrue(couchbase_setup.create_collection_if_not_exists(manager, "haystack", "docs"))
        self.assertEqual(manager.created_collections, [("haystack", "docs")])

    def test_existing_collection_is_left_alone(self):
        manager = FakeCollectionManager({"haystack": ["docs"]})
        self.assertFalse(couchbase_setup.create_collection_if_not_exists(manager, "haystack", "docs"))
        self.assertEqual(manager.created_collections, [])

    def test_missing_scope_is_reported(self):
        manager = FakeCollectionManager({"_default": ["_default"]})
        self.assertFalse(couchbase_setup.create_collection_if_not_exists(manager, "haystack", "docs"))
        self.st.error.assert_called_once()

    def test_already_exists_race_is_not_an_error(self):
        manager = FakeCollectionManager({"haystack": []})
        manager.create_collection = mock.Mock(side_effect=CollectionAlreadyExistsException())
        self.assertFalse(couchbase_setup.create_collection_if_not_exists(manager, "haystack", "docs"))
        self.st.warning.assert_not_called()


class SetupCouchbaseResourcesTests(SetupTestCase):
    def _run(self, manager):
        cluster = mock.Mock()
        cluster.bucket.return_value.collections.return_value = manager
        with mock.patch.object(couchbase_setup, "Cluster", return_value=cluster) as cluster_cls, \
                mock.patch.object(couchbase_setup.time, "sleep") as sleep:
            result = couchbase_setup.setup_couchbase_resources(
                cluster_connection_string="couchbase://localhost",
                username="Administrator",
                password="password",
                bucket_name="sample_bucket",
                scope_name="haystack",
                collection_name="docs",
            )
        return result, cluster, cluster_cls, sleep

    def test_creates_scope_and_collection_and_returns_cluster(self):
        manager = FakeCollectionManager({"_default": ["_default"]})
        result, cluster, cluster_cls, sleep = self._run(manager)
        self.assertIs(result, cluster)
        cluster_cls.assert_called_once()
        self.assertEqual(cluster_cls.call_args.args[0], "couchbase://localhost")
        cluster.bucket.assert_called_once_with("sample_bucket")
        self.assertEqual(manager.created_scopes, ["haystack"])
        self.assertEqual(manager.created_collections, [("haystack", "docs")])
        sleep.assert_called_once()

    def test_existing_resources_skip_the_settle_delay(self):
        manager = FakeCollectionManager({"haystack": ["docs"]})
        result, cluster, _, sleep = self._run(manager)
        self.assertIs(result, cluster)
        sleep.assert_not_called()

    def test_connection_failure_returns_none_and_reports(self):
        with mock.patch.object(couchbase_setup, "Cluster", side_effect=RuntimeError("unreachable")):
            result = couchbase_setup.setup_couchbase_resources(
                "couchbase://nowhere", "u", "p", "b", "s", "c"
            )
        self.assertIsNone(result)
        self.st.error.assert_called_once()


class CheckEnvironmentVariableTests(SetupTestCase):
    def test_missing_variable_stops_the_app(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            couchbase_setup.check_environment_variable("DB_CONN_STR")
        self.st.error.assert_called_once()
        self.st.stop.assert_called_once()

    def test_present_variable_passes(self):
        with mock.patch.dict(os.environ, {"DB_CONN_STR": "couchbase://localhost"}):
            couchbase_setup.check_environment_variable("DB_CONN_STR")
        self.st.stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
