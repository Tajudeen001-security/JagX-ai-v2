"""Shared test configuration.

These environment variables MUST be set before jagx_api is imported
(anywhere), because settings capture them at import time. pytest imports
conftest.py before any test module, so this is the one reliable place.
"""

import os

os.environ.setdefault("JAGX_ADMIN_SECRET", "test-admin-secret")
os.environ.setdefault("JAGX_KEYS_FILE", "test_keys.json")
os.environ.setdefault("JAGX_MEMORY_FILE", "test_memory.json")
