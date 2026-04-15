#!/usr/bin/env python3
"""Validate config module can be imported and serialized without errors."""
from dataclasses import asdict
import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, "src")

from direct_indexing.config import AppConfig

# Test 1: Default config serializes to JSON
cfg = AppConfig()
json.dumps(asdict(cfg))
print("Config serialization: OK")

# Test 2: Env override works
os.environ["ALPACA_API_KEY"] = "test_key"
cfg = AppConfig.from_env()
assert cfg.alpaca.api_key == "test_key", f"Expected 'test_key', got '{cfg.alpaca.api_key}'"
print("Env config loading: OK")

print("All config validations passed!")