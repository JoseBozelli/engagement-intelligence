"""
Domain-neutral infrastructure.

Rule: nothing in this package imports a specific event vocabulary or any downstream module (schemas, simulation,
features, models, service).
Enforced by tests/test_architecture.py
"""