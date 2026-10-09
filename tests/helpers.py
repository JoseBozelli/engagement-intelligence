"""Shared test utilities (importable from tests because tests/ is on sys.path)."""


def replace_config(model, path: str, value):
    """Copy of a frozen config with the nested attribute at dotted `path` replaced."""
    head, *rest = path.split(".")
    new = value if not rest else replace_config(getattr(model, head), ".".join(rest), value)
    return model.model_copy(update={head: new})