from engagement_intelligence.core.config import deep_merge, fingerprint, load_yaml_layers

def test_deep_merge_overrides_nested_without_mutating():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    out = deep_merge(base, {"a": {"y": 3}})
    assert out == {"a": {"x": 1, "y": 3}, "b": 1}
    assert base == {"a": {"x": 1, "y": 2}, "b": 1}

def test_lists_are_replaced_not_merged():
    assert deep_merge({"l": [1, 2]}, {"l": [3]}) == {"l": [3]}

def test_load_yaml_layers(tmp_path):
    (tmp_path / "a.yaml").write_text("p:\n n: 1\n m: 2\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("p:\n n: 5\n", encoding="utf-8")
    assert load_yaml_layers(tmp_path / "a.yaml", tmp_path / "b.yaml") == {"p": {"n": 5, "m": 2}}

def test_fingerprint_is_key_order_invariant():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})

def test_fingerprint_detects_changes():
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})