import ast
from pathlib import Path

import engagement_intelligence.core as core

FORBIDDEN_PREFIXES = (
    "engagement_intelligence.schemas",
    "engagement_intelligence.simulation",
    "engagement_intelligence.features",
    "engagement_intelligence.models",
    "engagement_intelligence.service"
)

def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module

def test_core_does_not_import_downstream_modules():
    core_dir = Path(core.__file__).parent
    offenders = [
        (p.name, mod)
        for p in core_dir.glob("*.py")
        for mod in _imports(p)
        if mod.startswith(FORBIDDEN_PREFIXES)
    ]
    assert offenders == []

def test_labels_do_not_depend_on_simulation():
    import engagement_intelligence.labels as labels

    labels_dir = Path(labels.__file__).parent
    offenders = [
        (p.name, mod)
        for p in labels_dir.glob("*.py")
        for mod in _imports(p)
        if mod.startswith("engagement_intelligence.simulation")
    ]
    assert offenders == []

def test_features_do_not_depend_on_simulation():
    import engagement_intelligence.features as features

    features_dir = Path(features.__file__).parent
    offenders = [
        (p.name, mod)
        for p in features_dir.glob("*.py")
        for mod in _imports(p)
        if mod.startswith("engagement_intelligence.simulation")
    ]
    assert offenders == []

def test_scoring_does_not_depend_on_simulation():
    import engagement_intelligence.scoring as scoring

    scoring_dir = Path(scoring.__file__).parent
    offenders = [
        (p.name, mod)
        for p in scoring_dir.glob("*.py")
        for mod in _imports(p)
        if mod.startswith("engagement_intelligence.simulation")
    ]
    assert offenders == []