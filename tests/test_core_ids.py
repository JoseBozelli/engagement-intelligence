import pytest

from engagement_intelligence.core import ids


def test_event_id_is_deterministic():
    assert ids.make_event_id(42, "acc_000001", 7) == ids.make_event_id(42, "acc_000001", 7)


@pytest.mark.parametrize(
    "changed", [(43, "acc_000001", 7), (42, "acc_000002", 7), (42, "acc_000001", 8)]
)
def test_event_id_changes_with_any_input(changed):
    assert ids.make_event_id(*changed) != ids.make_event_id(42, "acc_000001", 7)


def test_event_and_ref_namespaces_differ():
    e = ids.make_event_id(42, "acc_000001", 7)
    r = ids.make_ref_id(42, "acc_000001", 7)
    assert e.removeprefix("ev_") != r.removeprefix("ref_")


def test_generated_ids_match_their_formats_and_generic_rule():
    e = ids.make_event_id(1, "acc_000000", 0)
    r = ids.make_ref_id(1, "acc_000000", 0)
    a = ids.make_account_id(12)
    assert ids.EVENT_ID_RE.fullmatch(e)
    assert ids.REF_ID_RE.fullmatch(r)
    assert ids.ACCOUNT_ID_RE.fullmatch(a)
    assert all(ids.is_valid_id(x) for x in (e, r, a))


@pytest.mark.parametrize(
    "external", ["acc_000001", "3f2b8c1e-9a4d-4c2e-8f1a-2b3c4d5e6f70", "sess_00020", "tenant-01"]
)
def test_generic_validator_accepts_external_formats(external):
    assert ids.is_valid_id(external)


@pytest.mark.parametrize("bad", ["", " acc", "acc 1", "a" * 129, "-leading"])
def test_generic_validator_rejects_malformed(bad):
    assert not ids.is_valid_id(bad)


def test_account_index_bounds():
    with pytest.raises(ValueError):
        ids.make_account_id(-1)
    with pytest.raises(ValueError):
        ids.make_account_id(1_000_000)