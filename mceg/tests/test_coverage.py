"""Apply a small review policy to MCEG's native coverage report."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from mceg import ContractV06
from mceg.coverage import Inventory, check_coverage
from mceg.validator import read_document


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "mceg/evidence/coverage.json"
KNOWN_GAPS = {"unreviewed_file", "uncontracted_model", "unreviewed_model"}


def check_review_policy(report, baseline):
    """Allow recorded unfinished reviews, never invalid or newly broken records.

    MCEG reports the gaps; this provider policy compares their identities and
    retains the reviewed inventory boundaries. It does not discover science.
    """
    identity = lambda issue: (issue["kind"], issue["path"], issue["model"])
    assert all(issue["kind"] in KNOWN_GAPS for issue in baseline["issues"])
    defects = [issue for issue in report["issues"] if issue["kind"] not in KNOWN_GAPS]
    assert not defects, defects
    added = {identity(i) for i in report["issues"]} - {identity(i) for i in baseline["issues"]}
    assert not added, added
    assert report["source_roots"] == baseline["source_roots"], "Source scope changed; review the baseline"
    assert report["contract_roots"] == baseline["contract_roots"], "Contract scope changed; review the baseline"
    current = {model["id"]: model for model in report["models"]}
    for previous in baseline["models"]:
        assert previous["id"] in current, "Removed model: " + previous["id"]
        model = current[previous["id"]]
        for collection in ("bindings", "contracts"):
            old = {json.dumps(item, sort_keys=True) for item in previous[collection]}
            new = {json.dumps(item, sort_keys=True) for item in model[collection]}
            assert old <= new, (previous["id"], collection, old - new)


def test_current_inventory_retains_known_gaps_without_new_breakage():
    inventory = Inventory.model_validate(read_document(ROOT / "mceg/models.yaml"))
    report = check_coverage(ROOT, inventory)
    baseline = json.loads(BASELINE.read_text())
    check_review_policy(report, baseline)
    assert report["summary"]["discovered_contracts"] == 6
    assert report["summary"]["uncontracted_models"] > 0
    assert report["summary"]["files_needing_review"] > 0


@pytest.fixture
def example(tmp_path):
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/provider.py"
    source.write_text("def value(x):\n    return x\n")
    digest = sha256(source.read_bytes()).hexdigest()
    implementation = {"repository": "https://example.org/provider", "revision": "fixture",
                      "entry_points": ["src/provider.py:value"]}
    contract = ContractV06.model_validate({
        "schema_version": "0.6",
        "model": {"id": "fixture.contract", "name": "Coverage fixture",
                  "description": "A native contract for policy mutation checks.",
                  "implementation": implementation},
        "assumptions": [], "limitations": [],
        "domains": [{"kind": "scientific_applicability", "status": "unknown", "conditions": [],
                     "description": "No physical claim is established by this fixture.", "evidence": ["fixture.source"]}],
        "claims": [{"id": "fixture.claim", "revision": "1", "kind": "theoretical",
                    "statement": "The function represents the declared identity.", "scope": "Fixture only."}],
        "evidence": [{"id": "fixture.source", "kind": "implementation", "summary": "Fixture source.",
                      "sources": ["src/provider.py"], "scope": "No execution evidence."}],
        "subjects": [{"id": "fixture.subject", "implementation": implementation,
                      "selected_files": {"src/provider.py": digest}}],
        "contexts": [{"id": "fixture.context", "scope": "Coverage mutation fixture."}],
    })
    (tmp_path / "contracts").mkdir()
    (tmp_path / "contracts/example.json").write_text(contract.model_dump_json())
    inventory = Inventory.model_validate({
        "inventory_version": "0.1", "scope": "Policy fixture.",
        "source_roots": ["src"], "contract_roots": ["contracts"],
        "files": {"src/provider.py": {"sha256": digest, "role": "model", "note": "Reviewed fixture."}},
        "models": [
            {"id": "fixture.current", "description": "Registered fixture.",
             "bindings": [{"path": "src/provider.py", "symbol": "value"}],
             "contracts": [{"path": "contracts/example.json", "model_id": "fixture.contract"}]},
            {"id": "fixture.pending", "description": "A known missing contract.", "review_pending": True,
             "bindings": [{"path": "src/provider.py", "symbol": "value", "settings": {"variant": "pending"}}],
             "contracts": []},
        ],
    })
    return tmp_path, inventory, check_coverage(tmp_path, inventory)


def test_known_gaps_are_explicit_and_allowed(example):
    root, inventory, baseline = example
    assert baseline["status"] == "gaps"
    check_review_policy(check_coverage(root, inventory), baseline)


@pytest.mark.parametrize("change", ["new_file", "changed_file", "missing_symbol", "removed_link",
                                   "invalid_contract", "removed_model", "reduced_scope", "removed_variant_binding"])
def test_native_mutation_cannot_be_hidden_by_known_gaps(example, change):
    root, inventory, baseline = example
    document = inventory.model_dump(mode="json")
    if change == "new_file":
        (root / "src/new_model.py").write_text("def extra(x):\n    return x*x\n")
    elif change == "changed_file":
        (root / "src/provider.py").write_text("def value(x):\n    return x*x\n")
    elif change == "missing_symbol":
        # Even a refreshed file hash does not repair a stale symbol binding.
        (root / "src/provider.py").write_text("def renamed(x):\n    return x\n")
        document["files"]["src/provider.py"]["sha256"] = sha256((root / "src/provider.py").read_bytes()).hexdigest()
    elif change == "removed_link":
        document["models"][0]["contracts"] = []
    elif change == "invalid_contract":
        (root / "contracts/example.json").write_text("{}")
    elif change == "removed_model":
        document["models"].pop()
    elif change == "reduced_scope":
        document["source_roots"] = ["src/provider.py"]
    elif change == "removed_variant_binding":
        document["models"][1]["bindings"] = deepcopy(document["models"][0]["bindings"])
    with pytest.raises(AssertionError):
        check_review_policy(check_coverage(root, Inventory.model_validate(document)), baseline)


def test_recording_a_defect_as_a_known_gap_is_rejected(example):
    root, inventory, _ = example
    (root / "src/provider.py").write_text("def value(x):\n    return x*x\n")
    defective = check_coverage(root, inventory)
    with pytest.raises(AssertionError):
        check_review_policy(defective, defective)
