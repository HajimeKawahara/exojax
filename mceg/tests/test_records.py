"""Native record checks; this module never imports the numerical provider."""

from importlib.metadata import distribution
import json
from pathlib import Path

import pytest

from mceg import load_contract, mathematical_model_digest
from mceg.check import check_contract
from mceg.validator import read_document


ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "mceg/contracts"
FAMILIES = ("planck", "lte_line_opacity", "atmospheric_columns", "thermal_emission",
            "transmission", "spectral_observation")


def assert_native_review_checks(report, assessment):
    """Local policy: decisive execution reviews need native source/artifact checks.

    A recorded contradiction is a judgment, not a broken record.
    """
    checks = report["checks"]
    if assessment.kind == "evidence_support":
        support = [item["status"] for item in checks if item["name"] == "recorded_support"]
        assert support == [{"supported": "PASS", "contradicted": "FAIL",
                            "unknown": "UNKNOWN"}[assessment.status]]
    failures = [item for item in checks if item["status"] == "FAIL"
                and not (item["name"] == "recorded_support"
                         and assessment.kind == "evidence_support"
                         and assessment.status == "contradicted")]
    assert not failures, failures
    if assessment.kind == "evidence_support" and assessment.status in {"supported", "contradicted"}:
        # These provider judgments require a pinned source and execution record.
        for name in ("selected_files", "artifact_hashes"):
            assert [item["status"] for item in checks if item["name"] == name] == ["PASS"]


def test_installed_mceg_matches_the_pinned_revision():
    requirement = next(line for line in (ROOT / "mceg/requirements.txt").read_text().splitlines()
                       if line.startswith("mceg[symbolic] @ git+"))
    expected = requirement.rsplit("@", 1)[1]
    direct_url = json.loads(distribution("mceg").read_text("direct_url.json") or "{}")
    assert direct_url.get("vcs_info", {}).get("commit_id") == expected


@pytest.mark.parametrize("family", FAMILIES)
def test_native_definitions_local_reviews_and_selected_sources(family):
    path = CONTRACTS / (family + ".yaml")
    contract = load_contract(path)
    assert contract.schema_version == "0.6"
    template = read_document(ROOT / "mceg/templates" / (family + ".yaml"))
    lock = read_document(ROOT / "mceg/imports" / (family + ".lock.json"))
    for item in lock["imports"]:
        pin = item["model"]
        assert mathematical_model_digest(contract, pin["id"], pin["revision"]) == pin["sha256"]
    for collection in ("subjects", "contexts", "claims", "assessments", "bindings", "follow_ups"):
        records = getattr(contract, collection)
        original = template.get(collection, [])
        assert len(records) == len(original)
        for record, source in zip(records, original):
            assert record == type(record).model_validate(source)
            assert record.id.startswith("exojax."), (collection, record.id)
    pending = {item.target for item in contract.follow_ups if item.target_kind == "assessment"}
    for assessment in contract.assessments:
        report = check_contract(path, {}, assessment.id, root=ROOT)
        assert_native_review_checks(report, assessment)
        if assessment.kind == "evidence_support" and assessment.status == "unknown":
            assert assessment.id in pending, "Unknown support needs an explicit follow-up"


@pytest.mark.parametrize("digest", ["0" * 64, None], ids=["changed_artifact", "removed_artifacts"])
def test_recorded_contradiction_does_not_hide_artifact_defects(tmp_path, digest):
    path = CONTRACTS / "spectral_observation.yaml"
    contract = load_contract(path)
    assessment = next(item for item in contract.assessments
                      if item.kind == "evidence_support" and item.status == "contradicted")
    report = check_contract(path, {}, assessment.id, root=ROOT)
    assert_native_review_checks(report, assessment)
    selected = report["integrity"]["artifact_hashes"][0]
    altered = contract.model_copy(deep=True)
    if digest is None:
        for evidence in altered.evidence:
            if evidence.id in report["audited_evidence"]:
                evidence.artifacts = []
    else:
        evidence = next(item for item in altered.evidence if item.id == selected["evidence"])
        artifact = next(item for item in evidence.artifacts if item.source == selected["source"])
        artifact.sha256 = digest
    changed = tmp_path / "changed.json"
    changed.write_text(altered.model_dump_json())
    report = check_contract(changed, {}, assessment.id, root=ROOT)
    assert next(item["status"] for item in report["checks"]
                if item["name"] == "artifact_hashes") == ("UNKNOWN" if digest is None else "FAIL")
    with pytest.raises(AssertionError):
        assert_native_review_checks(report, assessment)
