"""Verify assembly boundaries without allocating production opacity tables."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def assembly(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("assemble_rce_wasp18b_opacity")


def fixture(assembly, tmp_path):
    settings = {"resolved_nu_min": 3300., "resolved_nu_max": 12500.}
    fingerprint = assembly.base._digest_json(settings)
    star, cia = tmp_path / "star.txt", tmp_path / "cia.txt"
    star.write_text("Pinned stellar input")
    cia.write_text("Pinned CIA input")
    config = dict(schema_version=assembly.base.SCHEMA_VERSION,
        files={"star": assembly.base._file_entry(star, "stellar"), "cia": assembly.base._file_entry(cia, "CIA")},
        stellar={"asset": "star"}, cia=[{"asset": "cia"}], column={"pressure_boundaries_bar": [1e-5, 1., 10.]},
        opacity={"ckd": {}, "lbl": {}, "mapping": {}, "omitted_absorbers": ["OH", "Fe", "atomic metals", "TiO"]},
        preparation={"settings": settings, "grid_fingerprint": fingerprint, "species": {}},
        missing_requirements=["Previous convergence requirement"])
    manifest = tmp_path / "baseline.json"
    manifest.write_text(json.dumps(config))

    def prepare(species, atomic):
        directory = tmp_path / species
        directory.mkdir()
        source = directory / "lines.txt"
        source.write_text("Pinned raw transitions")
        source_entry = assembly.base._file_entry(source, "line list")
        source_entry["stat"] = [source.stat().st_size, source.stat().st_mtime_ns]
        plan = dict(species=species, settings=settings, grid_fingerprint=fingerprint,
                    number_of_chunks=1, mdb_path=str(source))
        provenance = dict(sources={str(source): source_entry}, broadener="ExoMol H2 test broadening")
        if atomic:
            plan["atomic_preparation"] = dict(driver_sha256="pinned atomic driver", source_url="http://kurucz.harvard.edu/linelists/gfall/gf2600.all",
                broadening="Native atomic widths", perturber_fractions_H_He_H2=[0., .16, .84])
            provenance["atomic_preparation"] = plan["atomic_preparation"]
        else:
            plan["supplemental_preparation"] = dict(driver_sha256="pinned supplemental driver")
            provenance["supplemental_preparation"] = plan["supplemental_preparation"]
        files = {}
        for name in ("ckd.npz", "diffgrid.npz", "diffgrid_metadata.json"):
            path = directory / name
            path.write_text("Hash fixture for " + species + name)
            files[name] = assembly.base._file_entry(path, "prepared opacity")
        chunk = directory / "chunk_complete.json"
        chunk.write_text(json.dumps(dict(grid_fingerprint=fingerprint, mdb_path=str(source),
                                         files=files, provenance=provenance)))
        complete = directory / "complete.json"
        complete.write_text(json.dumps(dict(plan=plan, files=files, no_line_chunks=[],
            chunk_records=[assembly.base._file_entry(chunk, "chunk provenance")])))
        return complete

    return manifest, prepare("H1O1", False), prepare("Fe1", True)


def test_preserves_inputs_and_records_correct_atomic_physics(assembly, tmp_path):
    manifest, molecular, atomic = fixture(assembly, tmp_path)
    before = manifest.read_bytes()
    output = tmp_path / "combined.json"
    assert assembly.assemble(manifest, [molecular, atomic], output) == output
    combined = assembly.load_manifest(output).config
    baseline = json.loads(before)
    assert manifest.read_bytes() == before
    for field in ("stellar", "cia", "column"):
        assert combined[field] == baseline[field]
    for name, entry in baseline["files"].items():
        assert combined["files"][name] == entry
    mapping = combined["opacity"]["mapping"]
    assert mapping["Fe1"]["isotope_fraction"] == mapping["H1O1"]["isotope_fraction"] == 1.
    assert mapping["Fe1"]["broadening"].startswith("Native atomic widths")
    assert mapping["H1O1"]["broadening"] == "ExoMol H2 test broadening"
    assert "not a terrestrial" in mapping["H1O1"]["isotope_convention"]
    assert not combined["opacity_assembly"]["physical_validation_complete"]
    assert combined["opacity"]["omitted_absorbers"] == ["atomic metals", "TiO"]
    assert baseline["missing_requirements"][0] in combined["missing_requirements"]
    with pytest.raises(ValueError, match="new manifest"):
        assembly.assemble(manifest, [molecular], manifest)


@pytest.mark.parametrize("failure", ["grid", "table", "chunk", "source"])
def test_rejects_incompatible_or_changed_preparation(assembly, tmp_path, failure):
    manifest, molecular, _ = fixture(assembly, tmp_path)
    complete = json.loads(molecular.read_text())
    if failure == "grid":
        complete["plan"]["grid_fingerprint"] = "changed"
        molecular.write_text(json.dumps(complete))
    elif failure == "table":
        Path(complete["files"]["ckd.npz"]["path"]).write_text("changed")
    elif failure == "chunk":
        Path(complete["chunk_records"][0]["path"]).write_text("changed")
    else:
        Path(complete["plan"]["mdb_path"]).write_text("changed raw data")
    output = tmp_path / "combined.json"
    with pytest.raises(ValueError, match="Incompatible|Changed"):
        assembly.assemble(manifest, [molecular], output)
    assert not output.exists()


def test_explicit_isotope_fraction_requires_source(assembly, tmp_path):
    manifest, molecular, _ = fixture(assembly, tmp_path)
    output = tmp_path / "combined.json"
    with pytest.raises(ValueError, match="isotope-source"):
        assembly.assemble(manifest, [molecular], output, isotope_fractions={"H1O1": .99})
    assembly.assemble(manifest, [molecular], output, isotope_fractions={"H1O1": .99},
                      isotope_source="Explicit sensitivity-test convention")
    mapping = json.loads(output.read_text())["opacity"]["mapping"]["H1O1"]
    assert mapping["isotope_fraction"] == .99
    assert mapping["isotope_source"] == "Explicit sensitivity-test convention"
