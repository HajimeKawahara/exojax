"""Keep inspection independent of provider execution and its dependencies."""

import ast
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_runtime_and_default_tests_do_not_require_the_companion():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    for requirement in project["project"]["dependencies"] + project["build-system"]["requires"]:
        assert not requirement.lower().startswith(("mceg", "planetlib")), requirement
    for path in project["tool"]["pytest"]["ini_options"]["testpaths"]:
        assert not path.startswith("mceg") and "model_contracts" not in path
    for path in (ROOT / "src/exojax").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not {name.split(".")[0] for name in names} & {"mceg", "planetlib"}, path


def test_native_inspection_never_imports_exojax_or_jax():
    script = r'''
import importlib.abc
from pathlib import Path
import sys

class BlockProvider(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'exojax', 'jax', 'jaxlib'}:
            raise AssertionError('Inspection attempted to import ' + fullname)

sys.meta_path.insert(0, BlockProvider())
from mceg import load_contract
from mceg.check import check_contract
from mceg.coverage import Inventory, check_coverage
from mceg.validator import read_document

root = Path(sys.argv[1])
inventory = Inventory.model_validate(read_document(root / 'mceg/models.yaml'))
check_coverage(root, inventory)
for path in (root / 'mceg/contracts').glob('*.yaml'):
    contract = load_contract(path)
    for assessment in contract.assessments:
        check_contract(path, {}, assessment.id, root=root)
assert not {'exojax', 'jax', 'jaxlib'} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT)],
                            cwd=ROOT.parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
