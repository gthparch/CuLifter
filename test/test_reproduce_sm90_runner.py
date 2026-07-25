from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from reproduce import run_cpu_sm90 as runner


MANIFEST_PATH = Path(runner.__file__).with_name(runner.MANIFEST_NAME)


@pytest.fixture(scope="module")
def real_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def test_real_manifest_matches_published_invariants(real_manifest: dict) -> None:
    units = runner.validate_manifest_structure(real_manifest)

    assert real_manifest["schema_version"] == runner.MANIFEST_SCHEMA_VERSION
    assert real_manifest["totals"] == dict(runner.PUBLISHED_TOTALS)
    assert len(units) == 650
    assert Counter(
        unit["expected_sass_function_count"] == 0 for unit in units
    ) == {False: 645, True: 5}
    assert sum(
        unit["expected_sass_function_count"] for unit in units
    ) == 13227
    assert sum(
        1 if unit["symbol"] is not None else unit["expected_sass_function_count"]
        for unit in units
    ) == 11977

    flash_attention = [
        unit for unit in units if unit["suite"] == "flash_attention"
    ]
    assert len(flash_attention) == 63
    assert all(unit["symbol"] and unit["symbol_file"] for unit in flash_attention)
    assert all(unit["sha256"]["symbol_file"] for unit in flash_attention)


def test_manifest_rejects_total_drift(real_manifest: dict) -> None:
    mutated = copy.deepcopy(real_manifest)
    mutated["totals"]["paper_units"] -= 1

    with pytest.raises(runner.ReproductionError, match="totals disagree"):
        runner.validate_manifest_structure(mutated)


def test_manifest_rejects_suite_count_drift(real_manifest: dict) -> None:
    mutated = copy.deepcopy(real_manifest)
    mutated["suites"]["cublas"]["stored_units"] -= 1

    with pytest.raises(
        runner.ReproductionError, match="differs from the published invariant"
    ):
        runner.validate_manifest_structure(mutated)


def test_manifest_accepts_any_pinned_empty_unit_without_named_exception(
    real_manifest: dict,
) -> None:
    mutated = copy.deepcopy(real_manifest)
    empty_unit = next(
        unit
        for unit in mutated["units"]
        if unit["expected_sass_function_count"] == 0
    )
    empty_unit["name"] += "-renamed"

    units = runner.validate_manifest_structure(mutated)

    assert empty_unit in units


@pytest.fixture
def tiny_cuda_sdk(tmp_path: Path) -> tuple[Path, dict, list[dict], Path, Path]:
    sass = tmp_path / "cuda_sdk" / "sass" / "sample.sass"
    binary = tmp_path / "cuda_sdk" / "elf" / "sample.elf"
    sass.parent.mkdir(parents=True)
    binary.parent.mkdir(parents=True)
    sass.write_text("Function : sample_kernel\n")
    binary.write_bytes(b"synthetic ELF metadata")

    unit = {
        "suite": "cuda_sdk",
        "name": "sample",
        "lifter": "x86",
        "symbol": None,
        "symbol_file": None,
        "sass": "cuda_sdk/sass/sample.sass",
        "binary": "cuda_sdk/elf/sample.elf",
        "expected_sass_function_count": 1,
        "sha256": {
            "sass": runner.file_sha256(sass),
            "binary": runner.file_sha256(binary),
            "symbol_file": None,
        },
    }
    metadata = {
        "paper_units": 1,
        "paper_kernels": 1,
        "shipped_sass_functions": 1,
        "required_llvm_definitions": 1,
    }
    return tmp_path, metadata, [unit], sass, binary


def test_tiny_inventory_and_hashes_pass(
    tiny_cuda_sdk: tuple[Path, dict, list[dict], Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark, metadata, units, _, _ = tiny_cuda_sdk

    runner.preflight_suite("cuda_sdk", metadata, units, benchmark)

    assert "1 stored" in capsys.readouterr().out


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_preflight_rejects_extra_or_missing_inventory(
    tiny_cuda_sdk: tuple[Path, dict, list[dict], Path, Path],
    mutation: str,
) -> None:
    benchmark, metadata, units, sass, binary = tiny_cuda_sdk
    if mutation == "extra":
        (sass.parent / "extra.sass").write_text("Function : extra_kernel\n")
        (binary.parent / "extra.elf").write_bytes(b"extra metadata")
    else:
        sass.unlink()
        binary.unlink()

    with pytest.raises(
        runner.ReproductionError, match="bundle/manifest inventory differs"
    ):
        runner.preflight_suite("cuda_sdk", metadata, units, benchmark)


def test_preflight_rejects_content_hash_mismatch(
    tiny_cuda_sdk: tuple[Path, dict, list[dict], Path, Path],
) -> None:
    benchmark, metadata, units, sass, _ = tiny_cuda_sdk
    sass.write_text("Function : sample_kernel\n// content drift\n")

    with pytest.raises(runner.ReproductionError, match="SASS SHA-256 differs"):
        runner.preflight_suite("cuda_sdk", metadata, units, benchmark)


def test_invalid_llvm_is_rejected() -> None:
    with pytest.raises(RuntimeError):
        runner.llvm_definitions("define void @broken(")


def test_empty_sass_unit_succeeds_without_launching_lifter(tmp_path: Path) -> None:
    unit = {
        "suite": "hecbench",
        "name": "empty",
        "lifter": "x86",
        "symbol": None,
        "symbol_file": None,
        "sass": "hecbench/empty.sass",
        "binary": "hecbench/cubins/empty.cubin",
        "expected_sass_function_count": 0,
        "sha256": {"sass": "0" * 64, "binary": "0" * 64, "symbol_file": None},
    }

    result = runner.run_unit(unit, str(tmp_path), str(tmp_path), str(tmp_path))

    assert result == runner.LiftResult("hecbench", "empty", True, "OK", 0)


def test_missing_expected_llvm_definitions_are_reported() -> None:
    definitions = runner.llvm_definitions(
        "define void @present() { ret void }\n"
        "declare void @declaration_only()\n"
    )

    assert definitions == {"present"}
    assert runner.missing_expected_definitions(
        {"present", "declaration_only", "absent"}, definitions
    ) == ["absent", "declaration_only"]


def test_full_mode_reports_paper_claim(real_manifest: dict) -> None:
    full = runner.final_paper_claim(
        suite="all",
        total_failures=0,
        total_kernels=1681,
        total_passed_units=650,
        total_validated_definitions=11977,
        document=real_manifest,
    )

    assert full == (
        "Published lifting result: 1681/1681 kernels PASS "
        "(650/650 units; 11977/11977 required LLVM definitions)"
    )
    assert runner.suite_paper_result(
        failed_units=0, paper_kernels_passed=229
    ) == "229/229 PASS"
    assert runner.suite_paper_result(
        failed_units=1, paper_kernels_passed=""
    ) == "NOT REPRODUCED"
