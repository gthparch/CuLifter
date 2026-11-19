#!/usr/bin/env python3
"""Canonical CPU-only reproduction runner for the MICRO'26 sm_90 bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable


PUBLISHED_SUITES = MappingProxyType(
    {
        "hecbench": (229, 229, 770, 770),
        "cublas": (177, 972, 4137, 4137),
        "cutlass": (105, 105, 641, 641),
        "flash_attention": (63, 63, 1313, 63),
        "cudnn_cnn": (7, 7, 46, 46),
        "cudnn_ops": (20, 256, 2283, 2283),
        "cudnn_adv": (30, 30, 4018, 4018),
        "cuda_sdk": (19, 19, 19, 19),
    }
)
PUBLISHED_TOTALS = MappingProxyType(
    {
        "stored_units": 650,
        "paper_units": 650,
        "paper_kernels": 1681,
        "shipped_sass_functions": 13227,
        "required_llvm_definitions": 11977,
    }
)
SUITE_ORDER = tuple(PUBLISHED_SUITES)
MANIFEST_NAME = "sm90_manifest.json"
MANIFEST_SCHEMA_VERSION = 3
FUNCTION_RE = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$", re.MULTILINE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIB = 1024**3
FULL_RESULT_SPACE_BYTES = 12 * GIB
UNIT_KEYS = {
    "suite",
    "name",
    "lifter",
    "symbol",
    "symbol_file",
    "sass",
    "binary",
    "expected_sass_function_count",
    "sha256",
}


class ReproductionError(RuntimeError):
    """A fail-closed benchmark or validation error."""


@dataclass(frozen=True)
class LiftResult:
    suite: str
    name: str
    ok: bool
    status: str
    expected_definitions: int
    detail: str = ""


def fail(message: str) -> None:
    raise ReproductionError(message)


def parse_sass_functions(text: str) -> list[str]:
    names = FUNCTION_RE.findall(text)
    if len(names) != len(set(names)):
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        fail(f"duplicate SASS function declarations: {', '.join(duplicates[:5])}")
    return names


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_file(benchmark: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        fail(f"{label} is not a safe relative path: {value!r}")
    path = benchmark.joinpath(*relative.parts)
    try:
        path.resolve(strict=False).relative_to(benchmark.resolve())
    except ValueError:
        fail(f"{label} escapes the benchmark root: {value!r}")
    return path


def index_paths(paths: Iterable[Path], name_of: Any, label: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(paths):
        name = name_of(path)
        if not name or name in indexed:
            fail(f"{label} has a duplicate or empty unit name: {name!r}")
        indexed[name] = path
    return indexed


def paired_inventory(
    suite: str, benchmark: Path
) -> dict[str, tuple[str, str]]:
    root = benchmark / suite
    if suite == "hecbench":
        sass = index_paths(root.glob("*.sass"), lambda path: path.stem, f"{suite} SASS")
        binary = index_paths(
            (root / "cubins").glob("*.cubin"),
            lambda path: path.name.removesuffix(".cubin"),
            f"{suite} binaries",
        )
    elif suite in {"cublas", "cudnn_cnn", "cudnn_ops", "cudnn_adv"}:
        sass = index_paths(
            root.glob("*.cubin.sass"),
            lambda path: path.name.removesuffix(".sass"),
            f"{suite} SASS",
        )
        binary = index_paths(
            (root / "cubins").glob("*.cubin"),
            lambda path: path.name,
            f"{suite} binaries",
        )
    elif suite == "cutlass":
        sass = index_paths(
            root.glob("*/*_sm90.sass"),
            lambda path: path.parent.name,
            f"{suite} SASS",
        )
        binary = index_paths(
            root.glob("*/*_sm90.elf"),
            lambda path: path.parent.name,
            f"{suite} binaries",
        )
        for name, path in sass.items():
            if path.name != f"{name}_sm90.sass":
                fail(f"{suite}/{name}: non-canonical SASS filename {path.name!r}")
        for name, path in binary.items():
            if path.name != f"{name}_sm90.elf":
                fail(f"{suite}/{name}: non-canonical ELF filename {path.name!r}")
    elif suite == "flash_attention":
        selection = root / "THE_63_KERNELS.txt"
        if not selection.is_file():
            fail(f"{suite}: missing {selection.name}")
        selected = [line.strip() for line in selection.read_text().splitlines() if line.strip()]
        if len(selected) != len(set(selected)):
            fail(f"{suite}: {selection.name} contains duplicate names")
        sass = index_paths(
            root.glob("*/kernel.sass"),
            lambda path: path.parent.name,
            f"{suite} SASS",
        )
        binary = index_paths(
            root.glob("*/kernel.cubin"),
            lambda path: path.parent.name,
            f"{suite} binaries",
        )
        symbol_files = index_paths(
            root.glob("*/kname.txt"),
            lambda path: path.parent.name,
            f"{suite} symbol selectors",
        )
        if set(selected) != set(sass) or set(selected) != set(symbol_files):
            missing = sorted(set(selected) - set(sass) | (set(selected) - set(symbol_files)))
            extra = sorted((set(sass) | set(symbol_files)) - set(selected))
            fail(
                f"{suite}: selection/SASS/symbol inventory differs "
                f"(missing={missing[:5]}, extra={extra[:5]})"
            )
    elif suite == "cuda_sdk":
        sass = index_paths(
            (root / "sass").glob("*.sass"),
            lambda path: path.stem,
            f"{suite} SASS",
        )
        binary = index_paths(
            (root / "elf").glob("*.elf"),
            lambda path: path.name.removesuffix(".elf"),
            f"{suite} binaries",
        )
    else:
        fail(f"unsupported suite in manifest: {suite!r}")

    if set(sass) != set(binary):
        missing_binary = sorted(set(sass) - set(binary))
        missing_sass = sorted(set(binary) - set(sass))
        fail(
            f"{suite}: SASS/binary inventory differs "
            f"(without binary={missing_binary[:5]}, without SASS={missing_sass[:5]})"
        )
    return {
        name: (
            sass[name].relative_to(benchmark).as_posix(),
            binary[name].relative_to(benchmark).as_posix(),
        )
        for name in sorted(sass)
    }


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError:
        fail(f"checked-in input manifest is missing: {path}")
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read input manifest {path}: {error}")
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != MANIFEST_SCHEMA_VERSION
    ):
        fail(f"{path}: unsupported manifest schema")
    return document


def validate_manifest_structure(document: dict[str, Any]) -> list[dict[str, Any]]:
    suites = document.get("suites")
    totals = document.get("totals")
    units = document.get("units")
    if not isinstance(suites, dict) or tuple(suites) != SUITE_ORDER:
        fail(f"manifest suites must be exactly: {', '.join(SUITE_ORDER)}")
    if not isinstance(totals, dict) or not isinstance(units, list):
        fail("manifest totals/units have invalid types")

    seen_names: set[tuple[str, str]] = set()
    seen_sass: set[str] = set()
    seen_binary: set[str] = set()
    seen_symbol_files: set[str] = set()
    for index, unit in enumerate(units):
        label = f"manifest unit #{index + 1}"
        if not isinstance(unit, dict) or set(unit) != UNIT_KEYS:
            fail(f"{label} has inconsistent fields")
        suite = unit["suite"]
        name = unit["name"]
        if suite not in suites or not isinstance(name, str) or not name:
            fail(f"{label} has an invalid suite/name")
        key = (suite, name)
        if key in seen_names:
            fail(f"manifest repeats unit {suite}/{name}")
        seen_names.add(key)
        for field, seen in (("sass", seen_sass), ("binary", seen_binary)):
            value = unit[field]
            if not isinstance(value, str) or not value or value in seen:
                fail(f"{suite}/{name}: duplicate or invalid {field} path")
            seen.add(value)

        count = unit["expected_sass_function_count"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            fail(f"{suite}/{name}: invalid expected SASS function count")

        lifter = unit["lifter"]
        symbol = unit["symbol"]
        symbol_file = unit["symbol_file"]
        if suite == "flash_attention":
            if (
                lifter != "x86warp"
                or not isinstance(symbol, str)
                or not symbol
                or not isinstance(symbol_file, str)
                or not symbol_file
            ):
                fail(
                    f"{suite}/{name}: expected x86warp, one selected symbol, "
                    "and its selector file"
                )
            expected_symbol_file = f"flash_attention/{name}/kname.txt"
            if symbol_file != expected_symbol_file or symbol_file in seen_symbol_files:
                fail(f"{suite}/{name}: invalid or duplicate symbol selector path")
            seen_symbol_files.add(symbol_file)
        elif lifter != "x86" or symbol is not None or symbol_file is not None:
            fail(f"{suite}/{name}: expected x86 and no selected symbol/selector")
        hashes = unit["sha256"]
        if (
            not isinstance(hashes, dict)
            or set(hashes) != {"sass", "binary", "symbol_file"}
        ):
            fail(f"{suite}/{name}: invalid SHA-256 fields")
        if any(
            not isinstance(hashes[field], str)
            or not SHA256_RE.fullmatch(hashes[field])
            for field in ("sass", "binary")
        ):
            fail(f"{suite}/{name}: malformed SHA-256")
        selector_hash = hashes["symbol_file"]
        if suite == "flash_attention":
            if not isinstance(selector_hash, str) or not SHA256_RE.fullmatch(selector_hash):
                fail(f"{suite}/{name}: malformed selector SHA-256")
        elif selector_hash is not None:
            fail(f"{suite}/{name}: unexpected selector SHA-256")

    computed_stored = len(units)
    computed_kernels = 0
    computed_sass_functions = 0
    computed_required_definitions = 0
    for suite, metadata in suites.items():
        suite_units = [unit for unit in units if unit["suite"] == suite]
        (
            published_units,
            published_kernels,
            published_sass_functions,
            published_required_definitions,
        ) = PUBLISHED_SUITES[suite]
        expected_metadata = {
            "paper_kernels": published_kernels,
            "stored_units": published_units,
            "paper_units": published_units,
            "shipped_sass_functions": published_sass_functions,
            "required_llvm_definitions": published_required_definitions,
        }
        if metadata != expected_metadata:
            fail(
                f"{suite}: metadata differs from the published invariant; "
                f"expected {expected_metadata}"
            )
        if len(suite_units) != published_units:
            fail(
                f"{suite}: manifest has {len(suite_units)} jobs; "
                f"published invariant is {published_units}"
            )
        suite_sass_functions = sum(
            unit["expected_sass_function_count"] for unit in suite_units
        )
        suite_required_definitions = sum(
            1 if unit["symbol"] is not None else unit["expected_sass_function_count"]
            for unit in suite_units
        )
        if suite_sass_functions != published_sass_functions:
            fail(
                f"{suite}: manifest rows contain {suite_sass_functions} SASS functions; "
                f"published bundle invariant is {published_sass_functions}"
            )
        if suite_required_definitions != published_required_definitions:
            fail(
                f"{suite}: manifest rows require {suite_required_definitions} LLVM "
                f"definitions; bundle invariant is {published_required_definitions}"
            )
        computed_kernels += metadata["paper_kernels"]
        computed_sass_functions += suite_sass_functions
        computed_required_definitions += suite_required_definitions

    expected_totals = {
        "stored_units": computed_stored,
        "paper_units": computed_stored,
        "paper_kernels": computed_kernels,
        "shipped_sass_functions": computed_sass_functions,
        "required_llvm_definitions": computed_required_definitions,
    }
    if totals != expected_totals:
        fail(f"manifest totals disagree with rows/suites: expected {expected_totals}")
    if expected_totals != dict(PUBLISHED_TOTALS):
        fail(f"manifest does not encode the published 650/1,681 inventory: {expected_totals}")
    return units


def preflight_suite(
    suite: str,
    metadata: dict[str, Any],
    units: list[dict[str, Any]],
    benchmark: Path,
) -> None:
    discovered = paired_inventory(suite, benchmark)
    manifest_units = {unit["name"]: unit for unit in units if unit["suite"] == suite}
    if set(discovered) != set(manifest_units):
        missing = sorted(set(manifest_units) - set(discovered))
        extra = sorted(set(discovered) - set(manifest_units))
        fail(
            f"{suite}: bundle/manifest inventory differs "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )

    for name in sorted(manifest_units):
        unit = manifest_units[name]
        expected_paths = discovered[name]
        if (unit["sass"], unit["binary"]) != expected_paths:
            fail(
                f"{suite}/{name}: paths differ from bundle layout; "
                f"manifest={(unit['sass'], unit['binary'])}, bundle={expected_paths}"
            )
        sass = relative_file(benchmark, unit["sass"], f"{suite}/{name} SASS")
        binary = relative_file(benchmark, unit["binary"], f"{suite}/{name} binary")
        if not sass.is_file() or not binary.is_file():
            fail(f"{suite}/{name}: a pinned input file is missing")
        actual_sass_hash = file_sha256(sass)
        actual_binary_hash = file_sha256(binary)
        if actual_sass_hash != unit["sha256"]["sass"]:
            fail(f"{suite}/{name}: shipped SASS SHA-256 differs from the manifest")
        if actual_binary_hash != unit["sha256"]["binary"]:
            fail(f"{suite}/{name}: shipped binary SHA-256 differs from the manifest")

        functions = parse_sass_functions(sass.read_text(errors="replace"))
        if len(functions) != unit["expected_sass_function_count"]:
            fail(
                f"{suite}/{name}: found {len(functions)} SASS functions; "
                f"manifest expects {unit['expected_sass_function_count']}"
            )
        symbol = unit["symbol"]
        if symbol is not None and symbol not in functions:
            fail(f"{suite}/{name}: selected symbol {symbol!r} is absent from SASS")
        symbol_file = unit["symbol_file"]
        if symbol_file is not None:
            selector = relative_file(
                benchmark, symbol_file, f"{suite}/{name} symbol selector"
            )
            if not selector.is_file():
                fail(f"{suite}/{name}: pinned symbol selector is missing")
            if file_sha256(selector) != unit["sha256"]["symbol_file"]:
                fail(f"{suite}/{name}: symbol selector SHA-256 differs from the manifest")
            try:
                selector_lines = selector.read_text(errors="strict").splitlines()
            except (OSError, UnicodeError) as error:
                fail(f"{suite}/{name}: cannot read symbol selector: {error}")
            if selector_lines != [symbol]:
                fail(f"{suite}/{name}: symbol selector does not match the manifest")

    print(
        f"  {suite:<18} {len(manifest_units):>3} stored  "
        f"{metadata['paper_units']:>3} paper units  "
        f"{metadata['paper_kernels']:>4} paper kernels  "
        f"{metadata['required_llvm_definitions']:>5} required LLVM definitions  OK"
    )


def check_python_environment(root: Path) -> None:
    missing = []
    for module in ("elftools", "libcst", "llvmlite.binding", "yaml"):
        try:
            importlib.import_module(module)
        except Exception as error:  # report all broken imports together
            missing.append(f"{module}: {error}")
    if missing:
        fail(
            "Python environment is incomplete; recreate it from environment.yml\n  "
            + "\n  ".join(missing)
        )
    result = subprocess.run(
        [sys.executable, str(root / "main.py"), "--help"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        fail(f"main.py --help failed: {(detail[-1] if detail else 'unknown error')[:200]}")


def check_result_space(results_dir: Path) -> None:
    required = FULL_RESULT_SPACE_BYTES
    available = shutil.disk_usage(results_dir).free
    print(
        f"Result space: {available / GIB:.1f} GiB available; "
        f"{required / GIB:.0f} GiB required."
    )
    if available < required:
        fail(
            f"insufficient result space at {results_dir}: "
            f"{available / GIB:.1f} GiB available, {required / GIB:.0f} GiB required"
        )


def output_slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:150] or "unit"
    suffix = hashlib.sha256(name.encode()).hexdigest()[:10]
    return f"{cleaned}.{suffix}"


def tail_detail(path: Path) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return "no lift log"
    return (lines[-1] if lines else "no diagnostic")[:200]


def llvm_definitions(ir_text: str) -> set[str]:
    """Parse and verify LLVM IR, returning the names of defined functions."""
    import llvmlite.binding as llvm

    llvm_module = llvm.parse_assembly(ir_text)
    llvm_module.verify()
    return {
        function.name for function in llvm_module.functions if not function.is_declaration
    }


def missing_expected_definitions(
    expected: Iterable[str], definitions: Iterable[str]
) -> list[str]:
    """Return a deterministic list of expected functions absent from the IR."""
    return sorted(set(expected) - set(definitions))


def run_unit(
    unit: dict[str, Any],
    root_text: str,
    benchmark_text: str,
    results_text: str,
) -> LiftResult:
    suite = unit["suite"]
    name = unit["name"]
    expected_count = (
        1 if unit["symbol"] is not None else unit["expected_sass_function_count"]
    )
    if expected_count == 0:
        return LiftResult(suite, name, True, "OK", 0)

    root = Path(root_text)
    benchmark = Path(benchmark_text)
    results = Path(results_text)
    stem = results / f"{suite}.{output_slug(name)}"
    ll_path = Path(f"{stem}.ll")
    lift_log = Path(f"{stem}.lift.log")
    validation_log = Path(f"{stem}.validation.log")
    for generated in (ll_path, lift_log, validation_log):
        generated.unlink(missing_ok=True)

    command = [
        sys.executable,
        str(root / "main.py"),
        "-i",
        str(benchmark / PurePosixPath(unit["sass"])),
        str(benchmark / PurePosixPath(unit["binary"])),
        "-o",
        str(stem),
        "--lifter",
        unit["lifter"],
    ]
    if unit["symbol"] is not None:
        command.extend(["-name", unit["symbol"]])

    with lift_log.open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        return LiftResult(
            suite, name, False, "LIFT-FAIL", expected_count, tail_detail(lift_log)
        )
    if not ll_path.is_file() or ll_path.stat().st_size == 0:
        return LiftResult(
            suite,
            name,
            False,
            "LIFT-EMPTY",
            expected_count,
            "main.py emitted no LLVM IR",
        )

    try:
        definitions = llvm_definitions(ll_path.read_text(errors="strict"))
    except Exception as error:
        validation_log.write_text(f"LLVM parse+verify: FAIL\n{type(error).__name__}: {error}\n")
        return LiftResult(
            suite,
            name,
            False,
            "LLVM-INVALID",
            expected_count,
            str(error).splitlines()[0][:200],
        )

    if unit["symbol"] is not None:
        expected = {unit["symbol"]}
    else:
        sass = benchmark / PurePosixPath(unit["sass"])
        expected = set(parse_sass_functions(sass.read_text(errors="replace")))
    missing = missing_expected_definitions(expected, definitions)
    if missing:
        validation_log.write_text(
            "LLVM parse+verify: PASS\n"
            f"expected definitions: {len(expected)}\n"
            f"emitted definitions: {len(definitions)}\n"
            "missing definitions:\n"
            + "\n".join(missing)
            + "\n"
        )
        detail = ", ".join(missing[:3])
        if len(missing) > 3:
            detail += f", ... ({len(missing)} missing)"
        return LiftResult(
            suite, name, False, "SYMBOL-MISSING", expected_count, detail
        )

    validation_log.write_text(
        "LLVM parse+verify: PASS\n"
        f"expected definitions: {len(expected)}\n"
        f"emitted definitions: {len(definitions)}\n"
        "all expected definitions emitted: PASS\n"
    )
    return LiftResult(suite, name, True, "OK", expected_count)


def execute_suite(
    suite_units: list[dict[str, Any]],
    workers: int,
    root: Path,
    benchmark: Path,
    results: Path,
) -> list[LiftResult]:
    if workers == 1:
        results_list = [
            run_unit(unit, str(root), str(benchmark), str(results)) for unit in suite_units
        ]
    else:
        results_list = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_unit, unit, str(root), str(benchmark), str(results)
                ): unit
                for unit in suite_units
            }
            for future in as_completed(futures):
                unit = futures[future]
                try:
                    results_list.append(future.result())
                except Exception as error:
                    results_list.append(
                        LiftResult(
                            unit["suite"],
                            unit["name"],
                            False,
                            "RUNNER-FAIL",
                            (
                                1
                                if unit["symbol"] is not None
                                else unit["expected_sass_function_count"]
                            ),
                            f"{type(error).__name__}: {error}"[:200],
                        )
                    )

    for result in sorted(results_list, key=lambda item: item.name):
        suffix = f"\t{result.detail}" if result.detail else ""
        print(f"[{result.status}]\t{result.suite}\t{result.name}{suffix}")
    return results_list




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lift and independently validate the manifest-pinned MICRO'26 sm_90 "
            "benchmark inputs without a GPU."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the exact bundle inventory and content without lifting",
    )
    parser.add_argument(
        "--suite",
        choices=("all",) + SUITE_ORDER,
        default="all",
        help="evaluate one suite (default: all)",
    )
    default_jobs = max(1, min(8, (os.cpu_count() or 1) // 2))
    parser.add_argument("--jobs", type=int, default=default_jobs, help=f"workers (default: {default_jobs})")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    return args


def suite_paper_result(*, failed_units: int, paper_kernels_passed: int | str) -> str:
    if failed_units or paper_kernels_passed == "":
        return "NOT REPRODUCED"
    return f"{paper_kernels_passed}/{paper_kernels_passed} PASS"


def final_paper_claim(
    *,
    suite: str,
    total_failures: int,
    total_kernels: int,
    total_passed_units: int,
    total_validated_definitions: int,
    document: dict[str, Any],
) -> str:
    totals = document["totals"]
    verified = (total_kernels, total_passed_units, total_validated_definitions)
    expected = (totals["paper_kernels"], totals["paper_units"], totals["required_llvm_definitions"])
    if suite != "all" or total_failures or verified != expected:
        return ""
    return (
        f"Published lifting result: {total_kernels}/{totals['paper_kernels']} kernels PASS "
        f"({total_passed_units}/{totals['paper_units']} units; "
        f"{total_validated_definitions}/{totals['required_llvm_definitions']} required LLVM definitions)"
    )


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    benchmark = Path(os.environ.get("SLIFTER_BENCHMARK_DIR", root / "benchmark")).resolve()
    manifest_path = Path(__file__).resolve().with_name(MANIFEST_NAME)
    if not benchmark.is_dir():
        fail(
            f"benchmark bundle not found at {benchmark}; "
            "install it with reproduce/setup_data.sh"
        )

    document = load_manifest(manifest_path)
    units = validate_manifest_structure(document)
    selected_suites = SUITE_ORDER if args.suite == "all" else (args.suite,)

    print(f"Checking benchmark bundle: {benchmark}")
    for suite in selected_suites:
        preflight_suite(suite, document["suites"][suite], units, benchmark)
    selected_stored = sum(document["suites"][suite]["stored_units"] for suite in selected_suites)
    selected_paper = sum(document["suites"][suite]["paper_units"] for suite in selected_suites)
    selected_definitions = sum(
        document["suites"][suite]["required_llvm_definitions"]
        for suite in selected_suites
    )
    print(
        f"Bundle preflight PASS: {selected_stored} stored units, "
        f"{selected_paper} paper units, {selected_definitions} required LLVM definitions."
    )
    if args.check:
        print("Benchmark bundle OK. No lifting result is claimed by --check.")
        return 0

    check_python_environment(root)
    configured_results = os.environ.get("SLIFTER_RESULTS_DIR")
    if configured_results:
        results_dir = Path(configured_results).resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = root / "reproduce" / "results" / f"lift_{stamp}_{os.getpid()}"
    results_dir.mkdir(parents=True, exist_ok=True)
    check_result_space(results_dir)
    print(f"results -> {results_dir}   (jobs={args.jobs})")

    summary_rows = []
    total_failures = 0
    total_kernels = 0
    total_passed_units = 0
    total_validated_definitions = 0
    for suite in selected_suites:
        metadata = document["suites"][suite]
        paper_units = [unit for unit in units if unit["suite"] == suite]
        attempted_units = paper_units
        if not attempted_units:
            fail(f"{suite}: no evaluation units selected")
        workers = min(args.jobs, len(attempted_units))
        if suite == "flash_attention":
            try:
                fa_jobs = int(os.environ.get("FA_JOBS", "2"))
            except ValueError:
                fail("FA_JOBS must be a positive integer")
            if fa_jobs < 1:
                fail("FA_JOBS must be a positive integer")
            workers = min(workers, fa_jobs)

        print(f"\n== {suite} ==")
        suite_results = execute_suite(
            attempted_units, workers, root, benchmark, results_dir
        )
        passed = sum(result.ok for result in suite_results)
        failures = len(suite_results) - passed
        checked_definitions = sum(
            result.expected_definitions for result in suite_results
        )
        passed_definitions = sum(
            result.expected_definitions for result in suite_results if result.ok
        )
        total_failures += failures
        paper_kernels_passed: int | str = ""
        if (
            failures == 0
            and passed == metadata["stored_units"]
            and passed_definitions == metadata["required_llvm_definitions"]
        ):
            paper_kernels_passed = metadata["paper_kernels"]
            total_kernels += metadata["paper_kernels"]
        total_passed_units += passed
        total_validated_definitions += passed_definitions
        summary_rows.append(
            {
                "suite": suite,
                "paper_kernels": metadata["paper_kernels"],
                "paper_units": metadata["paper_units"],
                "shipped_sass_functions": metadata["shipped_sass_functions"],
                "required_llvm_definitions": metadata["required_llvm_definitions"],
                "checked_units": len(suite_results),
                "passed_units": passed,
                "failed_units": failures,
                "checked_definitions": checked_definitions,
                "passed_definitions": passed_definitions,
                "paper_kernels_passed": paper_kernels_passed,
            }
        )
        print(f"----- {suite}: {passed}/{len(suite_results)} checked units passed -----")

    csv_path = results_dir / "summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n==================== SUMMARY (CPU backend, sm90) ====================")
    print(
        f"{'suite':<18} {'checked units':>15} "
        f"{'LLVM definitions':>18} {'paper kernels':>16}"
    )
    for row in summary_rows:
        checked = f"{row['passed_units']}/{row['checked_units']}"
        definitions = (
            f"{row['passed_definitions']}/{row['checked_definitions']}"
        )
        paper_result = suite_paper_result(
            failed_units=row["failed_units"],
            paper_kernels_passed=row["paper_kernels_passed"],
        )
        print(
            f"{row['suite']:<18} {checked:>15} "
            f"{definitions:>18} {paper_result:>16}"
        )

    print()
    claim = final_paper_claim(
        suite=args.suite,
        total_failures=total_failures,
        total_kernels=total_kernels,
        total_passed_units=total_passed_units,
        total_validated_definitions=total_validated_definitions,
        document=document,
    )
    if claim:
        print(claim)
    print(
        f"units lifted and verified: {total_passed_units}; "
        f"LLVM definitions emitted: {total_validated_definitions}"
    )
    print(f"per-artifact LLVM IR, logs, and summary.csv: {results_dir}")
    print(f"total failures: {total_failures}")
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReproductionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
