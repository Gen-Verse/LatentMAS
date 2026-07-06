#!/usr/bin/env python
"""Firewall CI check for the dual-track codebase separation (strategy.md §6).

Rules enforced (hard failures unless noted):

Rule 1 — SVD-free hub (``src/latent_coordination/``):
    * no ``torch.linalg.svd`` / ``numpy.linalg.svd`` / ``scipy.linalg.svd``
      imports or calls (``svdvals`` — a scalar-spectrum diagnostic with no
      U/V decomposition and no projection operator — is explicitly allowed);
    * no orthogonal-projection operator identifiers (``U_L``, ``U_R``,
      ``P_L``, ``P_s``);
    * no imports from ``mechanistic_disentangle``.

Rule 2 — Recurrent exclusion (``src/mechanistic_disentangle/``):
    * no CVAE definitions/imports, no agent-role attention router, no
      multi-agent dispatch/topology structures;
    * no imports from ``latent_coordination``.

Rule 3 — Citation-only reuse of Paper 1 (soft: logged, not fatal):
    * functions in ``latent_coordination`` whose names/docstrings reproduce
      CLAP / Logit-Lens / Stage-2 anchoring math are flagged to the audit log.

Rule 4 — Terminology discipline:
    * ``latent_coordination`` must not label its hub mechanism with Paper 2
      terminology (``MLRS``, ``disentangle``, ``U_L/U_R``) outside comments
      explicitly citing Paper 2 for contrast.

Every run (pass or fail) appends a timestamped entry to
``ARTIFACTS/firewall_audit_log.md``.

Usage:
    python scripts/firewall_check.py [--target latent_coordination|mechanistic_disentangle|all]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
AUDIT_LOG = REPO_ROOT / "ARTIFACTS" / "firewall_audit_log.md"

# Rule 1 patterns ------------------------------------------------------------
SVD_CALL = re.compile(r"\blinalg\.svd\s*\(|\blinalg\.svd\b(?!vals)")
PROJECTION_IDENTIFIERS = {"U_L", "U_R", "P_L", "P_s"}
FORBIDDEN_IMPORT_LC = "mechanistic_disentangle"

# Rule 2 patterns ------------------------------------------------------------
FORBIDDEN_IMPORT_MD = "latent_coordination"
CVAE_PATTERN = re.compile(r"\bcvae\b|\bCVAE\b|\bq_phi\b|\bp_theta\b")
ROUTER_PATTERN = re.compile(r"\bAttentionRouter\b|\bAdaptiveOrchestrator\b|\bRoutingPlan\b")

# Rule 3 patterns (soft) -----------------------------------------------------
PAPER1_MATH = re.compile(r"logit[\s_-]?lens|stage[\s_-]?2 anchoring|CLAP computation", re.IGNORECASE)

# Rule 4 patterns ------------------------------------------------------------
TERMINOLOGY = re.compile(r"\bMLRS\b|\bdisentanglement\b|\bU_L/U_R\b")
CITATION_MARKERS = re.compile(r"paper\s*2|mechanistic_disentangle|strategy\.md|firewall", re.IGNORECASE)


def _py_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _imports_of(tree: ast.AST) -> List[str]:
    mods: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
    return mods


def _identifiers_of(tree: ast.AST) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def check_latent_coordination() -> Tuple[List[str], List[str]]:
    """Returns (violations, soft_flags) for Rule 1 + 3 + 4."""
    violations: List[str] = []
    soft_flags: List[str] = []
    root = SRC / "latent_coordination"
    for path in _py_files(root):
        rel = path.relative_to(REPO_ROOT)
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            violations.append(f"{rel}: unparseable ({exc})")
            continue

        # Rule 1a: SVD usage (svdvals allowed — see module docstring).
        for i, line in enumerate(text.splitlines(), 1):
            if SVD_CALL.search(line):
                violations.append(f"{rel}:{i}: forbidden SVD decomposition usage: {line.strip()}")

        # Rule 1b: projection-operator identifiers.
        bad_ids = PROJECTION_IDENTIFIERS & _identifiers_of(tree)
        if bad_ids:
            violations.append(f"{rel}: projection-operator identifier(s) {sorted(bad_ids)}")

        # Rule 1c: forbidden cross-package imports.
        for mod in _imports_of(tree):
            if mod.split(".")[0] == FORBIDDEN_IMPORT_LC:
                violations.append(f"{rel}: imports '{mod}' (firewall Rule 1)")

        # Rule 3 (soft): re-derivations of Paper 1 math.
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node) or ""
                if PAPER1_MATH.search(doc):
                    soft_flags.append(
                        f"{rel}: function '{node.name}' docstring references Paper 1 "
                        "diagnostic math — verify it cites rather than re-derives"
                    )

        # Rule 4: Paper 2 terminology outside citing comments.
        for i, line in enumerate(text.splitlines(), 1):
            if TERMINOLOGY.search(line) and not CITATION_MARKERS.search(line):
                violations.append(
                    f"{rel}:{i}: Paper 2 terminology without citation marker: {line.strip()[:100]}"
                )
    return violations, soft_flags


def check_mechanistic_disentangle() -> Tuple[List[str], List[str]]:
    """Returns (violations, soft_flags) for Rule 2."""
    violations: List[str] = []
    root = SRC / "mechanistic_disentangle"
    if not root.exists():
        return [f"{root} does not exist — the firewall requires it"], []
    for path in _py_files(root):
        rel = path.relative_to(REPO_ROOT)
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            violations.append(f"{rel}: unparseable ({exc})")
            continue

        for mod in _imports_of(tree):
            if mod.split(".")[0] == FORBIDDEN_IMPORT_MD:
                violations.append(f"{rel}: imports '{mod}' (firewall Rule 2)")

        # CVAE / multi-agent router definitions are latent_coordination-only.
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and CVAE_PATTERN.search(node.name):
                violations.append(f"{rel}: defines CVAE-like class '{node.name}' (firewall Rule 2)")
        for i, line in enumerate(text.splitlines(), 1):
            if ROUTER_PATTERN.search(line):
                violations.append(f"{rel}:{i}: references multi-agent router machinery: {line.strip()[:100]}")
    return violations, []


def append_audit_entry(target: str, violations: List[str], soft_flags: List[str]) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = "PASS" if not violations else "FAIL"
    lines = [f"\n## {ts} — target={target} — **{status}**\n"]
    if violations:
        lines.append("Violations:\n")
        lines.extend(f"- {v}\n" for v in violations)
    if soft_flags:
        lines.append("Soft flags (Rule 3, review-only):\n")
        lines.extend(f"- {f}\n" for f in soft_flags)
    if not violations and not soft_flags:
        lines.append("No violations, no soft flags.\n")
    if not AUDIT_LOG.exists():
        header = "# Firewall audit log (append-only)\n\nEvery `scripts/firewall_check` run appends here (strategy.md §6).\n"
        AUDIT_LOG.write_text(header, encoding="utf-8")
    with AUDIT_LOG.open("a", encoding="utf-8") as fh:
        fh.writelines(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["latent_coordination", "mechanistic_disentangle", "all"],
        default="all",
    )
    args = parser.parse_args()

    violations: List[str] = []
    soft_flags: List[str] = []
    if args.target in ("latent_coordination", "all"):
        v, s = check_latent_coordination()
        violations += v
        soft_flags += s
    if args.target in ("mechanistic_disentangle", "all"):
        v, s = check_mechanistic_disentangle()
        violations += v
        soft_flags += s

    append_audit_entry(args.target, violations, soft_flags)

    for f in soft_flags:
        print(f"[SOFT] {f}")
    if violations:
        for v in violations:
            print(f"[FAIL] {v}", file=sys.stderr)
        print(f"\nFirewall check FAILED with {len(violations)} violation(s).", file=sys.stderr)
        return 1
    print(f"Firewall check PASSED (target={args.target}, {len(soft_flags)} soft flag(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
