"""Build and verify ``BASELINE_MANIFEST.json`` (PLAN.md M0).

The manifest pins four classes of input:

``git_repo``    a vendored reference checkout, pinned by commit + worktree digest
``content_tree``a directory pinned by structure and/or content digest
``file_set``    explicit files pinned by sha256 (engine version, .umap payloads)
``tool_env``    the interpreter/package set used to produce milestone evidence

Verification recomputes every digest and reports per-assertion expected vs
observed values, which is the shape PLAN.md's verification standard requires.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from embodiedbench.artifacts.hashing import TreeDigest, digest_tree, sha256_file

SCHEMA_ID = "embodiedbench/baseline_manifest/v0.1"
ROOTS_SCHEMA_ID = "embodiedbench/baseline_roots/v0.1"

EntryKind = Literal["git_repo", "content_tree", "file_set", "tool_env"]

# Never contribute to a digest: caches and VCS internals are not inputs.
DEFAULT_EXCLUDE_DIRS = (".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache")


# ─────────────────────────────────────────────────────────────────────────────
# Symbolic roots
# ─────────────────────────────────────────────────────────────────────────────


def load_roots(path: Path) -> dict[str, Path]:
    """Load the untracked host mapping from symbolic root to absolute path."""
    data = json.loads(Path(path).read_text())
    if data.get("schema") != ROOTS_SCHEMA_ID:
        raise ValueError(f"{path}: expected schema {ROOTS_SCHEMA_ID}, got {data.get('schema')!r}")
    return {name: Path(p) for name, p in data["roots"].items()}


def resolve(root: str, relpath: str, roots: dict[str, Path]) -> Path:
    """Resolve ``$ROOT`` + relpath against the host root mapping."""
    if root not in roots:
        raise KeyError(
            f"symbolic root {root!r} is not defined in baseline_roots.local.json "
            f"(defined: {sorted(roots)})"
        )
    base = roots[root]
    return base if relpath in ("", ".") else base / relpath


# ─────────────────────────────────────────────────────────────────────────────
# Manifest model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ManifestEntry:
    """One pinned baseline input."""

    id: str
    kind: EntryKind
    role: str
    root: str = ""
    relpath: str = ""
    # git_repo
    remote: str | None = None
    branch: str | None = None
    commit: str | None = None
    worktree_dirty: bool | None = None
    # content_tree
    digests: list[dict[str, Any]] = field(default_factory=list)
    # file_set / pinned files inside a tree
    files: list[dict[str, Any]] = field(default_factory=list)
    # tool_env
    env: dict[str, Any] = field(default_factory=dict)
    # governance
    license: str = "unknown"
    publishable: bool = False
    access_note: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "kind": self.kind, "role": self.role}
        if self.root:
            out["root"] = self.root
            out["relpath"] = self.relpath
        for name in ("remote", "branch", "commit", "worktree_dirty"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        if self.digests:
            out["digests"] = self.digests
        if self.files:
            out["files"] = self.files
        if self.env:
            out["env"] = self.env
        out["license"] = self.license
        out["publishable"] = self.publishable
        if self.access_note:
            out["access_note"] = self.access_note
        if self.notes:
            out["notes"] = self.notes
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestEntry":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class BaselineManifest:
    schema: str
    generated_at: str
    generator: dict[str, Any]
    entries: list[ManifestEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "generated_at": self.generated_at,
            "generator": self.generator,
            "entries": [e.to_dict() for e in self.entries],
        }

    def entry(self, entry_id: str) -> ManifestEntry:
        for e in self.entries:
            if e.id == entry_id:
                return e
        raise KeyError(entry_id)


def load_manifest(path: Path) -> BaselineManifest:
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA_ID:
        raise ValueError(f"{path}: expected schema {SCHEMA_ID}, got {data.get('schema')!r}")
    return BaselineManifest(
        schema=data["schema"],
        generated_at=data["generated_at"],
        generator=data["generator"],
        entries=[ManifestEntry.from_dict(e) for e in data["entries"]],
    )


def write_manifest(manifest: BaselineManifest, path: Path) -> None:
    Path(path).write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# git helpers
# ─────────────────────────────────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} in {repo} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def git_state(repo: Path) -> dict[str, Any]:
    """Commit, branch, remote, and dirty flag for a checkout.

    ``worktree_dirty`` is recorded rather than rejected: PLAN.md M0 requires that
    dirty user changes in existing repositories remain untouched, so the freeze
    must describe reality instead of demanding a clean tree.
    """
    return {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "remote": _git(repo, "config", "--get", "remote.origin.url") or None,
        "worktree_dirty": bool(_git(repo, "status", "--porcelain")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────


def _tree_digests(path: Path, levels: Iterable[str]) -> list[dict[str, Any]]:
    return [
        digest_tree(path, level=level, exclude_dirs=DEFAULT_EXCLUDE_DIRS).to_dict()
        for level in levels
    ]


def _file_digests(base: Path, relpaths: Iterable[str]) -> list[dict[str, Any]]:
    out = []
    for rel in relpaths:
        full = base / rel
        out.append(
            {"relpath": rel, "size": full.stat().st_size, "sha256": sha256_file(full)}
        )
    return out


def tool_env_record() -> dict[str, Any]:
    """Interpreter and package versions used to produce milestone evidence."""
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in (
        "pydantic",
        "gymnasium",
        "numpy",
        "pillow",
        "jsonschema",
        "networkx",
        "pytest",
        "pytest-asyncio",
        "pandas",
        "pyarrow",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "python_executable": os.path.basename(sys.executable),
        "platform": platform.platform(),
        "packages": packages,
    }


def build_manifest(
    spec: list[dict[str, Any]],
    roots: dict[str, Path],
    *,
    generated_at: str,
    generator: dict[str, Any],
) -> BaselineManifest:
    """Materialize digests for every entry in ``spec``.

    ``spec`` is the declarative entry list from ``baseline_spec.py``; this
    function is the side-effecting part that reads the filesystem.
    """
    entries: list[ManifestEntry] = []
    for item in spec:
        entry = ManifestEntry.from_dict(item)
        if entry.kind == "tool_env":
            entry.env = tool_env_record()
            entries.append(entry)
            continue

        target = resolve(entry.root, entry.relpath, roots)
        if not target.exists():
            raise FileNotFoundError(f"entry {entry.id!r}: {target} does not exist")

        if entry.kind == "git_repo":
            state = git_state(target)
            entry.commit = state["commit"]
            entry.branch = entry.branch or state["branch"]
            entry.remote = entry.remote or state["remote"]
            entry.worktree_dirty = state["worktree_dirty"]

        levels = item.get("digest_levels", [])
        if levels:
            entry.digests = _tree_digests(target, levels)
        pinned = item.get("pinned_files", [])
        if pinned:
            entry.files = _file_digests(target, pinned)
        entries.append(entry)

    return BaselineManifest(
        schema=SCHEMA_ID, generated_at=generated_at, generator=generator, entries=entries
    )


# ─────────────────────────────────────────────────────────────────────────────
# Verify
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Assertion:
    name: str
    expected: Any
    observed: Any
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected": self.expected,
            "observed": self.observed,
            "status": "pass" if self.passed else "fail",
        }


@dataclass
class VerificationResult:
    assertions: list[Assertion]
    errors: list[str]

    @property
    def passed(self) -> bool:
        return not self.errors and all(a.passed for a in self.assertions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.passed else "fail",
            "assertion_count": len(self.assertions),
            "failed_count": sum(1 for a in self.assertions if not a.passed),
            "errors": self.errors,
            "assertions": [a.to_dict() for a in self.assertions],
        }


def verify_manifest(
    manifest: BaselineManifest, roots: dict[str, Path], *, skip_content: bool = False
) -> VerificationResult:
    """Recompute every pinned digest and compare against the manifest.

    ``skip_content`` omits the expensive ``content``-level tree digests, for a
    fast structural smoke check. A milestone gate must run without it.
    """
    assertions: list[Assertion] = []
    errors: list[str] = []

    for entry in manifest.entries:
        if entry.kind == "tool_env":
            observed = tool_env_record()
            for key in ("python",):
                assertions.append(
                    Assertion(
                        f"{entry.id}.env.{key}", entry.env.get(key), observed.get(key),
                        entry.env.get(key) == observed.get(key),
                    )
                )
            continue

        try:
            target = resolve(entry.root, entry.relpath, roots)
        except KeyError as exc:
            errors.append(str(exc))
            continue

        exists = target.exists()
        assertions.append(Assertion(f"{entry.id}.exists", True, exists, exists))
        if not exists:
            continue

        if entry.kind == "git_repo":
            try:
                state = git_state(target)
            except RuntimeError as exc:
                errors.append(f"{entry.id}: {exc}")
                continue
            assertions.append(
                Assertion(
                    f"{entry.id}.commit", entry.commit, state["commit"],
                    entry.commit == state["commit"],
                )
            )

        for pinned in entry.digests:
            level = pinned["level"]
            if skip_content and level == "content":
                continue
            observed = digest_tree(target, level=level, exclude_dirs=DEFAULT_EXCLUDE_DIRS)
            assertions.append(
                Assertion(
                    f"{entry.id}.tree[{level}].digest", pinned["digest"], observed.digest,
                    pinned["digest"] == observed.digest,
                )
            )
            assertions.append(
                Assertion(
                    f"{entry.id}.tree[{level}].file_count",
                    pinned["file_count"], observed.file_count,
                    pinned["file_count"] == observed.file_count,
                )
            )

        for pinned_file in entry.files:
            full = target / pinned_file["relpath"]
            if not full.exists():
                assertions.append(
                    Assertion(f"{entry.id}.file[{pinned_file['relpath']}].exists", True, False, False)
                )
                continue
            observed_hash = sha256_file(full)
            assertions.append(
                Assertion(
                    f"{entry.id}.file[{pinned_file['relpath']}].sha256",
                    pinned_file["sha256"], observed_hash,
                    pinned_file["sha256"] == observed_hash,
                )
            )

    return VerificationResult(assertions=assertions, errors=errors)
