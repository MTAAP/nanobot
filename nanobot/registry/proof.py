"""Proof of work system for verifying agent task completion."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from loguru import logger


class ProofOfWork:
    """Structured evidence container for task completion verification.

    Collects typed evidence items that can be independently verified.
    """

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add_git_proof(
        self,
        branch: str,
        commit_hash: str,
        diff_summary: str = "",
    ) -> None:
        """Add git-based proof (branch + commit)."""
        self.items.append(
            {
                "type": "git",
                "branch": branch,
                "commit_hash": commit_hash,
                "diff_summary": diff_summary,
            }
        )

    def add_file_proof(
        self,
        path: str,
        sha256: str,
        size: int,
    ) -> None:
        """Add file-based proof (existence + hash)."""
        self.items.append(
            {
                "type": "file",
                "path": path,
                "sha256": sha256,
                "size": size,
            }
        )

    def add_command_proof(
        self,
        command: str,
        exit_code: int,
        output_snippet: str = "",
    ) -> None:
        """Add command execution proof."""
        self.items.append(
            {
                "type": "command",
                "command": command,
                "exit_code": exit_code,
                "output_snippet": output_snippet[:500],
            }
        )

    def add_test_proof(
        self,
        passed: int,
        failed: int,
        output: str = "",
    ) -> None:
        """Add test execution proof."""
        self.items.append(
            {
                "type": "test",
                "passed": passed,
                "failed": failed,
                "output": output[:1000],
            }
        )

    def add_pr_proof(
        self,
        pr_url: str,
        pr_number: int,
        branch: str,
        commit_hash: str,
        test_results: dict[str, Any] | None = None,
    ) -> None:
        """Add pull request proof for self-evolution tasks."""
        self.items.append(
            {
                "type": "pr",
                "pr_url": pr_url,
                "pr_number": pr_number,
                "branch": branch,
                "commit_hash": commit_hash,
                "test_results": test_results or {},
            }
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({"items": self.items})

    @classmethod
    def from_json(cls, data: str) -> "ProofOfWork":
        """Deserialize from JSON string."""
        parsed = json.loads(data)
        proof = cls()
        proof.items = parsed.get("items", [])
        return proof

    def is_empty(self) -> bool:
        """Check if any evidence has been collected."""
        return len(self.items) == 0


class ProofVerifier:
    """Independently verifies proof of work items.

    Runs verification checks against the actual workspace/system state
    to confirm that claimed work was actually performed.
    """

    def __init__(self, workspace: Path):
        self._workspace = workspace

    async def verify(self, proof: ProofOfWork) -> dict[str, Any]:
        """Verify all proof items.

        Returns:
            Dict with: valid (bool), verified_count, failed_items, details.
        """
        if proof.is_empty():
            return {
                "valid": False,
                "verified_count": 0,
                "failed_items": [],
                "details": "No proof items to verify",
            }

        results: list[dict[str, Any]] = []
        failed_items: list[dict[str, Any]] = []

        for item in proof.items:
            proof_type = item.get("type", "unknown")
            verifier = {
                "git": self._verify_git,
                "file": self._verify_file,
                "command": self._verify_command,
                "test": self._verify_tests,
                "pr": self._verify_pr,
            }.get(proof_type)

            if verifier is None:
                result = {"type": proof_type, "ok": False, "error": "unknown proof type"}
            else:
                result = await verifier(item)

            results.append(result)
            if not result.get("ok"):
                failed_items.append(result)

        verified = sum(1 for r in results if r.get("ok"))
        return {
            "valid": len(failed_items) == 0,
            "verified_count": verified,
            "failed_items": failed_items,
            "details": results,
        }

    async def _verify_git(self, item: dict[str, Any]) -> dict[str, Any]:
        """Verify git proof: check branch/commit exists."""
        branch = item.get("branch", "")
        commit_hash = item.get("commit_hash", "")

        try:
            # Check if branch exists (remote or local)
            proc = await asyncio.create_subprocess_exec(
                "git",
                "branch",
                "-a",
                cwd=str(self._workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            branches_output = stdout.decode("utf-8")

            if branch not in branches_output:
                return {
                    "type": "git",
                    "ok": False,
                    "error": f"branch '{branch}' not found",
                }

            # Verify commit exists if provided
            if commit_hash:
                proc = await asyncio.create_subprocess_exec(
                    "git",
                    "cat-file",
                    "-t",
                    commit_hash,
                    cwd=str(self._workspace),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                if proc.returncode != 0:
                    return {
                        "type": "git",
                        "ok": False,
                        "error": f"commit {commit_hash} not found",
                    }

            return {"type": "git", "ok": True}

        except Exception as e:
            return {"type": "git", "ok": False, "error": str(e)}

    async def _verify_file(self, item: dict[str, Any]) -> dict[str, Any]:
        """Verify file proof: check file exists and hash matches."""
        file_path = Path(item.get("path", ""))
        expected_sha = item.get("sha256", "")

        # Resolve relative paths against workspace
        if not file_path.is_absolute():
            file_path = self._workspace / file_path

        try:
            if not file_path.exists():
                return {
                    "type": "file",
                    "ok": False,
                    "error": f"file {file_path} not found",
                }

            if expected_sha:
                actual_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
                if actual_sha != expected_sha:
                    return {
                        "type": "file",
                        "ok": False,
                        "error": f"hash mismatch: expected {expected_sha[:12]}, "
                        f"got {actual_sha[:12]}",
                    }

            return {"type": "file", "ok": True}

        except Exception as e:
            return {"type": "file", "ok": False, "error": str(e)}

    async def _verify_command(self, item: dict[str, Any]) -> dict[str, Any]:
        """Verify command proof: re-run and check exit code."""
        command = item.get("command", "")
        expected_exit = item.get("exit_code", 0)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self._workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await asyncio.wait_for(proc.communicate(), timeout=60)

            if proc.returncode != expected_exit:
                return {
                    "type": "command",
                    "ok": False,
                    "error": f"exit code {proc.returncode} != expected {expected_exit}",
                }

            return {"type": "command", "ok": True}

        except asyncio.TimeoutError:
            return {"type": "command", "ok": False, "error": "command timed out"}
        except Exception as e:
            return {"type": "command", "ok": False, "error": str(e)}

    async def _verify_tests(self, item: dict[str, Any]) -> dict[str, Any]:
        """Verify test proof: check that tests actually pass."""
        passed = item.get("passed", 0)
        failed = item.get("failed", 0)

        # Basic sanity: claimed to have passed tests with no failures
        if failed > 0:
            return {
                "type": "test",
                "ok": False,
                "error": f"{failed} tests failed",
            }

        if passed == 0:
            return {
                "type": "test",
                "ok": False,
                "error": "no tests passed",
            }

        return {"type": "test", "ok": True}

    async def _verify_pr(self, item: dict[str, Any]) -> dict[str, Any]:
        """Verify PR proof: check PR exists via gh CLI."""
        pr_number = item.get("pr_number")
        branch = item.get("branch", "")

        if not pr_number:
            return {"type": "pr", "ok": False, "error": "no PR number provided"}

        try:
            # Check PR exists via gh CLI
            proc = await asyncio.create_subprocess_exec(
                "gh",
                "pr",
                "view",
                str(pr_number),
                "--repo",
                "MTAAP/nanobot",
                "--json",
                "number,state,headRefName",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                return {
                    "type": "pr",
                    "ok": False,
                    "error": f"PR #{pr_number} not found: {stderr.decode('utf-8')[:200]}",
                }

            pr_data = json.loads(stdout.decode("utf-8"))
            if branch and pr_data.get("headRefName") != branch:
                return {
                    "type": "pr",
                    "ok": False,
                    "error": f"PR branch mismatch: expected {branch}, "
                    f"got {pr_data.get('headRefName')}",
                }

            return {"type": "pr", "ok": True, "pr_state": pr_data.get("state")}

        except Exception as e:
            logger.warning(f"PR verification failed: {e}")
            return {"type": "pr", "ok": False, "error": str(e)}
