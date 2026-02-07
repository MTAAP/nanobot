"""Tests for proof of work system."""

import asyncio
import hashlib
import json

import pytest

from nanobot.registry.proof import ProofOfWork, ProofVerifier
from nanobot.registry.store import AgentRegistry, TaskState


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ------------------------------------------------------------------
# ProofOfWork serialization
# ------------------------------------------------------------------


class TestProofOfWork:
    def test_empty_proof(self):
        proof = ProofOfWork()
        assert proof.is_empty()
        assert proof.items == []

    def test_add_git_proof(self):
        proof = ProofOfWork()
        proof.add_git_proof("feature/test", "abc123", "added tests")
        assert len(proof.items) == 1
        assert proof.items[0]["type"] == "git"
        assert proof.items[0]["branch"] == "feature/test"
        assert not proof.is_empty()

    def test_add_file_proof(self):
        proof = ProofOfWork()
        proof.add_file_proof("test.py", "sha256hash", 1234)
        assert proof.items[0]["type"] == "file"
        assert proof.items[0]["size"] == 1234

    def test_add_command_proof(self):
        proof = ProofOfWork()
        proof.add_command_proof("pytest", 0, "all passed")
        assert proof.items[0]["type"] == "command"
        assert proof.items[0]["exit_code"] == 0

    def test_add_test_proof(self):
        proof = ProofOfWork()
        proof.add_test_proof(passed=10, failed=0, output="10 passed")
        assert proof.items[0]["type"] == "test"
        assert proof.items[0]["passed"] == 10

    def test_add_pr_proof(self):
        proof = ProofOfWork()
        proof.add_pr_proof(
            pr_url="https://github.com/MTAAP/nanobot/pull/12",
            pr_number=12,
            branch="feature/test",
            commit_hash="abc123",
            test_results={"passed": 10, "failed": 0},
        )
        assert proof.items[0]["type"] == "pr"
        assert proof.items[0]["pr_number"] == 12

    def test_serialization_roundtrip(self):
        proof = ProofOfWork()
        proof.add_git_proof("feature/x", "sha1")
        proof.add_test_proof(5, 0)
        proof.add_pr_proof("url", 1, "branch", "hash")

        json_str = proof.to_json()
        restored = ProofOfWork.from_json(json_str)
        assert len(restored.items) == 3
        assert restored.items[0]["type"] == "git"
        assert restored.items[1]["type"] == "test"
        assert restored.items[2]["type"] == "pr"

    def test_output_truncation(self):
        proof = ProofOfWork()
        long_output = "x" * 2000
        proof.add_command_proof("cmd", 0, long_output)
        assert len(proof.items[0]["output_snippet"]) == 500

    def test_test_output_truncation(self):
        proof = ProofOfWork()
        long_output = "y" * 2000
        proof.add_test_proof(1, 0, long_output)
        assert len(proof.items[0]["output"]) == 1000


# ------------------------------------------------------------------
# ProofVerifier
# ------------------------------------------------------------------


class TestProofVerifier:
    def test_verify_empty_proof(self, tmp_path):
        verifier = ProofVerifier(tmp_path)
        proof = ProofOfWork()
        result = run(verifier.verify(proof))
        assert result["valid"] is False
        assert result["verified_count"] == 0

    def test_verify_file_proof_valid(self, tmp_path):
        # Create a real file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello", encoding="utf-8")
        expected_sha = hashlib.sha256(test_file.read_bytes()).hexdigest()

        proof = ProofOfWork()
        proof.add_file_proof(str(test_file), expected_sha, test_file.stat().st_size)

        verifier = ProofVerifier(tmp_path)
        result = run(verifier.verify(proof))
        assert result["valid"] is True
        assert result["verified_count"] == 1

    def test_verify_file_proof_wrong_hash(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello", encoding="utf-8")

        proof = ProofOfWork()
        proof.add_file_proof(str(test_file), "wrong_hash", 5)

        verifier = ProofVerifier(tmp_path)
        result = run(verifier.verify(proof))
        assert result["valid"] is False
        assert len(result["failed_items"]) == 1

    def test_verify_file_proof_missing_file(self, tmp_path):
        proof = ProofOfWork()
        proof.add_file_proof(str(tmp_path / "nonexistent.txt"), "hash", 0)

        verifier = ProofVerifier(tmp_path)
        result = run(verifier.verify(proof))
        assert result["valid"] is False

    def test_verify_file_relative_path(self, tmp_path):
        test_file = tmp_path / "relative.txt"
        test_file.write_text("data", encoding="utf-8")
        expected_sha = hashlib.sha256(test_file.read_bytes()).hexdigest()

        proof = ProofOfWork()
        proof.add_file_proof("relative.txt", expected_sha, 4)

        verifier = ProofVerifier(tmp_path)
        result = run(verifier.verify(proof))
        assert result["valid"] is True

    def test_verify_test_proof_passing(self, tmp_path):
        proof = ProofOfWork()
        proof.add_test_proof(passed=10, failed=0)

        verifier = ProofVerifier(tmp_path)
        result = run(verifier.verify(proof))
        assert result["valid"] is True

    def test_verify_test_proof_failures(self, tmp_path):
        proof = ProofOfWork()
        proof.add_test_proof(passed=8, failed=2)

        verifier = ProofVerifier(tmp_path)
        result = run(verifier.verify(proof))
        assert result["valid"] is False

    def test_verify_test_proof_no_tests(self, tmp_path):
        proof = ProofOfWork()
        proof.add_test_proof(passed=0, failed=0)

        verifier = ProofVerifier(tmp_path)
        result = run(verifier.verify(proof))
        assert result["valid"] is False

    def test_verify_command_proof(self, tmp_path):
        proof = ProofOfWork()
        proof.add_command_proof("echo hello", 0)

        verifier = ProofVerifier(tmp_path)
        result = run(verifier.verify(proof))
        assert result["valid"] is True

    def test_verify_command_proof_wrong_exit(self, tmp_path):
        proof = ProofOfWork()
        proof.add_command_proof("false", 0)  # `false` returns 1

        verifier = ProofVerifier(tmp_path)
        result = run(verifier.verify(proof))
        assert result["valid"] is False

    def test_verify_unknown_proof_type(self, tmp_path):
        proof = ProofOfWork()
        proof.items.append({"type": "unknown"})

        verifier = ProofVerifier(tmp_path)
        result = run(verifier.verify(proof))
        assert result["valid"] is False

    def test_mixed_proofs(self, tmp_path):
        test_file = tmp_path / "ok.txt"
        test_file.write_text("ok", encoding="utf-8")
        sha = hashlib.sha256(test_file.read_bytes()).hexdigest()

        proof = ProofOfWork()
        proof.add_file_proof(str(test_file), sha, 2)
        proof.add_test_proof(5, 0)
        proof.add_command_proof("echo ok", 0)

        verifier = ProofVerifier(tmp_path)
        result = run(verifier.verify(proof))
        assert result["valid"] is True
        assert result["verified_count"] == 3


# ------------------------------------------------------------------
# SubmitProofTool
# ------------------------------------------------------------------


class TestSubmitProofTool:
    def test_submit_stores_proof(self, tmp_path):
        from nanobot.agent.tools.proof import SubmitProofTool

        registry = AgentRegistry(tmp_path)
        run(registry.create_task("t1", "Work"))
        run(registry.register_agent("a1", "sub"))
        run(registry.assign_task("t1", "a1"))
        run(registry.update_task_state("t1", TaskState.IN_PROGRESS))

        tool = SubmitProofTool(registry=registry, task_id="t1")
        result = run(tool.execute(
            proof_type="test",
            data={"passed": 5, "failed": 0, "output": "all passed"},
        ))
        assert "submitted" in result.lower()

        task = run(registry.get_task("t1"))
        assert task["proof_of_work"] is not None
        assert task["state"] == TaskState.VERIFYING.value

    def test_submit_unknown_type(self, tmp_path):
        from nanobot.agent.tools.proof import SubmitProofTool

        registry = AgentRegistry(tmp_path)
        run(registry.create_task("t1", "Work"))

        tool = SubmitProofTool(registry=registry, task_id="t1")
        result = run(tool.execute(proof_type="magic", data={}))
        assert "unknown proof type" in result.lower()

    def test_submit_git_proof(self, tmp_path):
        from nanobot.agent.tools.proof import SubmitProofTool

        registry = AgentRegistry(tmp_path)
        run(registry.create_task("t1", "Work"))
        run(registry.register_agent("a1", "sub"))
        run(registry.assign_task("t1", "a1"))
        run(registry.update_task_state("t1", TaskState.IN_PROGRESS))

        tool = SubmitProofTool(registry=registry, task_id="t1")
        result = run(tool.execute(
            proof_type="git",
            data={"branch": "feature/x", "commit_hash": "abc123"},
        ))
        assert "submitted" in result.lower()

    def test_submit_pr_proof(self, tmp_path):
        from nanobot.agent.tools.proof import SubmitProofTool

        registry = AgentRegistry(tmp_path)
        run(registry.create_task("t1", "Work"))
        run(registry.register_agent("a1", "sub"))
        run(registry.assign_task("t1", "a1"))
        run(registry.update_task_state("t1", TaskState.IN_PROGRESS))

        tool = SubmitProofTool(registry=registry, task_id="t1")
        result = run(tool.execute(
            proof_type="pr",
            data={
                "pr_url": "https://github.com/MTAAP/nanobot/pull/1",
                "pr_number": 1,
                "branch": "feature/x",
                "commit_hash": "abc",
            },
        ))
        assert "submitted" in result.lower()
