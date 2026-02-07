"""Submit proof of work tool for subagents."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.registry.store import AgentRegistry


class SubmitProofTool(Tool):
    """Tool for subagents to submit proof of completed work.

    Builds a ProofOfWork object, stores it in the registry,
    and transitions the task to VERIFYING state.
    """

    def __init__(self, registry: "AgentRegistry", task_id: str):
        self._registry = registry
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "submit_proof"

    @property
    def description(self) -> str:
        return (
            "Submit proof that you completed a task. "
            "Provide evidence of your work (git commits, file hashes, "
            "test results, PR URLs) so the daemon can verify it."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "proof_type": {
                    "type": "string",
                    "enum": ["git", "file", "command", "test", "pr"],
                    "description": "Type of proof to submit",
                },
                "data": {
                    "type": "object",
                    "description": (
                        "Proof data. For git: {branch, commit_hash, diff_summary}. "
                        "For file: {path, sha256, size}. "
                        "For command: {command, exit_code, output_snippet}. "
                        "For test: {passed, failed, output}. "
                        "For pr: {pr_url, pr_number, branch, commit_hash, test_results}."
                    ),
                },
            },
            "required": ["proof_type", "data"],
        }

    async def execute(
        self,
        proof_type: str,
        data: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Build proof, store in registry, transition task to VERIFYING."""
        try:
            from nanobot.registry.proof import ProofOfWork
            from nanobot.registry.store import TaskState

            proof = ProofOfWork()

            if proof_type == "git":
                proof.add_git_proof(
                    branch=data.get("branch", ""),
                    commit_hash=data.get("commit_hash", ""),
                    diff_summary=data.get("diff_summary", ""),
                )
            elif proof_type == "file":
                proof.add_file_proof(
                    path=data.get("path", ""),
                    sha256=data.get("sha256", ""),
                    size=data.get("size", 0),
                )
            elif proof_type == "command":
                proof.add_command_proof(
                    command=data.get("command", ""),
                    exit_code=data.get("exit_code", 0),
                    output_snippet=data.get("output_snippet", ""),
                )
            elif proof_type == "test":
                proof.add_test_proof(
                    passed=data.get("passed", 0),
                    failed=data.get("failed", 0),
                    output=data.get("output", ""),
                )
            elif proof_type == "pr":
                proof.add_pr_proof(
                    pr_url=data.get("pr_url", ""),
                    pr_number=data.get("pr_number", 0),
                    branch=data.get("branch", ""),
                    commit_hash=data.get("commit_hash", ""),
                    test_results=data.get("test_results"),
                )
            else:
                return f"Error: unknown proof type '{proof_type}'"

            # Store proof in registry
            await self._registry.submit_proof(self._task_id, proof.to_json())

            # Transition task to VERIFYING
            task = await self._registry.get_task(self._task_id)
            if task and task["state"] == TaskState.IN_PROGRESS.value:
                await self._registry.update_task_state(
                    self._task_id, TaskState.VERIFYING, reason="proof submitted"
                )

            return f"Proof submitted ({proof_type}). Task moved to verification."

        except Exception as e:
            return f"Error submitting proof: {e}"
