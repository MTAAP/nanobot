"""Self-evolution tool for nanobot code modification."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.registry.evolve import SelfEvolveManager


class SelfEvolveTool(Tool):
    """Tool for nanobot to modify its own codebase.

    Creates feature branches, runs tests, pushes changes, and creates
    pull requests. All changes are PR-gated -- never push to main.
    """

    def __init__(self, evolve_manager: "SelfEvolveManager"):
        self._manager = evolve_manager

    @property
    def name(self) -> str:
        return "self_evolve"

    @property
    def description(self) -> str:
        return (
            "Modify nanobot's own source code. Creates a feature branch, "
            "runs tests and linting, commits changes, and creates a PR. "
            "Use the actions in order: setup_repo -> create_branch -> "
            "(make changes with read_file/write_file) -> run_tests -> "
            "run_lint -> commit_push -> create_pr"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "setup_repo",
                        "create_branch",
                        "run_tests",
                        "run_lint",
                        "commit_push",
                        "create_pr",
                        "status",
                    ],
                    "description": "The self-evolution action to perform",
                },
                "branch_name": {
                    "type": "string",
                    "description": "Branch name (for create_branch)",
                },
                "commit_message": {
                    "type": "string",
                    "description": "Commit message (for commit_push)",
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to stage (for commit_push, optional)",
                },
                "pr_title": {
                    "type": "string",
                    "description": "PR title (for create_pr)",
                },
                "pr_body": {
                    "type": "string",
                    "description": "PR body/description (for create_pr)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        """Dispatch to the appropriate evolve manager method."""
        try:
            if action == "setup_repo":
                repo_path = await self._manager.ensure_repo()
                return f"Repository ready at {repo_path}"

            elif action == "create_branch":
                branch_name = kwargs.get("branch_name")
                if not branch_name:
                    return "Error: branch_name is required for create_branch"
                branch = await self._manager.create_branch(branch_name)
                return f"Created and checked out branch: {branch}"

            elif action == "run_tests":
                result = await self._manager.run_tests()
                status = "PASSED" if result["ok"] else "FAILED"
                return (
                    f"Tests {status}: {result['passed']} passed, "
                    f"{result['failed']} failed\n{result['output']}"
                )

            elif action == "run_lint":
                result = await self._manager.run_lint()
                status = "PASSED" if result["ok"] else "FAILED"
                return f"Lint {status}\n{result['output']}"

            elif action == "commit_push":
                message = kwargs.get("commit_message")
                if not message:
                    return "Error: commit_message is required for commit_push"
                files = kwargs.get("files")

                # Verify actual file changes exist before commit
                repo_path = self._manager._repo_path
                unstaged = await self._manager._run_git(["diff", "--stat"], cwd=repo_path)
                staged = await self._manager._run_git(["diff", "--cached", "--stat"], cwd=repo_path)
                untracked = await self._manager._run_git(
                    ["ls-files", "--others", "--exclude-standard"],
                    cwd=repo_path,
                )
                if not unstaged.strip() and not staged.strip() and not untracked.strip():
                    return (
                        "Error: no file changes detected. "
                        "Use write_file or edit_file to make "
                        "changes before committing."
                    )

                result = await self._manager.commit_and_push(message, files)
                if not result["ok"]:
                    return f"Error: {result.get('error', 'push failed')}"

                # Include diff stats in success response
                diff_stats = await self._manager._run_git(
                    ["diff", "--stat", "HEAD~1..HEAD"],
                    cwd=repo_path,
                )
                return (
                    f"Pushed commit {result['commit_sha'][:8]} "
                    f"to branch {result['branch']}\n"
                    f"Changes:\n{diff_stats.strip()}"
                )

            elif action == "create_pr":
                title = kwargs.get("pr_title")
                body = kwargs.get("pr_body", "")
                if not title:
                    return "Error: pr_title is required for create_pr"

                # Verify commits ahead of main before creating PR
                repo_path = self._manager._repo_path
                commits_ahead = await self._manager._run_git(
                    ["log", "main..HEAD", "--oneline"],
                    cwd=repo_path,
                )
                if not commits_ahead.strip():
                    return (
                        "Error: no commits ahead of main. "
                        "Use commit_push first to create commits "
                        "before opening a PR."
                    )

                result = await self._manager.create_pull_request(title, body)
                if not result["ok"]:
                    return f"Error creating PR: {result.get('error')}"
                return (
                    f"PR #{result['pr_number']} created: "
                    f"{result['pr_url']}\n"
                    f"Commits:\n{commits_ahead.strip()}"
                )

            elif action == "status":
                repo_path = await self._manager.get_repo_path()
                exists = repo_path.exists()
                return (
                    f"Repo path: {repo_path}\n"
                    f"Exists: {exists}\n"
                    f"Protected branches: {self._manager._protected_branches}"
                )

            else:
                return f"Error: unknown action '{action}'"

        except Exception as e:
            return f"Error in self_evolve({action}): {e}"
