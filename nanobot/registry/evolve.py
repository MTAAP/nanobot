"""Self-evolution manager for nanobot code modification via PRs."""

import asyncio
import shutil
from pathlib import Path
from typing import Any

from loguru import logger


class SelfEvolveManager:
    """Manages the git workflow for nanobot self-modification.

    Handles cloning the repo, creating branches, running tests/lint,
    committing, pushing, and creating pull requests. All changes go
    through PRs -- never push to protected branches.
    """

    def __init__(
        self,
        workspace: Path,
        repo_url: str,
        github_token: str,
        protected_branches: list[str] | None = None,
        test_command: str = "pytest tests/ -v",
        lint_command: str = "ruff check nanobot/ && ruff format --check nanobot/",
        auto_merge: bool = False,
    ):
        self._workspace = workspace
        self._repo_url = repo_url
        self._github_token = github_token
        self._protected_branches = protected_branches or ["main", "master"]
        self._test_command = test_command
        self._lint_command = lint_command
        self._auto_merge = auto_merge
        self._repo_path = workspace / "nanobot"

    async def ensure_repo(self) -> Path:
        """Clone or pull the repo in workspace. Returns repo path."""
        # Build authenticated URL
        auth_url = self._make_auth_url(self._repo_url)

        if self._repo_path.exists() and (self._repo_path / ".git").exists():
            # Pull latest
            logger.debug("Pulling latest changes in self-evolve repo")
            await self._run_git(["fetch", "--all"], cwd=self._repo_path)
            await self._run_git(["checkout", "main"], cwd=self._repo_path)
            await self._run_git(["pull", "origin", "main"], cwd=self._repo_path)
        else:
            # Clone fresh
            logger.debug(f"Cloning repo to {self._repo_path}")
            self._repo_path.parent.mkdir(parents=True, exist_ok=True)
            await self._run_git(
                ["clone", auth_url, str(self._repo_path)],
                cwd=self._workspace,
            )

        # Set gh repo default
        if shutil.which("gh"):
            try:
                await self._run_cmd(
                    ["gh", "repo", "set-default", "MTAAP/nanobot"],
                    cwd=self._repo_path,
                )
            except Exception:
                pass  # Non-critical

        return self._repo_path

    async def create_branch(self, branch_name: str) -> str:
        """Create and checkout a feature branch from latest main.

        Args:
            branch_name: Name for the new branch.

        Returns:
            The branch name.

        Raises:
            ValueError: If branch_name is a protected branch.
        """
        if branch_name in self._protected_branches:
            raise ValueError(f"Cannot create branch '{branch_name}': it is protected")

        await self._run_git(["checkout", "main"], cwd=self._repo_path)
        await self._run_git(["pull", "origin", "main"], cwd=self._repo_path)
        await self._run_git(["checkout", "-b", branch_name], cwd=self._repo_path)

        logger.debug(f"Created branch: {branch_name}")
        return branch_name

    async def run_tests(self) -> dict[str, Any]:
        """Run test suite. Returns {passed, failed, output, ok}."""
        try:
            proc = await asyncio.create_subprocess_shell(
                self._test_command,
                cwd=str(self._repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=self._get_env(),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
            output = stdout.decode("utf-8")

            # Parse pytest output for pass/fail counts
            passed = 0
            failed = 0
            for line in output.split("\n"):
                if "passed" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            try:
                                passed = int(parts[i - 1])
                            except ValueError:
                                pass
                        if part == "failed" and i > 0:
                            try:
                                failed = int(parts[i - 1])
                            except ValueError:
                                pass

            return {
                "ok": proc.returncode == 0,
                "passed": passed,
                "failed": failed,
                "output": output[-2000:],  # Last 2000 chars
            }
        except asyncio.TimeoutError:
            return {"ok": False, "passed": 0, "failed": 0, "output": "Test timeout"}
        except Exception as e:
            return {"ok": False, "passed": 0, "failed": 0, "output": str(e)}

    async def run_lint(self) -> dict[str, Any]:
        """Run linting. Returns {ok, output}."""
        try:
            proc = await asyncio.create_subprocess_shell(
                self._lint_command,
                cwd=str(self._repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=self._get_env(),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = stdout.decode("utf-8")

            return {
                "ok": proc.returncode == 0,
                "output": output[-2000:],
            }
        except asyncio.TimeoutError:
            return {"ok": False, "output": "Lint timeout"}
        except Exception as e:
            return {"ok": False, "output": str(e)}

    async def commit_and_push(
        self,
        message: str,
        files: list[str] | None = None,
    ) -> dict[str, Any]:
        """Stage, commit, and push. Returns {commit_sha, branch, ok}.

        Refuses to push to protected branches.
        """
        # Get current branch
        result = await self._run_git(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=self._repo_path,
        )
        branch = result.strip()

        if branch in self._protected_branches:
            return {
                "ok": False,
                "error": f"Refusing to push to protected branch '{branch}'",
                "commit_sha": "",
                "branch": branch,
            }

        # Stage files
        if files:
            for f in files:
                await self._run_git(["add", f], cwd=self._repo_path)
        else:
            await self._run_git(["add", "-A"], cwd=self._repo_path)

        # Commit
        await self._run_git(
            ["commit", "-m", message],
            cwd=self._repo_path,
        )

        # Get commit SHA
        sha = (
            await self._run_git(
                ["rev-parse", "HEAD"],
                cwd=self._repo_path,
            )
        ).strip()

        # Push with auth URL
        auth_url = self._make_auth_url(self._repo_url)
        await self._run_git(
            ["push", "-u", auth_url, branch],
            cwd=self._repo_path,
        )

        logger.debug(f"Pushed {sha[:8]} to {branch}")
        return {"ok": True, "commit_sha": sha, "branch": branch}

    async def create_pull_request(
        self,
        title: str,
        body: str,
        base: str = "main",
    ) -> dict[str, Any]:
        """Create PR via gh CLI. Returns {pr_url, pr_number, ok}."""
        if not shutil.which("gh"):
            return {"ok": False, "error": "gh CLI not found", "pr_url": "", "pr_number": 0}

        try:
            proc = await asyncio.create_subprocess_exec(
                "gh",
                "pr",
                "create",
                "--repo",
                "MTAAP/nanobot",
                "--title",
                title,
                "--body",
                body,
                "--base",
                base,
                cwd=str(self._repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_env(),
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error = stderr.decode("utf-8").strip()
                return {"ok": False, "error": error, "pr_url": "", "pr_number": 0}

            pr_url = stdout.decode("utf-8").strip()
            # Extract PR number from URL
            pr_number = 0
            if "/pull/" in pr_url:
                try:
                    pr_number = int(pr_url.split("/pull/")[-1].strip("/"))
                except ValueError:
                    pass

            logger.debug(f"Created PR #{pr_number}: {pr_url}")
            return {"ok": True, "pr_url": pr_url, "pr_number": pr_number}

        except Exception as e:
            return {"ok": False, "error": str(e), "pr_url": "", "pr_number": 0}

    async def get_repo_path(self) -> Path:
        """Get path to the repo clone in workspace."""
        return self._repo_path

    def _make_auth_url(self, url: str) -> str:
        """Create an authenticated HTTPS git URL."""
        # Convert SSH or plain HTTPS to token-authenticated HTTPS
        if url.startswith("git@github.com:"):
            url = url.replace("git@github.com:", "https://github.com/")
        if url.startswith("https://github.com/"):
            return url.replace(
                "https://github.com/",
                f"https://x-access-token:{self._github_token}@github.com/",
            )
        return url

    def _get_env(self) -> dict[str, str]:
        """Get environment with GITHUB_TOKEN set."""
        import os

        env = {**os.environ}
        env["GITHUB_TOKEN"] = self._github_token
        env["GH_TOKEN"] = self._github_token
        return env

    async def _run_git(self, args: list[str], cwd: Path) -> str:
        """Run a git command and return stdout."""
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error = stderr.decode("utf-8").strip()
            raise RuntimeError(f"git {' '.join(args)} failed: {error}")

        return stdout.decode("utf-8")

    async def _run_cmd(self, cmd: list[str], cwd: Path) -> str:
        """Run a shell command and return stdout."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error = stderr.decode("utf-8").strip()
            raise RuntimeError(f"{' '.join(cmd)} failed: {error}")

        return stdout.decode("utf-8")
