"""Tests for self-evolution system."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nanobot.registry.evolve import SelfEvolveManager


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def manager(workspace):
    """Create a SelfEvolveManager with test settings."""
    return SelfEvolveManager(
        workspace=workspace,
        repo_url="https://github.com/MTAAP/nanobot.git",
        github_token="test-token-123",
        protected_branches=["main", "master"],
        test_command="echo 'tests passed'",
        lint_command="echo 'lint passed'",
    )


# ------------------------------------------------------------------
# SelfEvolveManager
# ------------------------------------------------------------------


class TestSelfEvolveManager:
    def test_repo_path(self, manager, workspace):
        assert manager._repo_path == workspace / "nanobot"

    def test_auth_url(self, manager):
        url = manager._make_auth_url("https://github.com/MTAAP/nanobot.git")
        assert "x-access-token:test-token-123@github.com" in url

    def test_auth_url_ssh(self, manager):
        url = manager._make_auth_url("git@github.com:MTAAP/nanobot.git")
        assert "x-access-token:test-token-123@github.com" in url
        assert url.startswith("https://")

    def test_protected_branch_check(self, manager):
        with pytest.raises(ValueError, match="protected"):
            run(manager.create_branch("main"))

    def test_protected_branch_master(self, manager):
        with pytest.raises(ValueError, match="protected"):
            run(manager.create_branch("master"))

    def test_get_repo_path(self, manager, workspace):
        path = run(manager.get_repo_path())
        assert path == workspace / "nanobot"


class TestEnsureRepo:
    @patch("nanobot.registry.evolve.SelfEvolveManager._run_git", new_callable=AsyncMock)
    @patch("nanobot.registry.evolve.SelfEvolveManager._run_cmd", new_callable=AsyncMock)
    def test_clone_fresh(self, mock_cmd, mock_git, manager, workspace):
        """Test cloning when repo doesn't exist."""
        mock_git.return_value = ""
        mock_cmd.return_value = ""

        path = run(manager.ensure_repo())
        assert path == workspace / "nanobot"

        # Should have called clone
        calls = mock_git.call_args_list
        assert any("clone" in str(c) for c in calls)

    @patch("nanobot.registry.evolve.SelfEvolveManager._run_git", new_callable=AsyncMock)
    @patch("nanobot.registry.evolve.SelfEvolveManager._run_cmd", new_callable=AsyncMock)
    def test_pull_existing(self, mock_cmd, mock_git, manager, workspace):
        """Test pulling when repo already exists."""
        repo_path = workspace / "nanobot"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        mock_git.return_value = ""
        mock_cmd.return_value = ""

        run(manager.ensure_repo())

        # Should have called fetch + pull, not clone
        calls = [str(c) for c in mock_git.call_args_list]
        assert any("fetch" in c for c in calls)
        assert not any("clone" in c for c in calls)


class TestRunTests:
    def test_run_tests_success(self, manager, workspace):
        """Test running tests that succeed."""
        # Repo path must exist for subprocess cwd
        manager._repo_path.mkdir(parents=True, exist_ok=True)
        result = run(manager.run_tests())
        assert result["ok"] is True

    def test_run_tests_failure(self, workspace):
        """Test running tests that fail."""
        mgr = SelfEvolveManager(
            workspace=workspace,
            repo_url="https://github.com/MTAAP/nanobot.git",
            github_token="test",
            test_command="false",
        )
        mgr._repo_path.mkdir(parents=True, exist_ok=True)
        result = run(mgr.run_tests())
        assert result["ok"] is False


class TestRunLint:
    def test_run_lint_success(self, manager, workspace):
        """Test running lint that succeeds."""
        # Repo path must exist for subprocess cwd
        manager._repo_path.mkdir(parents=True, exist_ok=True)
        result = run(manager.run_lint())
        assert result["ok"] is True

    def test_run_lint_failure(self, workspace):
        """Test running lint that fails."""
        mgr = SelfEvolveManager(
            workspace=workspace,
            repo_url="https://github.com/MTAAP/nanobot.git",
            github_token="test",
            lint_command="false",
        )
        mgr._repo_path.mkdir(parents=True, exist_ok=True)
        result = run(mgr.run_lint())
        assert result["ok"] is False


class TestCommitAndPush:
    @patch("nanobot.registry.evolve.SelfEvolveManager._run_git", new_callable=AsyncMock)
    def test_refuse_push_to_protected(self, mock_git, manager):
        """Test that push to protected branch is refused."""
        mock_git.return_value = "main"  # current branch

        result = run(manager.commit_and_push("test message"))
        assert result["ok"] is False
        assert "protected" in result["error"].lower()

    @patch("nanobot.registry.evolve.SelfEvolveManager._run_git", new_callable=AsyncMock)
    def test_commit_and_push_success(self, mock_git, manager):
        """Test successful commit and push."""
        # First call returns branch name, subsequent calls succeed
        mock_git.side_effect = [
            "feature/test\n",  # rev-parse HEAD (branch name)
            "",  # git add
            "",  # git commit
            "abc123def456\n",  # git rev-parse HEAD (sha)
            "",  # git push
        ]

        result = run(manager.commit_and_push("add feature", files=["test.py"]))
        assert result["ok"] is True
        assert result["commit_sha"] == "abc123def456"
        assert result["branch"] == "feature/test"


class TestCreatePullRequest:
    @patch("nanobot.registry.evolve.shutil.which", return_value=None)
    def test_no_gh_cli(self, mock_which, manager):
        """Test failure when gh CLI is not installed."""
        result = run(manager.create_pull_request("Title", "Body"))
        assert result["ok"] is False
        assert "gh CLI not found" in result["error"]


# ------------------------------------------------------------------
# SelfEvolveTool
# ------------------------------------------------------------------


class TestSelfEvolveTool:
    def test_tool_properties(self, manager):
        from nanobot.agent.tools.evolve import SelfEvolveTool

        tool = SelfEvolveTool(manager)
        assert tool.name == "self_evolve"
        assert "action" in tool.parameters["properties"]

    def test_status_action(self, manager, workspace):
        from nanobot.agent.tools.evolve import SelfEvolveTool

        tool = SelfEvolveTool(manager)
        result = run(tool.execute(action="status"))
        assert "nanobot" in result
        assert "Protected branches" in result

    @patch("nanobot.registry.evolve.SelfEvolveManager.ensure_repo", new_callable=AsyncMock)
    def test_setup_repo_action(self, mock_ensure, manager, workspace):
        from nanobot.agent.tools.evolve import SelfEvolveTool

        mock_ensure.return_value = workspace / "nanobot"
        tool = SelfEvolveTool(manager)
        result = run(tool.execute(action="setup_repo"))
        assert "ready" in result.lower()

    def test_create_branch_missing_name(self, manager):
        from nanobot.agent.tools.evolve import SelfEvolveTool

        tool = SelfEvolveTool(manager)
        result = run(tool.execute(action="create_branch"))
        assert "required" in result.lower()

    def test_commit_push_missing_message(self, manager):
        from nanobot.agent.tools.evolve import SelfEvolveTool

        tool = SelfEvolveTool(manager)
        result = run(tool.execute(action="commit_push"))
        assert "required" in result.lower()

    def test_create_pr_missing_title(self, manager):
        from nanobot.agent.tools.evolve import SelfEvolveTool

        tool = SelfEvolveTool(manager)
        result = run(tool.execute(action="create_pr"))
        assert "required" in result.lower()

    def test_unknown_action(self, manager):
        from nanobot.agent.tools.evolve import SelfEvolveTool

        tool = SelfEvolveTool(manager)
        result = run(tool.execute(action="magic"))
        assert "unknown action" in result.lower()

    @patch("nanobot.registry.evolve.SelfEvolveManager.run_tests", new_callable=AsyncMock)
    def test_run_tests_action(self, mock_tests, manager):
        from nanobot.agent.tools.evolve import SelfEvolveTool

        mock_tests.return_value = {"ok": True, "passed": 10, "failed": 0, "output": "all good"}
        tool = SelfEvolveTool(manager)
        result = run(tool.execute(action="run_tests"))
        assert "PASSED" in result
        assert "10 passed" in result

    @patch("nanobot.registry.evolve.SelfEvolveManager.run_lint", new_callable=AsyncMock)
    def test_run_lint_action(self, mock_lint, manager):
        from nanobot.agent.tools.evolve import SelfEvolveTool

        mock_lint.return_value = {"ok": True, "output": "clean"}
        tool = SelfEvolveTool(manager)
        result = run(tool.execute(action="run_lint"))
        assert "PASSED" in result


# ------------------------------------------------------------------
# Integration with registry PoW
# ------------------------------------------------------------------


class TestEvolveWithRegistry:
    def test_evolve_manager_protected_branches(self, manager):
        """Safety: refuse to push to any protected branch."""
        for branch in ["main", "master"]:
            with pytest.raises(ValueError, match="protected"):
                run(manager.create_branch(branch))

    def test_evolve_custom_protected_branches(self, workspace):
        """Custom protected branches are respected."""
        mgr = SelfEvolveManager(
            workspace=workspace,
            repo_url="https://github.com/MTAAP/nanobot.git",
            github_token="test",
            protected_branches=["main", "staging", "production"],
        )
        for branch in ["main", "staging", "production"]:
            with pytest.raises(ValueError, match="protected"):
                run(mgr.create_branch(branch))
