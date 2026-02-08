"""Tests for post-loop action verification (hallucinated action detection)."""

import pytest

from nanobot.agent.loop import AgentLoop


class TestContainsUnverifiedActions:
    """Test the _contains_unverified_actions regex detection."""

    def _check(self, text: str) -> bool:
        """Call the class method directly via the compiled pattern."""
        return bool(AgentLoop._ACTION_CLAIM_PATTERN.search(text))

    # -- Positive cases: should detect action claims --

    def test_ive_updated(self) -> None:
        assert self._check("I've updated HEARTBEAT.md with the new tasks.")

    def test_i_have_written(self) -> None:
        assert self._check("I have written your preferences to USER.md.")

    def test_ive_created(self) -> None:
        assert self._check("I've created the cron job for daily audits.")

    def test_i_have_configured(self) -> None:
        assert self._check("I have configured the webhook listener.")

    def test_ive_set_up(self) -> None:
        assert self._check("I've set up the scheduled tasks as requested.")

    def test_ive_initialized(self) -> None:
        assert self._check("I've initialized the memory system.")

    def test_ive_populated(self) -> None:
        assert self._check("I've populated HEARTBEAT.md with the roadmap items.")

    def test_ive_added(self) -> None:
        assert self._check("I've added the team information to USER.md.")

    def test_ive_installed(self) -> None:
        assert self._check("I've installed the MCP server.")

    def test_ive_scheduled(self) -> None:
        assert self._check("I've scheduled the daily audit for 9 AM.")

    def test_i_updated(self) -> None:
        assert self._check("I updated the configuration file.")

    def test_i_wrote(self) -> None:
        assert self._check("I wrote the new content to the file.")

    def test_i_created(self) -> None:
        assert self._check("I created a new cron job.")

    def test_changes_have_been_made(self) -> None:
        assert self._check("Changes have been made to the config.")

    def test_file_has_been_updated(self) -> None:
        assert self._check("File has been updated successfully.")

    def test_ive_re_initialized(self) -> None:
        assert self._check("I've re-initialized the heartbeat tasks.")

    def test_ive_activated(self) -> None:
        assert self._check("I've activated the cron jobs.")

    def test_ive_saved(self) -> None:
        assert self._check("I've saved the changes to disk.")

    # -- Negative cases: should NOT trigger false positives --

    def test_greeting(self) -> None:
        assert not self._check("Hello! How can I help you today?")

    def test_question(self) -> None:
        assert not self._check("What would you like me to update?")

    def test_future_intent(self) -> None:
        assert not self._check("I will update the file for you now.")

    def test_explanation(self) -> None:
        assert not self._check("The file contains the current configuration.")

    def test_clarification(self) -> None:
        assert not self._check("Which file would you like me to write to?")

    def test_empty(self) -> None:
        assert not self._check("")

    def test_tool_result_description(self) -> None:
        assert not self._check("The command returned exit code 0.")
