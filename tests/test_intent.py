"""Tests for IntentClassifier."""

from unittest.mock import AsyncMock

import pytest

from nanobot.agent.intent import IntentClassifier, QueryIntent
from nanobot.providers.base import LLMResponse

# ---------------------------------------------------------------------------
# Tier-1: FACTUAL patterns
# ---------------------------------------------------------------------------


class TestFactualIntent:
    @pytest.mark.asyncio
    async def test_what_time_is_it(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("What time is it?") == QueryIntent.FACTUAL

    @pytest.mark.asyncio
    async def test_current_date(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("What's the date today?") == QueryIntent.FACTUAL

    @pytest.mark.asyncio
    async def test_what_day_is_it(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("What day is it today?") == QueryIntent.FACTUAL

    @pytest.mark.asyncio
    async def test_weather(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("How's the weather?") == QueryIntent.FACTUAL

    @pytest.mark.asyncio
    async def test_calculate(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Calculate 5 * 3") == QueryIntent.FACTUAL

    @pytest.mark.asyncio
    async def test_disk_space(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("How much disk space is left?") == QueryIntent.FACTUAL

    @pytest.mark.asyncio
    async def test_temperature(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("What's the temperature outside?") == QueryIntent.FACTUAL


# ---------------------------------------------------------------------------
# Tier-1: VERIFY_STATE patterns
# ---------------------------------------------------------------------------


class TestVerifyStateIntent:
    @pytest.mark.asyncio
    async def test_beer_reminder(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("What about the beer reminder?") == QueryIntent.VERIFY_STATE

    @pytest.mark.asyncio
    async def test_any_reminders(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Do I have any reminders set?") == QueryIntent.VERIFY_STATE

    @pytest.mark.asyncio
    async def test_cron_job_running(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Is the cron job still running?") == QueryIntent.VERIFY_STATE

    @pytest.mark.asyncio
    async def test_scheduled_tomorrow(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("What's scheduled for tomorrow?") == QueryIntent.VERIFY_STATE

    @pytest.mark.asyncio
    async def test_timer_active(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Is my timer still active?") == QueryIntent.VERIFY_STATE


# ---------------------------------------------------------------------------
# Tier-1: CONVERSATIONAL patterns (no regex match -> default)
# ---------------------------------------------------------------------------


class TestConversationalIntent:
    @pytest.mark.asyncio
    async def test_greeting(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Hello!") == QueryIntent.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_thanks(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Thanks for your help") == QueryIntent.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_how_are_you(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("How are you?") == QueryIntent.CONVERSATIONAL


# ---------------------------------------------------------------------------
# ACTION patterns (tier-1 regex for imperative verbs, LLM fallback for others)
# ---------------------------------------------------------------------------


class TestActionIntent:
    @pytest.mark.asyncio
    async def test_set_reminder_matches_verify_state(self) -> None:
        """'Set a reminder' contains 'reminder', which the regex matches as VERIFY_STATE."""
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Set a reminder for 5pm") == QueryIntent.VERIFY_STATE

    @pytest.mark.asyncio
    async def test_run_script_via_regex(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Run the deployment script") == QueryIntent.ACTION

    @pytest.mark.asyncio
    async def test_create_file_via_regex(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Create a file called test.txt") == QueryIntent.ACTION

    @pytest.mark.asyncio
    async def test_write_file_via_regex(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Write my preferences to USER.md") == QueryIntent.ACTION

    @pytest.mark.asyncio
    async def test_update_file_via_regex(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Update the HEARTBEAT.md with new tasks") == QueryIntent.ACTION

    @pytest.mark.asyncio
    async def test_install_via_regex(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Install the new MCP server") == QueryIntent.ACTION

    @pytest.mark.asyncio
    async def test_please_add_via_regex(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Please add the team info to USER.md") == QueryIntent.ACTION

    @pytest.mark.asyncio
    async def test_configure_via_regex(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Configure the webhook listener") == QueryIntent.ACTION

    @pytest.mark.asyncio
    async def test_schedule_matches_verify_state(self) -> None:
        """'Schedule' matches VERIFY_STATE (0.90) over ACTION (0.85)."""
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Schedule a daily backup at 9am") == QueryIntent.VERIFY_STATE

    @pytest.mark.asyncio
    async def test_delete_via_regex(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Delete the old config files") == QueryIntent.ACTION

    @pytest.mark.asyncio
    async def test_deploy_via_regex(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("Deploy the new version") == QueryIntent.ACTION

    @pytest.mark.asyncio
    async def test_send_message_via_llm(self) -> None:
        """'Send' is not in the tier-1 regex, so it falls back to LLM."""
        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=LLMResponse(content='{"intent": "action"}'))
        c = IntentClassifier(llm_fallback=True, provider=provider)
        assert await c.classify("Send a message to John") == QueryIntent.ACTION


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_string(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("") == QueryIntent.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_whitespace_only(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        assert await c.classify("   ") == QueryIntent.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_very_long_message(self) -> None:
        c = IntentClassifier(llm_fallback=False)
        long_msg = "a " * 10_000
        result = await c.classify(long_msg)
        assert isinstance(result, QueryIntent)

    @pytest.mark.asyncio
    async def test_mixed_signals(self) -> None:
        """Messages with mixed signals should not crash and return a valid intent."""
        c = IntentClassifier(llm_fallback=False)
        result = await c.classify("Remember to check what time the meeting is")
        assert isinstance(result, QueryIntent)


# ---------------------------------------------------------------------------
# Disabled classifier
# ---------------------------------------------------------------------------


class TestDisabledClassifier:
    @pytest.mark.asyncio
    async def test_disabled_returns_conversational(self) -> None:
        c = IntentClassifier(enabled=False)
        assert await c.classify("What time is it?") == QueryIntent.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_disabled_ignores_factual(self) -> None:
        c = IntentClassifier(enabled=False)
        assert await c.classify("Calculate 5 * 3") == QueryIntent.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_disabled_ignores_verify_state(self) -> None:
        c = IntentClassifier(enabled=False)
        assert await c.classify("Do I have any reminders set?") == QueryIntent.CONVERSATIONAL


# ---------------------------------------------------------------------------
# LLM fallback (mocked)
# ---------------------------------------------------------------------------


class TestLLMFallback:
    @pytest.mark.asyncio
    async def test_llm_called_when_no_regex_match(self) -> None:
        """When regex has no confident match and LLM fallback enabled, call provider."""
        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=LLMResponse(content='{"intent": "memory"}'))
        c = IntentClassifier(llm_fallback=True, provider=provider)
        result = await c.classify("What did we talk about yesterday?")
        assert result == QueryIntent.MEMORY
        provider.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_not_called_when_regex_matches(self) -> None:
        """When regex matches with high confidence, LLM should not be called."""
        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=LLMResponse(content='{"intent": "conversational"}'))
        c = IntentClassifier(llm_fallback=True, provider=provider)
        result = await c.classify("What time is it?")
        assert result == QueryIntent.FACTUAL
        provider.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_not_called_when_fallback_disabled(self) -> None:
        """When llm_fallback=False, provider should never be called."""
        provider = AsyncMock()
        c = IntentClassifier(llm_fallback=False, provider=provider)
        await c.classify("What did we talk about yesterday?")
        provider.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_not_called_when_no_provider(self) -> None:
        """When no provider is set, should fall back to CONVERSATIONAL."""
        c = IntentClassifier(llm_fallback=True, provider=None)
        result = await c.classify("What did we talk about yesterday?")
        assert result == QueryIntent.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_llm_returns_empty_content(self) -> None:
        """When LLM returns empty content, fall back to CONVERSATIONAL."""
        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=LLMResponse(content=None))
        c = IntentClassifier(llm_fallback=True, provider=provider)
        result = await c.classify("What did we talk about yesterday?")
        assert result == QueryIntent.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_llm_returns_invalid_json(self) -> None:
        """When LLM returns invalid JSON, fall back to CONVERSATIONAL."""
        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=LLMResponse(content="not valid json"))
        c = IntentClassifier(llm_fallback=True, provider=provider)
        result = await c.classify("What did we talk about yesterday?")
        assert result == QueryIntent.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_llm_returns_unknown_intent(self) -> None:
        """When LLM returns an unknown intent value, fall back to CONVERSATIONAL."""
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=LLMResponse(content='{"intent": "unknown_category"}')
        )
        c = IntentClassifier(llm_fallback=True, provider=provider)
        result = await c.classify("What did we talk about yesterday?")
        assert result == QueryIntent.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back(self) -> None:
        """When LLM raises an exception, fall back to CONVERSATIONAL."""
        provider = AsyncMock()
        provider.chat = AsyncMock(side_effect=RuntimeError("API error"))
        c = IntentClassifier(llm_fallback=True, provider=provider)
        result = await c.classify("What did we talk about yesterday?")
        assert result == QueryIntent.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_llm_handles_markdown_fenced_json(self) -> None:
        """LLM response wrapped in markdown code fences should be parsed."""
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=LLMResponse(content='```json\n{"intent": "action"}\n```')
        )
        c = IntentClassifier(llm_fallback=True, provider=provider)
        result = await c.classify("Send an email to my boss")
        assert result == QueryIntent.ACTION

    @pytest.mark.asyncio
    async def test_llm_passes_model_parameter(self) -> None:
        """The model parameter should be forwarded to the provider."""
        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=LLMResponse(content='{"intent": "action"}'))
        c = IntentClassifier(llm_fallback=True, provider=provider, model="gpt-4o-mini")
        await c.classify("Do something for me")
        call_kwargs = provider.chat.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["max_tokens"] == 64
        assert call_kwargs["temperature"] == 0.0
