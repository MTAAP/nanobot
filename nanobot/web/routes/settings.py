"""Settings route for full subsystem configuration and runtime hot-swap."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from loguru import logger

from nanobot.config.loader import convert_to_camel, load_config, save_config

router = APIRouter(prefix="/settings", tags=["settings"])

PROVIDER_NAMES = [
    "anthropic",
    "openai",
    "openrouter",
    "groq",
    "gemini",
    "deepseek",
    "dashscope",
    "moonshot",
    "aihubmix",
    "vllm",
    "zhipu",
]


def _configured_providers(config) -> list[str]:
    """Return list of provider names that have an API key (or api_base for vllm)."""
    result = []
    for name in PROVIDER_NAMES:
        provider = getattr(config.providers, name, None)
        if not provider:
            continue
        if provider.api_key or (name == "vllm" and provider.api_base):
            result.append(name)
    return result


def _provider_status(config) -> list[dict]:
    """Return provider status dicts for the template table."""
    result = []
    for name in PROVIDER_NAMES:
        provider = getattr(config.providers, name, None)
        if provider:
            has_key = bool(provider.api_key) or (name == "vllm" and bool(provider.api_base))
            result.append({"name": name, "configured": has_key})
    return result


def _auth_check(request: Request) -> JSONResponse | None:
    """Return a 401 JSONResponse if auth fails, else None."""
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


def _get_agent(request: Request):
    """Return the running agent or None."""
    return getattr(request.app.state, "agent", None)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
async def settings_page(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return RedirectResponse("/login")

    config = load_config()
    agent = _get_agent(request)
    current_model = agent.model if agent else "unknown"
    saved_model = config.agents.defaults.model

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "current_model": current_model,
            "saved_model": saved_model,
            "model_differs": current_model != saved_model,
            "providers": _provider_status(config),
        },
    )


# ---------------------------------------------------------------------------
# GET /api/config — full defaults dump
# ---------------------------------------------------------------------------


@router.get("/api/config")
async def api_config(request: Request):
    if err := _auth_check(request):
        return err

    config = load_config()
    defaults = config.agents.defaults
    defaults_dict = convert_to_camel(defaults.model_dump())

    return JSONResponse(
        {
            "defaults": defaults_dict,
            "providers": _configured_providers(config),
        }
    )


# ---------------------------------------------------------------------------
# POST /api/model — backward compat (existing model-only endpoint)
# ---------------------------------------------------------------------------


@router.post("/api/model")
async def api_update_model(request: Request):
    if err := _auth_check(request):
        return err

    agent = _get_agent(request)
    if not agent:
        return JSONResponse({"error": "agent not available"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    new_model = body.get("model", "").strip()
    new_provider = body.get("provider", "").strip() or None
    persist = body.get("persist", False)

    if not new_model:
        return JSONResponse({"error": "model is required"}, status_code=400)

    old_model = agent.model
    try:
        agent.model = new_model
        if new_provider and agent.provider_resolver:
            api_key, api_base = agent.provider_resolver.resolve(new_provider)
            if api_key:
                from nanobot.providers.litellm_provider import LiteLLMProvider

                agent.provider = LiteLLMProvider(
                    api_key=api_key,
                    api_base=api_base,
                    default_model=new_model,
                )
        if persist:
            config = load_config()
            config.agents.defaults.model = new_model
            if new_provider:
                config.agents.defaults.provider = new_provider
            save_config(config)

        return JSONResponse(
            {
                "ok": True,
                "model": new_model,
                "previous_model": old_model,
                "persisted": persist,
            }
        )
    except Exception as e:
        agent.model = old_model
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# PATCH helpers
# ---------------------------------------------------------------------------


def _apply_fields(target, body: dict, field_map: dict[str, str]) -> list[str]:
    """Set attributes on target from body using field_map {camelKey: attr_name}.

    Returns list of field names that were actually changed.
    """
    changed = []
    for camel_key, attr_name in field_map.items():
        if camel_key in body:
            val = body[camel_key]
            setattr(target, attr_name, val)
            changed.append(attr_name)
    return changed


# ---------------------------------------------------------------------------
# 1. PATCH /api/core — Core Agent (Live)
# ---------------------------------------------------------------------------

CORE_FIELDS = {
    "model": "model",
    "provider": "provider",
    "maxTokens": "max_tokens",
    "temperature": "temperature",
    "toolTemperature": "tool_temperature",
    "maxToolIterations": "max_tool_iterations",
    "timezone": "timezone",
}

CORE_RUNTIME = {
    "model": "model",
    "temperature": "temperature",
    "toolTemperature": "tool_temperature",
    "maxToolIterations": "max_iterations",
    "timezone": "timezone",
}


@router.patch("/api/core")
async def api_update_core(request: Request):
    if err := _auth_check(request):
        return err
    agent = _get_agent(request)
    if not agent:
        return JSONResponse({"error": "agent not available"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    config = load_config()
    defaults = config.agents.defaults
    _apply_fields(defaults, body, CORE_FIELDS)

    # Swap provider if specified
    new_provider = body.get("provider", "").strip() if body.get("provider") else None
    new_model = body.get("model", "").strip() if body.get("model") else None
    if new_provider and agent.provider_resolver:
        api_key, api_base = agent.provider_resolver.resolve(new_provider)
        if api_key:
            from nanobot.providers.litellm_provider import LiteLLMProvider

            agent.provider = LiteLLMProvider(
                api_key=api_key,
                api_base=api_base,
                default_model=new_model or agent.model,
            )

    # Apply runtime hot-swap
    for camel_key, runtime_attr in CORE_RUNTIME.items():
        if camel_key in body:
            setattr(agent, runtime_attr, body[camel_key])

    save_config(config)
    logger.info("Settings: core agent config updated")
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# 2. PATCH /api/context — Context Window (Live)
# ---------------------------------------------------------------------------

CONTEXT_FIELDS = {
    "maxContextTokens": "max_context_tokens",
    "systemPromptBudget": "system_prompt_budget",
    "historyBudget": "history_budget",
    "toolResultBudget": "tool_result_budget",
    "safetyMargin": "safety_margin",
}


@router.patch("/api/context")
async def api_update_context(request: Request):
    if err := _auth_check(request):
        return err
    agent = _get_agent(request)
    if not agent:
        return JSONResponse({"error": "agent not available"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    config = load_config()
    ctx = config.agents.defaults.context
    _apply_fields(ctx, body, CONTEXT_FIELDS)
    save_config(config)

    # Replace runtime context_config
    from nanobot.config.schema import ContextConfig

    agent.context_config = ContextConfig(**{k: v for k, v in body.items() if k in CONTEXT_FIELDS})

    logger.info("Settings: context window config updated")
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# 3. PATCH /api/compaction — Compaction (Live)
# ---------------------------------------------------------------------------

COMPACTION_FIELDS = {
    "enabled": "enabled",
    "model": "model",
    "provider": "provider",
    "threshold": "threshold",
    "keepRecent": "keep_recent",
}


@router.patch("/api/compaction")
async def api_update_compaction(request: Request):
    if err := _auth_check(request):
        return err
    agent = _get_agent(request)
    if not agent:
        return JSONResponse({"error": "agent not available"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    config = load_config()
    comp = config.agents.defaults.compaction
    _apply_fields(comp, body, COMPACTION_FIELDS)
    save_config(config)

    # Replace runtime compaction_config
    from nanobot.config.schema import CompactionConfig

    agent.compaction_config = CompactionConfig(
        **{k: v for k, v in body.items() if k in COMPACTION_FIELDS}
    )

    logger.info("Settings: compaction config updated")
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# 4. PATCH /api/memory — Memory (Live)
# ---------------------------------------------------------------------------

MEMORY_FIELDS = {
    "enabled": "enabled",
    "embeddingModel": "embedding_model",
    "embeddingProvider": "embedding_provider",
    "extractionModel": "extraction_model",
    "extractionProvider": "extraction_provider",
    "consolidationProvider": "consolidation_provider",
    "indexConversations": "index_conversations",
    "extractFacts": "extract_facts",
    "autoRecall": "auto_recall",
    "searchTopK": "search_top_k",
    "minSimilarity": "min_similarity",
    "recencyWeight": "recency_weight",
    "enableCoreMemory": "enable_core_memory",
    "enableEntities": "enable_entities",
    "enableConsolidation": "enable_consolidation",
    "enableProactive": "enable_proactive",
    "deterministicRecall": "deterministic_recall",
}


@router.patch("/api/memory")
async def api_update_memory(request: Request):
    if err := _auth_check(request):
        return err
    agent = _get_agent(request)
    if not agent:
        return JSONResponse({"error": "agent not available"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    config = load_config()
    mem = config.agents.defaults.memory
    _apply_fields(mem, body, MEMORY_FIELDS)
    save_config(config)

    # Replace runtime memory_config
    from nanobot.config.schema import MemoryConfig

    agent.memory_config = MemoryConfig(**{k: v for k, v in body.items() if k in MEMORY_FIELDS})

    logger.info("Settings: memory config updated")
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# 5. PATCH /api/memory-extraction — Memory Extraction (Live)
# ---------------------------------------------------------------------------

EXTRACTION_FIELDS = {
    "enabled": "enabled",
    "extractionModel": "extraction_model",
    "embeddingModel": "embedding_model",
    "embeddingProvider": "embedding_provider",
    "extractionInterval": "extraction_interval",
    "maxFactsPerExtraction": "max_facts_per_extraction",
    "maxLessonsPerExtraction": "max_lessons_per_extraction",
    "candidateThreshold": "candidate_threshold",
    "enablePreCompactionFlush": "enable_pre_compaction_flush",
    "enableToolLessons": "enable_tool_lessons",
}


@router.patch("/api/memory-extraction")
async def api_update_memory_extraction(request: Request):
    if err := _auth_check(request):
        return err
    agent = _get_agent(request)
    if not agent:
        return JSONResponse({"error": "agent not available"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    config = load_config()
    ext = config.agents.defaults.memory_extraction
    _apply_fields(ext, body, EXTRACTION_FIELDS)
    save_config(config)

    # Replace runtime extraction config + cached scalars
    from nanobot.config.schema import MemoryExtractionConfig

    agent._extraction_config = MemoryExtractionConfig(
        **{k: v for k, v in body.items() if k in EXTRACTION_FIELDS}
    )
    if "extractionInterval" in body:
        agent._extraction_interval = body["extractionInterval"]
    if "enablePreCompactionFlush" in body:
        agent._enable_pre_compaction_flush = body["enablePreCompactionFlush"]
    if "enableToolLessons" in body:
        agent._enable_tool_lessons = body["enableToolLessons"]

    logger.info("Settings: memory extraction config updated")
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# 6. PATCH /api/daemon — Daemon (config-only, restart required)
# ---------------------------------------------------------------------------

DAEMON_FIELDS = {
    "enabled": "enabled",
    "interval": "interval",
    "triageModel": "triage_model",
    "triageProvider": "triage_provider",
    "executionModel": "execution_model",
    "executionProvider": "execution_provider",
    "maxIterations": "max_iterations",
    "cooldownAfterAction": "cooldown_after_action",
    "cooldownHigh": "cooldown_high",
    "cooldownMedium": "cooldown_medium",
    "cooldownLow": "cooldown_low",
}

REGISTRY_FIELDS = {
    "enabled": "enabled",
    "pulseInterval": "pulse_interval",
    "staleThreshold": "stale_threshold",
    "monitorInterval": "monitor_interval",
}

SELF_EVOLVE_FIELDS = {
    "enabled": "enabled",
    "repoUrl": "repo_url",
    "autoMerge": "auto_merge",
    "testCommand": "test_command",
    "lintCommand": "lint_command",
}


@router.patch("/api/daemon")
async def api_update_daemon(request: Request):
    if err := _auth_check(request):
        return err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    config = load_config()
    daemon = config.agents.defaults.daemon

    # Top-level daemon fields
    _apply_fields(daemon, body, DAEMON_FIELDS)

    # Nested registry fields
    reg_body = body.get("registry", {})
    if reg_body:
        _apply_fields(daemon.registry, reg_body, REGISTRY_FIELDS)

    # Nested selfEvolve fields
    evolve_body = body.get("selfEvolve", {})
    if evolve_body:
        _apply_fields(daemon.self_evolve, evolve_body, SELF_EVOLVE_FIELDS)

    save_config(config)
    logger.info("Settings: daemon config updated (restart required)")
    return JSONResponse({"ok": True, "restart_required": True})


# ---------------------------------------------------------------------------
# 7. PATCH /api/intent — Intent (Live)
# ---------------------------------------------------------------------------

INTENT_FIELDS = {
    "enabled": "enabled",
    "llmFallback": "llm_fallback",
    "classifierModel": "classifier_model",
    "classifierProvider": "classifier_provider",
}


@router.patch("/api/intent")
async def api_update_intent(request: Request):
    if err := _auth_check(request):
        return err
    agent = _get_agent(request)
    if not agent:
        return JSONResponse({"error": "agent not available"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    config = load_config()
    intent = config.agents.defaults.intent
    _apply_fields(intent, body, INTENT_FIELDS)
    save_config(config)

    # Replace runtime intent config + update classifier
    from nanobot.config.schema import IntentConfig

    agent._intent_config = IntentConfig(**{k: v for k, v in body.items() if k in INTENT_FIELDS})
    if hasattr(agent, "_intent_classifier"):
        if "enabled" in body:
            agent._intent_classifier.enabled = body["enabled"]
        if "llmFallback" in body:
            agent._intent_classifier.llm_fallback = body["llmFallback"]

    logger.info("Settings: intent config updated")
    return JSONResponse({"ok": True})
