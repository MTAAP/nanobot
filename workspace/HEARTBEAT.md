# Heartbeat Tasks

This file is checked every 30 minutes by your nanobot agent.
Add tasks below that you want to agent to work on periodically.

If this file has no tasks (only headers and comments), the agent will skip processing.

## Active Tasks

### Self-Evolution Tasks

- [ ] **Weekly**: Web search for AI agent error logging best practices 2025 and evaluate against current ErrorLogger implementation
- [ ] **Weekly**: Web search for AI agent observability frameworks and new features that could benefit nanobot
- [ ] **Weekly**: Review top 10 error categories from ErrorLogger and create feature issues for recurring patterns
- [ ] **Weekly**: Research new AI agentic frameworks (AutoGen, ReWOO, LangGraph, etc.) and evaluate for nanobot
- [ ] **Weekly**: Review nanobot repo GitHub issues and pull requests for feature requests and bugs
- [ ] **Monthly**: Review and update SELF_EVOLUTION.md roadmap based on findings
- [ ] **Monthly**: Generate summary report of self-evolution progress and learnings
- [ ] **Continuous**: Use `/learn` skill whenever you encounter a new technology or need deeper understanding

### Integration Tasks

- [ ] **High Priority**: Integrate ErrorLogger into nanobot agent loop (nanobot/agent/loop.py)
- [ ] **High Priority**: Register HealthCheckTool in agent loop (in _register_default_tools method)
- [ ] **High Priority**: Add OpenSpec workflows to nanobot repo (.github/workflows/openspec-interview-pr.yml and openspec-proposal-from-pr.yml)

### WedPilot Queue Monitoring (Auto-Fix)

- [ ] **Every Check**: Diagnose OpenSpec implementation queue health
  - Check for PRs stuck in `ready-for-implementation` state >2 hours
  - Check for workflows that skipped due to missing specs
  - Check for runners that are offline or stuck
  - Verify queue processor is running on strato VPS
- [ ] **Auto-Fix Actions**:
  - If PR missing spec files: Copy from develop branch and re-trigger
  - If workflow skipped: Re-trigger with correct parameters
  - If runner stuck: Alert user (cannot fix automatically)
  - If queue processor down: Attempt restart via SSH
- [ ] **Success Criteria**: All queued PRs should have active workflow runs or be processing on strato
- [ ] **Medium Priority**: Add execution tracing to capture full agent decision flows (not just errors)
- [ ] **Medium Priority**: Add decision logging to capture LLM choices before tool execution
- [ ] **Medium Priority**: Add token/latency metrics per tool call
- [ ] **Low Priority**: Add quality evaluation metrics for agent responses

## Completed

<!-- Move completed tasks here or delete them -->
