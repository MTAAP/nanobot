# Agent Swarm Concept for nanobot

## Executive Summary

This document proposes an **Agent Swarm Architecture** for nanobot that enables **parallel task execution** instead of the current sequential processing model. The swarm leverages the existing AgentRegistry (ACP protocol), SubagentManager, and HeartbeatService to create a multi-agent system that can work on multiple tasks concurrently.

## Current Architecture (Sequential)

```
User Message → Message Bus → Agent Loop (single instance)
    ↓
    LLM Call 1 → Tool Execution 1
    ↓
    LLM Call 2 → Tool Execution 2
    ↓
    LLM Call 3 → Tool Execution 3
    ↓
    Response
```

**Problems:**
- Sequential tool execution - one at a time
- Single agent bottleneck - only one agent instance
- Limited parallelization - can't work on multiple tasks
- Subagents are isolated - no coordination between them

## Proposed Architecture (Swarm)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Swarm Orchestrator                          │
│                   (New component in nanobot/agent/swarm/)         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
    │   Agent  │       │   Agent  │       │   Agent  │
    │   #1     │       │   #2     │       │   #3     │
    │  (Main)  │       │ Specialist│       │ Specialist│
    └────┬────┘       └────┬────┘       └────┬────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                    ┌─────────────▼─────────────┐
                    │   Shared Tool Registry   │
                    │   & Resource Pool     │
                    └──────────────────────────┘
```

## Key Components

### 1. Swarm Orchestrator (`nanobot/agent/swarm/orchestrator.py`)

**Responsibilities:**
- Decompose user requests into parallelizable tasks
- Dispatch tasks to available agents
- Aggregate results from multiple agents
- Handle dependencies between tasks
- Manage agent lifecycle (spawn/reuse/kill)
- Implement swarm coordination patterns

**Swarm Patterns:**

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Broadcast** | Send task to multiple agents, aggregate all results | "Research X from 5 sources" → 5 agents fetch, 1 aggregates |
| **Pipeline** | Chain agents where output of one becomes input of next | "Fetch → Parse → Transform → Store" |
| **Map-Reduce** | Distribute work, then aggregate | "Summarize 10 documents" → 10 agents summarize, 1 merges |
| **Specialist Pool** | Route tasks to agents by capability | "Code review" → code-reviewer agent, "Web search" → researcher agent |
| **Lead + Workers** | One agent coordinates, others execute | "Plan sprint" → planner agents, workers implement features |
| **Competitive** | Multiple agents solve, best result wins | "Generate 5 designs" → 5 agents, best selected |

### 2. Agent Pool (`nanobot/agent/swarm/pool.py`)

**Responsibilities:**
- Maintain pool of available agents
- Track agent states (idle/busy/stale)
- Match task requirements to agent capabilities
- Spawn/reuse agents based on workload
- Balance load across agents

**Agent Types:**

| Type | Capabilities | System Prompt |
|------|--------------|--------------|
| **Main Agent** | All tools, conversation management | Current agent loop |
| **Researcher** | web_search, web_fetch, memory_search | "Research: gather facts from web and memory" |
| **Coder** | read_file, write_file, edit_file, exec | "Code: implement features, fix bugs" |
| **Tester** | exec (pytest, ruff, lint) | "Test: verify code, run tests" |
| **Analyst** | memory_search, core_memory_read | "Analyze: interpret data, generate insights" |
| **Reporter** | write_file, message (for reports) | "Report: generate summaries, compile findings" |
| **Monitor** | exec (system checks), cron (schedules) | "Monitor: check system health, watch queues" |
| **Coordinator** | All tools (limited scope) | "Coordinate: manage subtasks, combine results" |

### 3. Task Graph (`nanobot/agent/swarm/graph.py`)

**Responsibilities:**
- Represent task dependencies
- Execute topological sort for parallelization
- Track task completion status
- Handle task failures and retries

**Task Graph Example:**

```
┌─────────┐
│ Task A  │  "Fetch specs from WedPilot"
└────┬────┘
     │
     ├─────────────────┐
     │                 │
┌────▼────┐    ┌────▼────┐
│ Task B  │    │ Task C  │
│"Analyze"│    │"Queue   │
└─────────┘    └────┬────┘
                     │
                ┌────▼────┐
                │ Task D  │
                │"Report" │
                └────┬────┘
                     │
                ┌────▼────┐
                │ Task E  │
                │"Notify" │
                └─────────┘
```

**Parallelization:**
- Tasks A → B & C → D → E: Sequential (dependencies)
- Tasks B & C: Parallel (no dependency)
- Maximum 3 agents running concurrently

### 4. Result Aggregator (`nanobot/agent/swarm/aggregator.py`)

**Responsibilities:**
- Collect results from multiple agents
- Merge/combine partial results
- Detect conflicts and resolve
- Format final response for user

**Aggregation Strategies:**

| Strategy | When Used | Example |
|----------|------------|---------|
| **Concatenation** | Independent results | 5 agents fetch web pages → concatenate summaries |
| **Merge (dict)** | Structured data | 3 agents analyze different APIs → merge to single dict |
| **Vote/Consensus** | Multiple opinions | 5 agents propose solutions → select most common |
| **Rank/Select** | Quality-based | 3 agents generate designs → select by score |
| **Synthesis** | Combines insights | Researchers gather facts → analyst synthesizes |

### 5. Swarm Coordinator (`nanobot/agent/swarm/coordinator.py`)

**Responsibilities:**
- High-level swarm orchestration
- Decide which pattern to use for a request
- Monitor swarm health and performance
- Handle failures and fallback

**Decision Logic:**

```
Request Analysis:
├─ Single task? → Main Agent (existing path)
├─ Multiple independent tasks? → Broadcast pattern
├─ Dependent tasks? → Pipeline pattern
├─ Same task for N items? → Map-Reduce pattern
├─ Domain-specific? → Specialist Pool pattern
└─ Complex multi-step? → Lead + Workers pattern
```

## Integration with Existing Components

### AgentRegistry (ACP Protocol)

**Current:**
- SQLite-backed agent and task tracking
- State transitions (idle → working → completed/failed)
- Proof of work verification

**Swarm Enhancements:**
- Add `swarm_id` to tasks (group related tasks)
- Add `parent_task_id` for dependencies
- Track agent capabilities in registry
- Add swarm metrics (parallelization efficiency, agent utilization)

### SubagentManager

**Current:**
- Spawn background subagents
- Isolated execution
- Result announcement via message bus

**Swarm Enhancements:**
- Reuse idle subagents from pool
- Agent specialization (specialist types)
- Swarm task dispatch (vs individual spawn)
- Result aggregation hooks

### HeartbeatService

**Current:**
- Periodic wake-up (~30 min)
- Daemon mode triage
- Task prioritization

**Swarm Enhancements:**
- Swarm health monitoring
- Agent pool status
- Parallel task queue
- Deadlock detection

### ToolRegistry

**Current:**
- Central tool registration
- Tool execution with validation
- Guardrail checks

**Swarm Considerations:**
- Tool access control (some agents get subset of tools)
- Resource locking (file writes, git operations)
- Tool result aggregation (multiple agents calling same tool)

## Implementation Plan

### Phase 1: Core Swarm Infrastructure (Week 1-2)

**Files:**
- `nanobot/agent/swarm/__init__.py`
- `nanobot/agent/swarm/orchestrator.py`
- `nanobot/agent/swarm/pool.py`
- `nanobot/agent/swarm/coordinator.py`

**Tasks:**
1. Implement SwarmOrchestrator class
2. Implement AgentPool class with capability matching
3. Integrate with existing SubagentManager
4. Add swarm dispatch to AgentLoop
5. Update config schema for swarm settings

**Tests:**
- Test broadcast pattern (parallel web searches)
- Test pipeline pattern (fetch → parse → store)
- Test agent pool management
- Test error handling and retries

### Phase 2: Task Graph & Parallelization (Week 2-3)

**Files:**
- `nanobot/agent/swarm/graph.py`
- `nanobot/agent/swarm/aggregator.py`

**Tasks:**
1. Implement TaskGraph with topological sort
2. Implement ResultAggregator with strategies
3. Add task decomposition logic to orchestrator
4. Implement conflict resolution

**Tests:**
- Test graph parallelization
- Test result aggregation
- Test failure handling in graph

### Phase 3: Integration & Optimization (Week 3-4)

**Files:**
- Update `nanobot/agent/loop.py` (swarm integration)
- Update `nanobot/registry/store.py` (swarm task tracking)
- Update `nanobot/agent/subagent.py` (pool-aware spawning)

**Tasks:**
1. Integrate SwarmOrchestrator into AgentLoop
2. Add swarm metrics to AgentRegistry
3. Implement intelligent pattern selection
4. Add swarm dashboard/status commands

**Tests:**
- End-to-end swarm workflows
- Performance benchmarks (sequential vs parallel)
- Load testing (many concurrent tasks)

## Swarm Configuration

```python
# config schema extension
class SwarmConfig(BaseModel):
    enabled: bool = False
    max_agents: int = 5  # Max concurrent agents
    agent_timeout_s: int = 300  # Per-task timeout
    pool_warmup: int = 2  # Pre-warm N agents
    parallel_patterns: list[str] = [
        "broadcast",
        "pipeline",
        "map_reduce",
        "specialist",
        "lead_workers",
    ]
    aggregation_strategy: str = "auto"  # auto/concatenate/merge/vote/rank
```

## Example Use Cases

### Use Case 1: OpenSpec Queue Monitoring (Current)

**Sequential (current):**
```
User: "Check all 42 implementation PRs"
    ↓
Agent: [Checks PR #1]
    ↓
Agent: [Checks PR #2]
    ↓
... (40 more sequential checks)
    ↓
Agent: "Report summary"
```

**Parallel (swarm):**
```
User: "Check all 42 implementation PRs"
    ↓
Coordinator: Decompose into 9 groups of 5 PRs each
    ↓
Swarm: Spawn 9 "Monitor" agents
    ├─ Agent 1: Checks PRs #1-5
    ├─ Agent 2: Checks PRs #6-10
    ├─ Agent 3: Checks PRs #11-15
    ...
    └─ Agent 9: Checks PRs #41-42
    ↓ (Parallel, ~30 seconds)
Aggregator: Combine all reports → Single summary
    ↓
Agent: "Report summary"
```

**Time Savings:** ~8 minutes → ~45 seconds (~10x speedup)

### Use Case 2: Gap Analysis Research

**Sequential (current):**
```
User: "Gap analysis for vendor hub"
    ↓
Agent: [Web search Joy vendor hub]
    ↓
Agent: [Web search Zola vendor hub]
    ↓
Agent: [Web search WeddingWire]
    ↓
Agent: [Analyze findings]
    ↓
Agent: [Create report]
```

**Parallel (swarm):**
```
User: "Gap analysis for vendor hub"
    ↓
Coordinator: Broadcast "Research X" to 3 Researchers
    ↓
Swarm:
    ├─ Researcher 1: Research Joy
    ├─ Researcher 2: Research Zola
    └─ Researcher 3: Research WeddingWire
    ↓ (Parallel, ~60 seconds)
Aggregator: Merge all research
    ↓
Coordinator: Dispatch "Analyze" to Analyst
    ↓
Coordinator: Dispatch "Create report" to Reporter
    ↓
Agent: "Here's the gap analysis report"
```

**Time Savings:** ~5 minutes → ~90 seconds (~3x speedup)

### Use Case 3: Multiple OpenSpec Interviews

**Sequential (current):**
```
Cron: "Answer OpenSpec interviews"
    ↓
Agent: [Check issue #177]
    ↓
Agent: [Answer question]
    ↓
Agent: [Check issue #176]
    ↓
Agent: [Answer question]
    ↓
... (40 more)
```

**Parallel (swarm):**
```
Cron: "Answer OpenSpec interviews"
    ↓
Coordinator: List all issues with active questions
    ↓
Swarm: Spawn N "Interview" agents (max 5 concurrent)
    ├─ Agent 1: Answer #177 (DACH)
    ├─ Agent 2: Answer #176 (Vendor)
    ├─ Agent 3: Answer #136 (API docs)
    ├─ Agent 4: Answer #134 (APM)
    └─ Agent 5: Answer #118 (Analytics)
    ↓ (Parallel)
When agent completes → pick next unanswered issue
    ↓
All complete → Report to cron job
```

**Time Savings:** ~40 minutes → ~8 minutes (~5x speedup)

## Benefits

| Metric | Current | Swarm | Improvement |
|--------|---------|-------|-------------|
| **Parallelism** | Single agent | 5+ concurrent | 5x+ |
| **Response Time** | Sequential task execution | Parallel task execution | 3-10x faster |
| **Resource Usage** | LLM calls idle between tools | Multiple LLM calls concurrent | Better efficiency |
| **Specialization** | Generalist agent | Specialist agents | Better quality per domain |
| **Fault Tolerance** | Single point of failure | Multiple agents | More resilient |
| **Scalability** | Bounded by single agent | Bounded by pool size | Horizontal scaling |

## Risks & Mitigations

| Risk | Mitigation |
|-------|-------------|
| **Deadlock** | Task graph cycle detection, timeout enforcement |
| **Resource contention** | Tool-level locking, file write coordination |
| **Agent starvation** | Fair scheduling, priority queues |
| **Cost escalation** | Token budgeting, limit max agents |
| **Complexity** | Gradual rollout, extensive testing |
| **Debugging difficulty** | Trace spans per agent, swarm logs |

## Research-Based Best Practices

### From CrewAI
- **Role-based agents**: Each agent has clear role and capabilities
- **Collaborative intelligence**: Agents work together on complex tasks
- **Visual editor**: Human-readable agent coordination (optional future)

### From OpenAI Swarm
- **Ergonomic orchestration**: Lightweight, easy to understand
- **Agent networks**: Interconnected agents communicating directly
- **Modular design**: Easy to add/remove agent types

### From ACP Protocol Research
- **Loosely connected swarm**: No central coordination bottleneck
- **DID-based identity**: Trust and discovery without central auth
- **Asynchronous communication**: RESTful task exchanges

### From Swarms.io (kyegomez/swarms)
- **Multi-provider support**: Different LLMs for different agents
- **Interoperable interfaces**: Unified schema for agent communication
- **Streaming results**: Real-time progress updates

## Future Enhancements

### Phase 4: Advanced Swarm Features (Month 2-3)

1. **Dynamic Agent Scaling**
   - Auto-scale pool based on workload
   - Hibernation for idle agents
   - Burst mode for urgent tasks

2. **Learning Swarm**
   - Learn from past task distributions
   - Predict optimal agent count per task type
   - Self-tune aggregation strategies

3. **Visual Swarm Dashboard**
   - Real-time agent visualization
   - Task graph explorer
   - Performance metrics
   - Manual intervention controls

4. **Cross-Swarm Communication**
   - Multiple swarms for different domains
   - Inter-swarm task delegation
   - Global resource pool

## Summary

The proposed Agent Swarm Architecture transforms nanobot from a **single sequential agent** to a **multi-agent collaborative system** that:

1. **Decomposes tasks** into parallelizable units
2. **Dispatches work** to specialized agents
3. **Aggregates results** from multiple agents
4. **Provides 3-10x speedup** for suitable workloads
5. **Leverages existing infrastructure** (AgentRegistry, SubagentManager)
6. **Follows ACP protocol** for interoperability
7. **Integrates with OpenSpec, OpenCode, WedPilot workflows**

**Key Insight**: Not every task benefits from swarmization. The swarm coordinator must intelligently decide when to use sequential vs parallel execution based on task nature, dependencies, and available resources.

---

*Document version: 1.0*  
*Date: 2026-02-11*  
*Author: nanobot self-evolution*
