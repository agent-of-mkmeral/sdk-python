# Subagents: A Vended `use_agent` Tool

**Status**: Proposed

**Date**: 2026-08-12

**Issue**: TBD

**Scope**: Python SDK first; the same shape ports to the TypeScript SDK later.

**Related**:
- [#3392: `use_agent` delegation shim](https://github.com/strands-agents/harness-sdk/pull/3392) — draft, parked pending this design pass
- [#3346: agent-as-tool delegation (`delegate=True`)](https://github.com/strands-agents/harness-sdk/pull/3346) — open, overlaps with the extraction protocol below
- [#3380: swarm shim](https://github.com/strands-agents/harness-sdk/pull/3380) — draft, shares the multiagent conventions this design needs
- [#3499: tool parameter named `agent` is silently dropped](https://github.com/strands-agents/harness-sdk/issues/3499) — open; constrains parameter naming in any derived schema

## Context

We want `use_agent` — model-driven delegation, where the model can shape the child's instructions, tools, and model at call time — in the SDK as a vended tool. `Agent.as_tool()` stays as the primitive for developer-authored children; this covers the model-authored side. [#3392](https://github.com/strands-agents/harness-sdk/pull/3392) is a first cut and is parked pending this design.

A straight port of `strands_tools.use_agent` is not the answer. Three problems have to be fixed on the way in.

## Problem

**1. One fixed parameter list cannot serve every harness.** A regulated harness wants the child's prompt and tools fixed and reviewed; a coding harness wants named roles; a research harness wants ad-hoc roles and a model-tier choice. Whatever fixed list we vend, someone forks a semver-bound tool to change it.

**2. The result has no contract.** Every delegation path returns `str(result)` — the text of the child's *final message* — with no declared shape, no machine-readable usage, and no handle to the child. (One known edge case: a plugin that re-invokes the child, e.g. `GoalLoop`, makes the final message a follow-up rather than the deliverable. v1 treats this as an edge case; configurable extraction is future work.)

**3. The child is not a member of the harness.** `strands_tools.use_agent` and #3392 both construct a bare `Agent(...)` — no sandbox, no interventions, no skills. In a harness that gates `shell` in `BeforeToolCall`, delegation is a structural bypass of the gate.

Two smaller points: a blocking call with no handle makes wait/resume/cancel a breaking change later; and the tools-repo `use_agent` carries defects that must not be ported ([Appendix B](#appendix-b); #3392 already cuts most of them).

## Goals and Non-Goals

Goals:

- One vended tool spanning "model supplies only a task" to "model supplies everything", configured, not forked — with the model-facing schema derived from that configuration and inspectable.
- Children built through the harness's construction path, inheriting sandbox and policy by default.
- `make_use_agent()` with no arguments ≈ #3392's surface; the result shape and factory signature reserve the seams — a handle, an extraction hook, plugin inheritance — so later extensions stay additive.

Non-Goals (v1): background execution, parallel fan-out (that's `swarm`/`graph`, [#3380](https://github.com/strands-agents/harness-sdk/pull/3380)), cross-process sessions, replacing `as_tool`, configurable output extraction, and plugin inheritance (see Future Work).

## Prior Art

| Implementation | Child identity authored by | Model may set | What comes back |
|---|---|---|---|
| [Codex](https://github.com/openai/codex) `spawn_agent` | Developer (roles in `agents.toml` + `agents/` dirs, layered) | `message`, `agent_type`, `fork_turns`, `model`, `reasoning_effort` — **each removable by config** | `{agent_id}`; content fetched via `wait_agent` |
| [opencode](https://github.com/sst/opencode) `task` | Developer (agent definition files) | `description`, `prompt`, `subagent_type`, `task_id`, `background` | Last text part in a `<task_result>` envelope |
| Claude Code `Task` | Developer (subagent files, five scopes) | `description`, `prompt`, `subagent_type` | The child's report |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) `as_tool` | Developer (an instance) | `input`, or a supplied Pydantic type | `custom_output_extractor` (default: last message, documented as such) |
| [Google ADK](https://github.com/google/adk-python) `AgentTool` | Developer | Derived from child's input schema | `skip_summarization` toggle |
| `strands_tools.use_agent` | **Model** | `prompt`, `system_prompt`, `tools`, `model_provider`, `model_settings` | Response + model id + metrics, as text |

Three takeaways. **Named developer-authored roles won everywhere**; model-authored system prompts are at most a secondary path. **Everyone has an extraction knob except us** — and Codex's current protocol goes furthest: `wait_agent` returns no content at all, only which children have updates. **Codex derives the spawn tool's schema from configuration**, literally `properties.remove("model")` per option flags ([Appendix A](#appendix-a)) — which settles that problem 1 is solvable with a shipping mechanism, not a research idea. Our own `make_sleep(max_duration=...)` already derives the *description* from config; this extends that to parameters. Extended matrix in [Appendix C](#appendix-c).

## Proposal

### Recommended: a factory whose tool schema follows its configuration

Everyone knows today's `use_agent`: one fixed parameter list — `prompt`, `system_prompt`, `tools`, `model_provider`, `model_settings` — take it or fork it. This proposal keeps the tool but makes it a factory: `make_use_agent(...)` returns the tool, and the configuration decides **which parameters the model sees**. A pre-configured `use_agent` default instance ships alongside, per the vended-tool convention. The whole surface:

```python
def make_use_agent(
    *,
    # Axes: who decides each part of the child. Each contributes zero or one
    # model-facing parameter to the tool schema.
    instructions: Open | Choice | Fixed = Open(max_bytes=8 * 1024),
    tools: Narrow | Choice | Fixed | Inherit = Narrow(),
    model: Inherit | Choice | Fixed = Inherit(),
    context: Fixed | Choice = Fixed("none"),          # "none" | "all" | int turns

    # Named roles the model picks from; adds an `agent_type` enum parameter.
    presets: Mapping[str, Preset] | Sequence[Skill] | Path | None = None,

    # How the child is constructed — the harness-membership hook.
    builder: AgentBuilder | None = None,

    # What comes back.
    result_model: type[BaseModel] | None = None,
    max_output_bytes: int = 8 * 1024,

    # Per-run limits.
    max_depth: int = 3,
    max_children: int = 12,
    wall_clock_s: float = 300.0,
    max_task_bytes: int = 32 * 1024,

    # Lifecycle (reserved in v1; see Future Work).
    allow_resume: bool = False,
    allow_define: bool = False,

    name: str = "use_agent",
    description: str | None = None,
) -> DecoratedFunctionTool: ...


use_agent = make_use_agent()      # vended default; ≈ #3392's surface
```

Supporting types:

```python
class Open:    max_bytes: int                                # model writes the value freely
class Choice:  options: Sequence[str | int] | ModelRouter    # model picks from a set
class Narrow:  allow: Sequence[str] | None = None            # model may only remove; None = parent's tools
class Fixed:   value: Any                                    # developer decides; no parameter
class Inherit: pass                                          # parent's live value; no parameter

@dataclass(frozen=True)
class Preset:
    instructions: str | None = None
    tools: Sequence[str] | None = None
    model: str | Model | None = None
    context: str | int | None = None                  # "none" | "all" | int turns
    result_model: type[BaseModel] | None = None
    description: str = ""

@dataclass(frozen=True)
class AgentSpec:
    """A fully resolved request for a child agent. Configuration plus the model's arguments."""
    task: str | list[ContentBlock]
    instructions: str | None
    tools: Sequence[str]
    model: Model | None
    context: str | int
    result_model: type[BaseModel] | None
    preset_name: str | None
    depth: int

class AgentBuilder(Protocol):
    def __call__(self, spec: AgentSpec, *, parent: Agent) -> Agent: ...
```

The rest of this section walks the signature, parameter by parameter.

#### The axes: `instructions`, `tools`, `model`, `context`

A subagent needs four things decided — instructions, tools, model, context — and for each the developer says *who* decides it:

| Authority | Meaning | Effect on the tool schema |
|---|---|---|
| `Open(max_bytes=...)` | Model writes the value freely | Adds a string parameter |
| `Choice(...)` | Model picks from a developer-supplied set | Adds an enum parameter; options described in the tool description |
| `Narrow(allow=...)` | Model may only *remove* from a set, never add | Adds an array parameter with enum items |
| `Fixed(value)` | Developer decides | No parameter |
| `Inherit()` | Take the parent's value | No parameter |

`Narrow` is the only mode safe by construction — a child can never gain a capability the parent lacked (opencode enforces the same property by deriving child permissions from the parent's, then adding denies); its allow-list is re-validated against the parent's *live* tool set at call time, so a tool the parent has since lost is an error, not a grant. `Fixed` and `Inherit` both remove the parameter but are not redundant: `Fixed` supplies a literal, `Inherit` tracks the parent's *live* value — its currently routed model, its current tool set. Not every mode applies to every axis (`instructions` cannot be `Narrow`ed or `Inherit`ed; `tools` cannot be `Open`, a child never gets tools the parent lacks); the legal combinations are the type hints above, and illegal ones raise `ValueError` at factory time.

The consequence: the regulated harness and the research harness are the same code path with different `inputSchema`s, and the developer can print the schema to see exactly what the model was handed:

```python
from strands.vended_tools.use_agent import make_use_agent, Fixed, Choice, Open, Narrow, Inherit

locked = make_use_agent(
    presets={"reviewer": reviewer, "researcher": researcher},
    instructions=Fixed(None),   # roles own the prompt; parameter removed
    tools=Fixed(None),          # roles own the tools; parameter removed
    model=Inherit(),            # parameter removed
    context=Fixed("none"),      # parameter removed
)
sorted(locked.tool_spec["inputSchema"]["json"]["properties"])
# ['agent_type', 'task']
```

The `context` axis deserves one note: it takes `"none" | "all" | N` turns (Codex's `fork_turns`), replacing `as_tool`'s boolean and `use_agent`'s implicit always-none. A reviewer that sees the last three turns beats one re-briefed in prose; a researcher should start clean. Like the other axes it can be fixed or offered to the model — but `"all"` forks the entire parent transcript (tool results included) into the child, so it is not offered in any vended default or example until its interaction with conversation managers and secret hygiene is traced (see Consequences).

#### `presets`: named roles

A `Preset` is a partially applied configuration — instructions, tools, model, result contract — registered under a name; passed as a mapping, a directory of definition files, or `Skill` objects. When presets exist, the tool gains an `agent_type` enum and the description lists each role. Precedence is fixed: a field the preset defines is `Fixed` for that role — the axes govern only fields the preset leaves unset, the model's arguments are validated against whichever applies, and a preset's `result_model` overrides the factory-level one. Preset fields are developer-authored configuration, so a `Model` instance there is the developer's own — the credential-injection concern from Appendix B applies to model-facing parameters, never to preset contents. Definition files are skill-shaped, so one dialect describes both top-level and delegated agents (opencode's `mode: subagent | primary | all`). Ad-hoc `instructions=Open(...)` stays available and is the vended default, for continuity with today's `use_agent`.

#### `builder`: construction through injection

The tool resolves config plus the model's arguments into an `AgentSpec` and hands it to an `AgentBuilder`. The default builder is roughly #3392's body; a harness passes its own, and the child is built the way the parent was — same sandbox, same interventions, same skills. This is the fix for problem 3, and it is Codex's split too: discovery resolves the invocation, the runtime only starts it, the host injects the spawn capability.

An illustrative default builder (`cfg` is the factory's resolved configuration, closed over):

```python
def make_default_builder(cfg: UseAgentConfig) -> AgentBuilder:
    def build(spec: AgentSpec, *, parent: Agent) -> Agent:
        tools = [parent.tool_registry.registry[n] for n in spec.tools]
        if spec.depth >= cfg.max_depth - 1:
            tools = [t for t in tools if t.tool_name != cfg.name]   # structural depth limit
        return Agent(
            model=spec.model or parent.model,
            system_prompt=spec.instructions,
            tools=tools,
            messages=fork_context(parent.messages, spec.context),
            # no plugins, no sandbox: the default builder constructs a plain agent.
            # Harness builders supply their own sandbox and hooks (see Consequences).
            structured_output_model=spec.result_model,
        )
    return build
```

The default builder does not copy the parent's plugins into the child; a harness builder decides what the child runs (filtered plugin inheritance is future work). And it does not silently weaken a guarded parent: when the parent has a sandbox or registered hooks that the default builder cannot carry over, the call fails with a configuration error telling the developer to supply a builder — otherwise delegation would be a structural bypass of exactly the gates problem 3 describes.

#### `result_model` and `max_output_bytes`: declared output, not `str(result)`

v1 keeps today's extraction — interrupts, else structured output, else the final message's text — with one addition: if the child ran more than one invocation (a plugin re-invoked it; the edge case from problem 2) the tool logs a loud warning. The stronger option is `result_model: type[BaseModel]`, which constructs the child with `structured_output_model=` — an existing `Agent` parameter — so the parent receives fields (`{summary, findings, files_changed, open_questions}`) instead of "whichever message was last". Opt-in per preset, since it constrains the child. A general `output_extractor` hook is future work, deferred so it lands once as a protocol shared with `as_tool`/`swarm`/`graph`; adding it later is purely additive. The tool's own result is declared via `outputSchema`:

```python
{"agent_id": str,           # handle, so wait/resume/cancel stay additive later
 "status": "completed" | "failed" | "cancelled",
 "output": str,             # the child's report
 "structured": dict | None, # present when result_model is set
 "truncated": bool, "artifact": str | None,   # offload target when over max_output_bytes;
                                              # written by the tool on the parent side — the child never picks the path
 "usage": {"input_tokens": int, "output_tokens": int},
 "execution_time_ms": int}
```

Two deliberate choices. Child interrupts are *not* a status: they propagate to the parent as `ToolInterruptEvent`, exactly as `Agent.as_tool` does today, so human-in-the-loop flows keep working through delegation. And the model id and metrics prose are absent — usage is a field and a trace attribute, not conversation content. (`outputSchema` is a declared contract; not every model provider enforces it.)

#### `max_depth`, `max_children`, `wall_clock_s`, `max_task_bytes`: per-run budgets, structural depth

These are accounted per *run* — per-call caps are defeated by calling twice. A run is one top-level invocation of the parent agent; the account lives in the parent's invocation state, not in the factory or the module-level instance (which would silently share budgets across agents), it is updated atomically since tool calls can run concurrently, and multiple `use_agent` instances attached to one agent draw from the same account. Depth is structural: at the limit the child is built *without* the delegation tool, so recursion is impossible rather than discouraged (opencode's `childToolDenies` does the same) — the depth counter travels in the spec and is enforced by the tool itself, so it holds across every child this factory produces. Budget exhaustion returns `{"status": "failed", ...}` as a tool result, not an exception, so the model can adapt.

#### `allow_resume`, `allow_define`: reserved

Both default to `False` and are inert in v1; they reserve the lifecycle seam (resume against the returned `agent_id`, runtime role registration) so enabling them later is additive. See Future Work.

**Pros:**

- One tool spans every harness posture; the model-facing surface is a deployment decision, inspectable by printing a schema.
- `make_use_agent()` ≈ #3392, so this is additive, not competing.
- Children built through the harness's own path inherit sandbox and policy; the result is data with a handle, not prose.
- Composes existing SDK features — `ModelRouter` candidates, `structured_output_model`, the dynamic `tool_spec` setter, `outputSchema` — rather than adding mechanism.

**Cons:**

- Five exported names (`Open`, `Choice`, `Fixed`, `Inherit`, `Narrow`) is real conceptual weight for one tool.
- A schema that varies by configuration is harder to document and benchmark; the common configurations need to ship as documented presets.
- v1 ships without configurable extraction and without plugin inheritance; `result_model` and the multi-invocation warning cover the known edge case until the extractor hook lands.
- The *default* builder still constructs a plain agent — harness membership arrives only when a harness passes its builder. Correctness by convention there, not construction.

### Alternative: ship #3392 as it stands

Fixed three-parameter tool; harnesses needing more write their own. **Pros:** written, small, ships in days. **Cons:** keeps all three problems and has no handle in the result. **Verdict:** not wrong, incomplete — adopt it as the factory's *default configuration* instead.

### Alternative: fixed schema plus a permissions object

Keep six parameters; validate the model's arguments against a developer policy at call time. **Pros:** one schema to document, no derived-schema machinery. **Cons:** the model sees parameters it may not use and finds out by being refused — wasted turns, wasted prompt budget on every request. Codex started near this and moved to removing the properties. **Verdict:** telling the model less beats telling it more and saying no.

### Alternative: ship the full lifecycle now

Five tools — spawn, wait, send input, resume, close (Codex's shape). **Pros:** most capable; parallelism and follow-ups come free. **Cons:** five tools of surface and concurrency semantics before one blocking call works well; parents must be taught not to poll. **Verdict:** v1 blocks; the reserved `agent_id` keeps this path open, and opencode shows most of it can arrive later as two parameters on one tool.

### Alternative: `as_tool` only

**Pros:** one existing primitive. **Cons:** no model-chosen roles, nothing varies per call, and it has the same result-contract gap. **Verdict:** answers a different question; both should exist.

## Developer Experience

### Minimal: the vended default

```python
from strands.vended_tools import use_agent

agent = Agent(model=model, tools=[shell, file_editor, use_agent])
```

Model may write ad-hoc instructions and narrow the tool set; child inherits the model, starts fresh. Today's behavior, deliberately.

<details>
<summary>Resulting tool schema</summary>

```json
{
  "name": "use_agent",
  "description": "Delegate a self-contained task to a subagent. The subagent starts with no knowledge of this conversation, so the task must contain everything it needs. It runs your current model and returns a final report. Do not poll for progress or duplicate the subagent's work.",
  "inputSchema": {"json": {
    "type": "object",
    "properties": {
      "task":         {"type": "string", "description": "The self-contained task, including all context the subagent needs."},
      "instructions": {"type": "string", "description": "Optional system prompt defining the subagent's role."},
      "tools":        {"type": "array",
                       "items": {"type": "string", "enum": ["shell", "file_editor"]},
                       "description": "Subset of your tools to grant the subagent. Omit to grant the full set shown."}
    },
    "required": ["task"]
  }},
  "outputSchema": {"json": {"$ref": "#/SubagentResult"}}
}
```

</details>

### Typical: named roles in a harness

```python
use_agent = make_use_agent(
    presets=Path(".strands/agents"),   # reviewer.md, researcher.md, test_writer.md
    instructions=Fixed(None),
    tools=Fixed(None),
    model=Inherit(),
    builder=my_harness.build_agent,
    max_depth=2,
    max_children=8,
)
```

A role definition — a skill with two extra fields:

```markdown
---
name: reviewer
description: Reviews a diff for correctness, API fit, and test coverage.
tools: [shell, file_editor]
model: deep
result_model: ReviewReport
---
You are a senior reviewer. Read the diff, then ...
```

<details>
<summary>Resulting tool schema (two parameters; roles listed in the description)</summary>

```json
{
  "name": "use_agent",
  "description": "Delegate a task to a specialized subagent.\n\nAvailable subagents:\n- reviewer: reviews a diff for correctness, API fit, and test coverage (tools: shell, file_editor)\n- researcher: answers questions about external sources (tools: http_request)\n- test_writer: writes and runs tests (tools: shell, file_editor)",
  "inputSchema": {"json": {
    "type": "object",
    "properties": {
      "task":       {"type": "string", "description": "The self-contained task for the subagent."},
      "agent_type": {"type": "string", "enum": ["reviewer", "researcher", "test_writer"]}
    },
    "required": ["task", "agent_type"]
  }}
}
```

</details>

### Maximal: a research harness

```python
use_agent = make_use_agent(
    instructions=Open(16 * 1024),
    tools=Narrow(),
    model=Choice(router),              # named candidates from a ModelRouter
    context=Choice(["none", 3]),          # integer options render as strings in the enum
    builder=my_harness.build_agent,
)
```

<details>
<summary>Resulting tool schema</summary>

```json
{
  "properties": {
    "task":         {"type": "string"},
    "instructions": {"type": "string"},
    "tools":        {"type": "array", "items": {"type": "string", "enum": ["shell", "file_editor", "http_request"]}},
    "model":        {"type": "string", "enum": ["fast", "balanced", "deep"],
                     "description": "fast: cheap, short tasks. balanced: the default. deep: hard reasoning, slower and costlier. Omit to inherit the caller's model."},
    "context":      {"type": "string", "enum": ["none", "3"],
                     "description": "How much of this conversation the subagent can see. More context costs more tokens."}
  },
  "required": ["task"]
}
```

</details>

The `model` axis consumes `ModelRouter` candidates — developer-named tiers over real `Model` instances — so the tool never touches provider strings or credentials.

### Errors

Budget/depth exhaustion is a tool result (`{"status": "failed", "output": "Subagent budget exhausted: 8 of 8 children used in this run."}`) so the model can adapt. At the depth limit the child simply lacks the tool (`child.tool_names` has no `use_agent`). Configuration errors raise at factory time, per the `make_sleep` precedent:

```python
>>> make_use_agent(tools=Choice([]))
ValueError: tools=Choice([]) offers no options; use Fixed(None) to remove the parameter.
```

## Consequences

Easier: a harness gets a posture-matched subagent tool from configuration; "what can my model do?" is answered by printing a schema; children inherit sandbox and policy; the result is a declared shape with a handle, so lifecycle tools and future extensions stay additive.

Needs attention:

- **A configurable schema is harder to document and evaluate** — ship the three configurations above as documented, tested presets.
- **The default builder constructs a plain agent.** It refuses guarded parents (see Proposal) rather than quietly building an ungated child, but harness membership still arrives only through a supplied builder.
- **`context="all"` vs conversation managers and session state** is untraced; needs a pass before that value is offered to a model.

Migration: nothing breaks. `strands_tools.use_agent` keeps working (migration note: `model_provider`/`model_settings` → `ModelRouter` candidates, not a passthrough). `as_tool` is untouched.

## Development Plan

- **Phase 0 (one PR):** multiagent conventions as code — depth counter, per-run budget accounting, status vocabulary (today a convention doc duplicated across #3392 and #3380 with colliding exports).
- **Phase 1 (v1):** the factory — axes, presets, injected builder, `result_model`, context dial, per-run budgets, declared result with `agent_id`, the multi-invocation warning. Default extraction only; no plugin inheritance. Blocking but streaming and cancellable. #3392 becomes the default configuration.
- **Later:** the Future Work items below, each as its own design or PR.

Prototype phase 1 in one of the internal harnesses first — a `build_agent` function and role files already exist there — before any of it becomes public API.

## Future Work

Reserved by the v1 surface, intentionally not designed here:

- **Configurable output extraction** — an `output_extractor` hook, landed once as a protocol shared with `as_tool`, `swarm`, and `graph` so the four paths don't diverge. This is also the proper fix for the re-invocation edge case in problem 2. One coordination point that can't wait: [#3346](https://github.com/strands-agents/harness-sdk/pull/3346)'s `delegate=True` should not hard-code `str(result)`, or the hook can't be added cleanly later.
- **Plugin inheritance** — `inherit_plugins`, gated on a way for plugins to declare themselves top-level-only. Until then the default builder passes no plugins and harness builders decide.
- **Lifecycle** — `session`/resume against the reserved `agent_id`; background execution as a transport on the spec; `define_agent` for runtime role registration.

## Key Decisions

To align on during review:

1. **The factory-with-derived-schema shape itself** — versus shipping #3392's fixed surface (the alternatives above).
2. **The result contract** — `agent_id` handle reserved from day one; interrupts propagate rather than appearing as a status.
3. **v1 scope** — no configurable extraction, no plugin inheritance, no resume (all Future Work).
4. **`result_model` in v1?** Recommendation: yes, opt-in per preset — it rides on the existing `structured_output_model`, no new primitive.
5. **Should the vended default expose ad-hoc `instructions`?** Recommendation: yes, for continuity with today's `use_agent` — accepting that prior art favors named roles; presets are the documented recommended posture.
6. **#3346 coordination** — needs a comment before it merges so `delegate` doesn't hard-code `str(result)`.

## Willingness to Implement

Yes — phase 0 and v1, prototyped in an internal harness first.

---

<a id="appendix-a"></a>
<details>
<summary><b>Appendix A: How Codex derives its spawn schema from configuration</b></summary>

From `codex-rs/core/src/tools/handlers/multi_agents_spec.rs` in [openai/codex](https://github.com/openai/codex):

```rust
pub struct SpawnAgentToolOptions {
    pub available_models: Vec<ModelPreset>,
    pub agent_type_description: String,
    pub expose_agent_type: bool,
    pub hide_agent_type_model_reasoning: bool,
    pub expose_spawn_agent_model_overrides: bool,
    pub multi_agent_version: MultiAgentVersion,
    pub usage_hint_text: Option<String>,
}

let mut properties = spawn_agent_common_properties_v2(&options.agent_type_description);
if !options.expose_agent_type              { properties.remove("agent_type"); }
if options.hide_agent_type_model_reasoning { properties.remove("service_tier"); }
if !options.expose_spawn_agent_model_overrides {
    properties.remove("model");
    properties.remove("reasoning_effort");
}
```

Points that map onto this design:

- **Parameters appear and disappear per configuration.** This is the authority-mode idea, without the vocabulary.
- **The description is composed from configuration** — the available model menu is interpolated, along with guidance that spawned agents inherit the caller's model by default. Our `sleep_description(max_duration)` is the same convention at a smaller scale.
- **`agent_type` resolves against a catalog** built from `agents.toml` roles plus an `agents/` directory, per configuration layer, warning on duplicate role names within a layer. This is our `presets`, and it is also what Claude Code and opencode do.
- **The tools declare output schemas** — `{agent_id, nickname}` for spawn, and a per-agent status union for wait. Ours should too.
- **The surface is explicitly versioned** (`MultiAgentVersion`, with v1 and v2 exposing different parameters), because they expected to break it. Reserving `agent_id` is how we try to avoid needing that.
- **Construction is split from resolution.** The runtime receives a fully resolved invocation and the code comments that discovery owns rendering the prompt while the runtime only starts it in a forked context. The host injects the spawn capability through a trait. That is `AgentSpec` plus an injected builder.
- **A child receives a whole configuration**, not a bare agent — the argument for builder injection.

Two hygiene details worth copying: Codex encrypts the child's task payload, and OpenAI's structured-input preamble instructs the model to treat a schema as data rather than instructions.

</details>

<a id="appendix-b"></a>
<details>
<summary><b>Appendix B: What not to port from <code>strands_tools.use_agent</code></b></summary>

For the record, since "bring `use_agent` to the SDK" should not mean porting this file. `strands-agents/tools/src/strands_tools/use_agent.py`, 289 lines:

1. **`model_provider` and `model_settings` are a credential and endpoint injection surface.** They reach a model factory that accepts `api_key` and `endpoint_url`. #3392 removed them, correctly. The replacement is named `ModelRouter` candidates or a developer-supplied key map.
2. **Silent fallback to the parent's model.** Three separate `except` paths reassign the child's model to the parent's. The model asks for something cheap, gets something expensive, and only a debug log records it.
3. **The child is invoked synchronously.** No `await`, no cancellation, no streaming. A long-running child blocks the loop and produces no output until it finishes.
4. **Three text blocks come back** — the response, the model id, and a formatted metrics dump — and all three enter the parent's conversation on every call.
5. **`tools=None` inherits the entire parent registry**, with no depth limit, no output cap, and no allowlist validation beyond a warning for unknown names.
6. **No membership.** The child is a bare `Agent`, so any policy the harness enforces through hooks does not apply to it.

Worth keeping: the idea of a per-call model choice (as a name, not a provider configuration), and returning usage information (as data, not prose).

</details>

<a id="appendix-c"></a>
<details>
<summary><b>Appendix C: Extended prior art</b></summary>

### Context inheritance and lifecycle

| Implementation | Context inheritance | Lifecycle | Child inherits harness policy |
|---|---|---|---|
| Codex `spawn_agent` | `fork_turns`: `"none"`, `"all"`, or a turn count | Five tools: spawn, wait, send input, resume, close. Spawn returns only an id | Yes — the child is a forked thread with the full configuration |
| opencode `task` | Fresh, or a resumed session via `task_id` | One tool plus `background` and `task_id` | Yes — permissions derived from the parent, then narrowed |
| Claude Code `Task` | Fresh | One blocking call | Yes |
| OpenAI Agents SDK | `session` / `conversation_id` | One call; `max_turns` per tool | Run config, hooks, and approval settings per tool |
| Google ADK | Session state | One call | — |
| `strands_tools.use_agent` | None | One blocking call | No |
| `Agent.as_tool()` | `preserve_context: bool` | One call; interrupts propagate | Whatever the wrapped instance has |

### Output extraction, specifically

| Implementation | Extraction surface | Default |
|---|---|---|
| OpenAI Agents SDK | `as_tool(custom_output_extractor=...)` | The agent's last message, documented as such |
| Google ADK | `AgentTool(skip_summarization=...)` | Summarize |
| opencode | Last text part, in a `<task_result>` envelope | Last text part |
| Codex (earlier protocol) | `wait_agent` returns the final message per agent id | Final message |
| Codex (current protocol) | `wait_agent` returns no content, only which agents have updates | Nothing; the parent must ask |
| Strands | `str(result)` | Final message text |

### Our own harnesses

Two internal harnesses independently arrived at the same shape and are the strongest evidence for this design.

One builds subagents through the harness's own `build_agent` function, selects models by tier name rather than model id, defines roles as files, streams the child's work as it happens — and gates four plugins on "am I a subagent?" — the experience behind this design's choice to leave plugin inheritance out of v1.

The other forwards the parent's full configuration into the child through an injected builder, exposes exactly one role, takes `task` as its only parameter, and returns `str(result)`. It is the minimal correct version of this tool, and it is what `presets` with a single entry should produce.

</details>

<a id="appendix-d"></a>
<details>
<summary><b>Appendix D: Sources and verification</b></summary>

**Verified against `harness-sdk` at `main` while writing this document:**

| Claim | Location |
|---|---|
| `as_tool(name, description, preserve_context)`; no `delegate` parameter yet | `agent/agent.py:1005` |
| `AgentResult.__str__` returns interrupts, then structured output, then the final message's text | `agent/agent_result.py:61-91` |
| Agent-as-tool extraction uses structured output if present, else `str(result)` | `agent/_agent_as_tool.py:220-233` |
| `Agent(structured_output_model=...)` exists, making `result_model` a one-line change | `agent/agent.py:188` |
| `RoutingCandidate(model, name, description)` — named model candidates already exist | `models/routing/router.py:74` |
| `Plugin` has `name`, `hooks`, `tools`, `init_agent`, and no scope concept | `plugins/plugin.py` |
| Tool specs may be replaced at runtime, explicitly to support dynamic configuration | `tools/decorator.py:551` |
| `ToolSpec.outputSchema` exists as an optional field | `types/tools.py:38` |
| Vended-tool convention: `make_sleep(...)`, a module-level default instance, and a description derived from configuration | `vended_tools/sleep/{sleep,types}.py` |
| A resume starts a full new invocation cycle | `vended_plugins/goal/plugin.py`, `hooks/events.py` |
| PR/issue states: #3392 open draft, #3346 open and unmerged, #3380 open draft, #3499 open | GitHub API |

**External sources**, read as source rather than documentation: `openai/codex` (`codex-rs/core/src/tools/handlers/multi_agents_spec.rs`, `multi_agents/{spawn,wait,send_input,resume_agent,close_agent}.rs`, `ext/agent/src/lib.rs`, `core/src/config/agent_roles.rs`); `sst/opencode` (`tool/task.ts`, `tool/task.txt`, `agent/agent.ts`, `agent/subagent-permissions.ts`); `openai/openai-agents-python` (`agents/agent.py`, `agent_tool_input.py`); `google/adk-python` (`tools/agent_tool.py`). Claude Code is closed source, so its behavior here is taken from published documentation.

**Not re-verified for this document:** exact line numbers in the two internal harnesses, and the external file contents quoted above, which were read during the research pass rather than re-fetched while writing.

</details>
