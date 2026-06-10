/**
 * Tool customization wrapper.
 *
 * Provides {@link customizeTool}, a free function that wraps an existing
 * {@link Tool} and overrides its model-facing surface metadata (name and
 * description) while delegating all execution to the wrapped tool.
 *
 * This solves the "vended tool customization" problem: built-in tools such as
 * `bash` and `fileEditor` are module-level singletons whose implementation
 * details (callbacks, Zod schemas) are private. A free function cannot reach
 * those privates, so customization works by **delegation**: the wrapper
 * reports new metadata and forwards `stream()`/`invoke()` to the original.
 *
 * Naming note: this is intentionally NOT called `cloneTool` — no state is
 * copied. The wrapper shares the inner tool's runtime state. For example, the
 * host `bash` tool keys its persistent session on the agent instance, so a
 * customized `bash` used by the same agent shares the same session as the
 * original (this is usually what you want).
 */

import { Tool, isValidToolName } from './tool.js'
import type { InvokableTool, ToolContext, ToolStreamGenerator } from './tool.js'
import type { ToolSpec } from './types.js'
import { ToolValidationError } from '../errors.js'

/**
 * Surface metadata overrides accepted by {@link customizeTool}.
 *
 * Deliberately limited to `name` and `description`.
 *
 * **Why `inputSchema` is not overridable:** input validation for Zod-based
 * tools happens *inside* the wrapped tool's private callback, against its
 * original Zod schema. A wrapper can only change the JSON-schema surface shown
 * to the model — it cannot change what the inner tool actually validates. If
 * we allowed `inputSchema` overrides, the spec the model sees and the
 * validation that runs could silently diverge, producing confusing runtime
 * validation errors for inputs the advertised schema claims are valid. If you
 * need a different input schema, build a new tool with `tool()` (or use a
 * factory like `makeBash`) so the schema and validation stay in lockstep.
 */
export interface ToolOverrides {
  /**
   * New tool name. Must satisfy the provider-accepted tool name format:
   * 1–64 characters of letters, digits, underscores, or hyphens.
   * The wrapper keeps `name` and `toolSpec.name` consistent.
   */
  name?: string

  /**
   * New tool description shown to the model. Must be non-empty.
   * The wrapper keeps `description` and `toolSpec.description` consistent.
   */
  description?: string
}

/**
 * Internal wrapper that reports overridden surface metadata and delegates
 * execution to the wrapped tool. Extends {@link Tool} so `instanceof Tool`
 * checks (and the `Tool[]` contracts used by the registry and agent) hold.
 */
class CustomizedTool extends Tool {
  readonly name: string
  readonly description: string
  readonly toolSpec: ToolSpec

  protected readonly _inner: Tool

  constructor(inner: Tool, overrides: ToolOverrides) {
    super()
    this._inner = inner
    this.name = overrides.name ?? inner.name
    this.description = overrides.description ?? inner.description
    // Rebuild the spec so name/description/inputSchema are always consistent
    // with the surface properties. The inner spec object is never mutated.
    this.toolSpec = {
      ...inner.toolSpec,
      name: this.name,
      description: this.description,
    }
  }

  /**
   * Delegates streaming execution to the wrapped tool.
   *
   * The context is passed through unchanged: `toolContext.toolUse.name`
   * reflects the externally visible (possibly overridden) name — i.e. the
   * name the model actually called. Built-in tool implementations read
   * `toolUse.input` and `toolUse.toolUseId`, not `toolUse.name`.
   */
  stream(toolContext: ToolContext): ToolStreamGenerator {
    return this._inner.stream(toolContext)
  }
}

/**
 * Wrapper variant that additionally preserves the typed `invoke()` method of
 * an {@link InvokableTool}.
 */
class CustomizedInvokableTool<TInput, TReturn> extends CustomizedTool implements InvokableTool<TInput, TReturn> {
  constructor(inner: InvokableTool<TInput, TReturn>, overrides: ToolOverrides) {
    super(inner, overrides)
  }

  /**
   * Delegates direct invocation (including the inner tool's input validation
   * and error semantics) to the wrapped tool.
   */
  invoke(input: TInput, context?: ToolContext): Promise<TReturn> {
    return (this._inner as InvokableTool<TInput, TReturn>).invoke(input, context)
  }
}

function isInvokable(tool: Tool): tool is InvokableTool<unknown, unknown> {
  return typeof (tool as Partial<InvokableTool<unknown, unknown>>).invoke === 'function'
}

/**
 * Wraps an existing tool with overridden surface metadata (name and/or
 * description), delegating all execution to the wrapped tool.
 *
 * The returned tool:
 * - reports the overridden `name`/`description`, with `toolSpec.name`,
 *   `toolSpec.description`, and `toolSpec.inputSchema` kept consistent,
 * - delegates `stream()` (and `invoke()` for invokable tools) to the
 *   wrapped tool, preserving its validation and error behavior,
 * - is a `Tool` subclass, so `instanceof Tool` holds,
 * - leaves the original tool completely untouched.
 *
 * Because this is delegation (not a clone), runtime state is shared with the
 * original. For example, the host `bash` tool keys its persistent session per
 * agent, so a customized `bash` on the same agent shares that session.
 *
 * Input schemas cannot be overridden — see {@link ToolOverrides} for why.
 *
 * @example Customize a vended singleton (agent-dev story)
 * ```typescript
 * import { customizeTool } from '@strands-agents/sdk'
 * import { bash } from '@strands-agents/sdk/vended-tools'
 *
 * const myBash = customizeTool(bash, {
 *   description: 'Executes bash commands. Prefer pipes over temp files.',
 * })
 * const agent = new Agent({ tools: [myBash] })
 * ```
 *
 * @example Prefix a tool name (used internally for sandbox-vended tools)
 * ```typescript
 * const prefixed = customizeTool(someTool, { name: `sandbox_${someTool.name}` })
 * ```
 *
 * @param tool - The tool to wrap. Any `Tool` implementation works.
 * @param overrides - Surface metadata overrides; omitted fields pass through.
 * @returns A new wrapped tool; the original is not modified.
 * @throws ToolValidationError if an override value is invalid.
 */
export function customizeTool<TInput, TReturn>(
  tool: InvokableTool<TInput, TReturn>,
  overrides: ToolOverrides
): InvokableTool<TInput, TReturn>
export function customizeTool(tool: Tool, overrides: ToolOverrides): Tool
export function customizeTool(tool: Tool, overrides: ToolOverrides): Tool {
  if (overrides.name !== undefined && !isValidToolName(overrides.name)) {
    throw new ToolValidationError(
      `Invalid tool name override '${overrides.name}': must be 1-64 characters of letters, digits, underscores, or hyphens`
    )
  }
  if (overrides.description !== undefined && overrides.description.length === 0) {
    throw new ToolValidationError('Tool description override must be a non-empty string')
  }

  return isInvokable(tool) ? new CustomizedInvokableTool(tool, overrides) : new CustomizedTool(tool, overrides)
}
