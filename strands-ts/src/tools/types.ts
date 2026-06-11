import type { JSONSchema, JSONValue } from '../types/json.js'
import type { z } from 'zod'

/**
 * Status of a tool execution.
 * Indicates whether the tool executed successfully or encountered an error.
 */
export type ToolResultStatus = 'success' | 'error'

/**
 * Tool-spec overrides accepted by every vended-tool factory.
 *
 * This is the SDK-wide convention for customizing the model-facing surface of a
 * built-in tool (`makeBash`, `makeHostBash`, `makeFileEditor`, ...). Factories
 * separate two concerns in their options:
 *
 * 1. **Execution binding** — runtime wiring such as `sandbox`, declared by each
 *    factory alongside these overrides.
 * 2. **Tool-spec overrides** — the metadata and schema the model sees, declared
 *    here and shared by all factories.
 *
 * Every vended-tool factory takes a single flat options object typed as
 * `<BindingOptions> & ToolSpecOverrides<TSchema>` so that call sites read
 * naturally and TypeScript's excess-property checking flags typos.
 *
 * **`inputSchema` semantics:** the replacement schema is used both to generate
 * the JSON schema advertised to the model and to validate inputs at runtime —
 * the tool is built fresh, so there is no stale-schema hazard. Each factory
 * constrains `TSchema` so the schema's *parsed output* must remain assignable
 * to the input contract its callback handles (e.g. `{ command, timeout? }` for
 * `makeBash`). Incompatible schemas fail to compile.
 *
 * **Excess-property caveat (TypeScript limitation):** typo'd keys are only
 * rejected on inline object literals. Spreading an intermediate variable
 * (`makeBash({ ...opts })`) bypasses excess-property checking whenever at least
 * one property overlaps; if *no* properties overlap, TypeScript's weak-type
 * check still rejects it. Prefer inline literals when calling factories.
 *
 * @typeParam TSchema - Zod schema type for the `inputSchema` override.
 *   Factories narrow this to enforce their input contract.
 */
export interface ToolSpecOverrides<TSchema extends z.ZodType = z.ZodType> {
  /**
   * Override the tool's name (e.g. to register two differently-bound variants
   * of the same tool on one agent).
   */
  name?: string

  /**
   * Override the tool's description, typically to add deployment-specific
   * context for the model (working directory, host name, usage guidance).
   */
  description?: string

  /**
   * Replace the tool's input schema. Used for both model-facing JSON schema
   * generation and runtime validation. The schema's parsed output must satisfy
   * the factory's input contract (enforced via the factory's generic bound),
   * and the schema must be representable as JSON Schema — schemas containing
   * `.transform()` throw at tool-creation time.
   */
  inputSchema?: TSchema
}

/**
 * Specification for a tool that can be used by the model.
 * Defines the tool's name, description, and input schema.
 */
export interface ToolSpec {
  /**
   * The unique name of the tool.
   */
  name: string

  /**
   * A description of what the tool does.
   * This helps the model understand when to use the tool.
   */
  description: string

  /**
   * JSON Schema defining the expected input structure for the tool.
   * If omitted, defaults to an empty object schema allowing no input parameters.
   */
  inputSchema?: JSONSchema
}

/**
 * Represents a tool usage request from the model.
 * The model generates this when it wants to use a tool.
 */
export interface ToolUse {
  /**
   * The name of the tool to execute.
   */
  name: string

  /**
   * Unique identifier for this tool use instance.
   * Used to match tool results back to their requests.
   */
  toolUseId: string

  /**
   * The input parameters for the tool.
   * Must be JSON-serializable.
   */
  input: JSONValue
}

/**
 * Specifies how the model should choose which tool to use.
 *
 * - `{ auto: {} }` - Let the model decide whether to use a tool
 * - `{ any: {} }` - Force the model to use one of the available tools
 * - `{ tool: { name: 'name' } }` - Force the model to use a specific tool
 */
export type ToolChoice = { auto: Record<string, never> } | { any: Record<string, never> } | { tool: { name: string } }
