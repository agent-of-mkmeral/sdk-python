import type { InvokableTool, ToolContext, ToolStreamGenerator } from './tool.js'
import { Tool } from './tool.js'
import type { ToolSpec } from './types.js'
import type { JSONSchema, JSONValue } from '../types/json.js'
import { FunctionTool } from './function-tool.js'
import { z, ZodVoid } from 'zod'
import { zodSchemaToJsonSchema } from './zod-utils.js'

/**
 * Helper type to infer input type from Zod schema or default to never.
 */
type ZodInferred<TInput> = TInput extends z.ZodType ? z.infer<TInput> : never

/**
 * Constraint for tool-specific configuration objects.
 *
 * Config keys must not collide with tool metadata keys (`name`, `description`,
 * `inputSchema`) because `clone()` accepts metadata overrides and config
 * overrides in a single flat object.
 */
export type ToolCustomConfig = object & {
  name?: never
  description?: never
  inputSchema?: never
}

/**
 * Metadata overrides accepted by {@link ZodTool.clone} when the input schema
 * is not being replaced.
 */
export interface ZodToolCloneOverrides {
  /** Replacement tool name. Defaults to the source tool's name. */
  name?: string

  /** Replacement tool description. Defaults to the source tool's description. */
  description?: string
}

/**
 * Configuration for creating a Zod-based tool.
 *
 * @typeParam TInput - Zod schema type for input validation
 * @typeParam TReturn - Return type of the callback function
 * @typeParam TConfig - Tool-specific configuration surfaced to the callback and overridable via clone()
 */
export interface ZodToolConfig<
  TInput extends z.ZodType | undefined,
  TReturn extends JSONValue = JSONValue,
  TConfig extends ToolCustomConfig = Record<never, never>,
> {
  /** The name of the tool */
  name: string

  /** A description of what the tool does (optional) */
  description?: string

  /**
   * Zod schema for input validation and JSON schema generation.
   * If omitted or z.void(), the tool takes no input parameters.
   */
  inputSchema?: TInput

  /**
   * Tool-specific configuration passed to the callback as its third argument.
   *
   * Declaring a config object defines the tool's customization surface: derived
   * tools created with {@link ZodTool.clone} can override any subset of these
   * keys with full type safety. Keys must not collide with `name`,
   * `description`, or `inputSchema`.
   *
   * @example
   * ```typescript
   * const defaultConfig: { sandbox?: Sandbox } = {}
   * const sandboxBash = tool({
   *   name: 'bash',
   *   description: '...',
   *   inputSchema: schema,
   *   config: defaultConfig,
   *   callback: (input, context, config) => {
   *     const sandbox = config?.sandbox ?? context?.agent.sandbox
   *     // ...
   *   },
   * })
   *
   * // Bind to a specific sandbox without touching the callback:
   * const bound = sandboxBash.clone({ sandbox: mySandbox })
   * ```
   */
  config?: TConfig

  /**
   * Callback function that implements the tool's functionality.
   *
   * @param input - Validated input matching the Zod schema
   * @param context - Optional execution context
   * @param config - The tool's configuration object (including any clone() overrides)
   * @returns The result (can be a value, Promise, or AsyncGenerator)
   */
  callback: (
    input: ZodInferred<TInput>,
    context?: ToolContext,
    config?: TConfig
  ) => AsyncGenerator<unknown, TReturn, never> | Promise<TReturn> | TReturn
}

/**
 * Zod-based tool implementation.
 * Extends Tool abstract class and implements InvokableTool interface.
 */
export class ZodTool<
  TInput extends z.ZodType | undefined,
  TReturn extends JSONValue = JSONValue,
  TConfig extends ToolCustomConfig = Record<never, never>,
>
  extends Tool
  implements InvokableTool<ZodInferred<TInput>, TReturn>
{
  /**
   * Internal FunctionTool for delegating stream operations.
   */
  private readonly _functionTool: FunctionTool

  /**
   * Zod schema for input validation.
   * Note: undefined is normalized to z.void() in constructor, so this is always defined.
   */
  private readonly _inputSchema: z.ZodType

  /**
   * Tool-specific configuration passed to the callback.
   */
  private readonly _config: TConfig

  /**
   * User callback function.
   */
  private readonly _callback: (
    input: ZodInferred<TInput>,
    context?: ToolContext,
    config?: TConfig
  ) => AsyncGenerator<unknown, TReturn, never> | Promise<TReturn> | TReturn

  constructor(config: ZodToolConfig<TInput, TReturn, TConfig>) {
    super()
    const { name, description = '', inputSchema, callback } = config

    // Normalize undefined to z.void() to simplify logic throughout
    this._inputSchema = inputSchema ?? z.void()
    this._callback = callback
    this._config = config.config ?? ({} as TConfig)

    let generatedSchema: JSONSchema

    // Handle z.void() - use default empty object schema
    if (this._inputSchema instanceof ZodVoid) {
      generatedSchema = {
        type: 'object',
        properties: {},
        additionalProperties: false,
      }
    } else {
      generatedSchema = zodSchemaToJsonSchema(this._inputSchema)
    }

    // Create a FunctionTool with a validation wrapper
    this._functionTool = new FunctionTool({
      name,
      description,
      inputSchema: generatedSchema,
      callback: (
        input: unknown,
        toolContext: ToolContext
      ): AsyncGenerator<JSONValue, JSONValue, never> | Promise<JSONValue> | JSONValue => {
        // Only validate if schema is not z.void() (after normalization, it's never undefined)
        const validatedInput = this._inputSchema instanceof ZodVoid ? input : this._inputSchema.parse(input)
        // Execute user callback with validated input
        return callback(validatedInput as ZodInferred<TInput>, toolContext, this._config) as
          | AsyncGenerator<JSONValue, JSONValue, never>
          | Promise<JSONValue>
          | JSONValue
      },
    })
  }

  /**
   * The unique name of the tool.
   */
  get name(): string {
    return this._functionTool.name
  }

  /**
   * Human-readable description of what the tool does.
   */
  get description(): string {
    return this._functionTool.description
  }

  /**
   * OpenAPI JSON specification for the tool.
   */
  get toolSpec(): ToolSpec {
    return this._functionTool.toolSpec
  }

  /**
   * Executes the tool with streaming support.
   * Delegates to internal FunctionTool implementation.
   *
   * @param toolContext - Context information including the tool use request and invocation state
   * @returns Async generator that yields ToolStreamEvents and returns a ToolResultBlock
   */
  stream(toolContext: ToolContext): ToolStreamGenerator {
    return this._functionTool.stream(toolContext)
  }

  /**
   * Creates a derived tool with overridden metadata and/or configuration,
   * keeping this tool's metadata where no override is given. This tool is not
   * modified.
   *
   * Overrides are a single flat object:
   * - **Metadata** — `name`, `description`, and optionally `inputSchema`.
   *   Replacing the input schema updates both the model-facing JSON schema and
   *   runtime validation, and retypes `invoke()` accordingly. The replacement
   *   schema must parse to output assignable to the callback's input contract
   *   (extra fields are allowed; incompatible shapes are compile errors).
   * - **Config** — any subset of the tool's declared config keys (see
   *   {@link ZodToolConfig.config}). Overrides are shallow-merged over the
   *   source tool's config; unknown keys are compile errors.
   *
   * @example
   * ```typescript
   * // Metadata customization (any ZodTool):
   * const myBash = bash.clone({ description: 'Prefer pipes | over temp files.' })
   *
   * // Bind a sandbox-aware tool to a concrete sandbox (tools that declare config):
   * const bound = sandboxBash.clone({ sandbox: mySandbox, description: 'Runs in Docker.' })
   *
   * // Replace the input schema — drives spec AND validation:
   * const audited = sandboxBash.clone({
   *   inputSchema: z.object({
   *     command: z.string(),
   *     reason: z.string().describe('Why this command is needed.'),
   *   }),
   * })
   * ```
   *
   * @param overrides - Metadata and/or config overrides
   * @returns A new, independent ZodTool sharing this tool's callback
   */
  clone<TNewInput extends z.ZodType<ZodInferred<TInput>> | undefined = undefined>(
    overrides: ZodToolCloneOverrides & { inputSchema?: TNewInput } & Partial<TConfig>
  ): ZodTool<TNewInput extends z.ZodType ? TNewInput : TInput, TReturn, TConfig> {
    const { name, description, inputSchema, ...configOverrides } = overrides

    const derived = new ZodTool<z.ZodType, TReturn, TConfig>({
      name: name ?? this.name,
      description: description ?? this.description,
      // Preserve the normalized schema (z.void() for schema-less tools) when not overridden.
      inputSchema: (inputSchema as z.ZodType | undefined) ?? this._inputSchema,
      config: { ...this._config, ...(configOverrides as Partial<TConfig>) },
      // Sound cast: the signature constraint guarantees any replacement schema's
      // output is assignable to the callback's declared input type.
      callback: this._callback as (
        input: z.infer<z.ZodType>,
        context?: ToolContext,
        config?: TConfig
      ) => AsyncGenerator<unknown, TReturn, never> | Promise<TReturn> | TReturn,
    })

    return derived as ZodTool<TNewInput extends z.ZodType ? TNewInput : TInput, TReturn, TConfig>
  }

  /**
   * Invokes the tool directly with type-safe input and returns the unwrapped result.
   *
   * Unlike stream(), this method:
   * - Returns the raw result (not wrapped in ToolResult)
   * - Consumes async generators and returns only the final value
   * - Lets errors throw naturally (not wrapped in error ToolResult)
   *
   * @param input - The input parameters for the tool
   * @param context - Optional tool execution context
   * @returns The unwrapped result
   */
  async invoke(input: ZodInferred<TInput>, context?: ToolContext): Promise<TReturn> {
    // Only validate if schema is not z.void() (after normalization, it's never undefined)
    const validatedInput = this._inputSchema instanceof ZodVoid ? input : this._inputSchema.parse(input)

    // Execute callback with validated input
    const result = this._callback(validatedInput as ZodInferred<TInput>, context, this._config)

    // Handle different return types
    if (result && typeof result === 'object' && Symbol.asyncIterator in result) {
      const generator = result as AsyncGenerator<unknown, TReturn, undefined>
      let iterResult = await generator.next()
      while (!iterResult.done) {
        iterResult = await generator.next()
      }
      return iterResult.value
    } else {
      // Regular value or Promise - return directly
      return await result
    }
  }
}
