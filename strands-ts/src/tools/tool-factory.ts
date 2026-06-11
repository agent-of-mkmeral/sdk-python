import type { InvokableTool } from './tool.js'
import { FunctionTool } from './function-tool.js'
import type { FunctionToolConfig } from './function-tool.js'
import type { JSONValue } from '../types/json.js'
import { z } from 'zod'
import { ZodTool, type ToolCustomConfig, type ZodToolConfig } from './zod-tool.js'

/**
 * Checks whether a value is a Zod schema type.
 *
 * @param value - The value to check
 * @returns True if the value is a Zod schema
 */
function isZodType(value: unknown): value is z.ZodType {
  return value instanceof z.ZodType
}

/**
 * Creates a ZodTool from a Zod schema and callback function.
 *
 * The returned tool supports `clone()` for deriving customized variants
 * (metadata overrides and, for tools that declare a `config` object,
 * configuration overrides).
 *
 * @typeParam TInput - Zod schema type for input validation
 * @typeParam TReturn - Return type of the callback function
 * @typeParam TConfig - Tool-specific configuration type (see ZodToolConfig.config)
 * @param config - Tool configuration with Zod schema
 * @returns A ZodTool with typed input, output, and configuration
 */
export function tool<
  TInput extends z.ZodType,
  TReturn extends JSONValue = JSONValue,
  TConfig extends ToolCustomConfig = Record<never, never>,
>(config: ZodToolConfig<TInput, TReturn, TConfig>): ZodTool<TInput, TReturn, TConfig>

/**
 * Creates an InvokableTool from a JSON schema and callback function.
 *
 * @param config - Tool configuration with optional JSON schema
 * @returns An InvokableTool with unknown input
 */
export function tool(config: FunctionToolConfig): InvokableTool<unknown, JSONValue>

/**
 * Creates an InvokableTool from either a Zod schema or JSON schema configuration.
 *
 * When a Zod schema is provided as `inputSchema`, input is validated at runtime and
 * the callback receives typed input. When a JSON schema (or no schema) is provided,
 * the callback receives `unknown` input with no runtime validation.
 *
 * @example
 * ```typescript
 * import { tool } from '@strands-agents/sdk'
 * import { z } from 'zod'
 *
 * // With Zod schema (typed + validated)
 * const calculator = tool({
 *   name: 'calculator',
 *   description: 'Adds two numbers',
 *   inputSchema: z.object({ a: z.number(), b: z.number() }),
 *   callback: (input) => input.a + input.b,
 * })
 *
 * // Derive a customized variant — same callback, new metadata:
 * const adder = calculator.clone({ name: 'adder', description: 'Adds two integers' })
 *
 * // With JSON schema (untyped, no validation)
 * const greeter = tool({
 *   name: 'greeter',
 *   description: 'Greets a person',
 *   inputSchema: {
 *     type: 'object',
 *     properties: { name: { type: 'string' } },
 *     required: ['name'],
 *   },
 *   callback: (input) => `Hello, ${(input as { name: string }).name}!`,
 * })
 * ```
 *
 * @param config - Tool configuration
 * @returns An InvokableTool that implements the Tool interface with invoke() method
 */
export function tool(
  config: ZodToolConfig<z.ZodType | undefined, JSONValue, ToolCustomConfig> | FunctionToolConfig
): InvokableTool<unknown, JSONValue> {
  if (config.inputSchema && isZodType(config.inputSchema)) {
    return new ZodTool(config as ZodToolConfig<z.ZodType, JSONValue, ToolCustomConfig>)
  }

  return new FunctionTool(config as FunctionToolConfig)
}
