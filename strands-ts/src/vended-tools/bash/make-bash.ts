/**
 * Sandbox-bound bash tool factory.
 *
 * Separated from bash.ts to avoid pulling Node dependencies (child_process, Buffer)
 * into sandbox implementations that import this.
 */

import { tool } from '../../tools/tool-factory.js'
import type { InvokableTool } from '../../tools/tool.js'
import type { ToolSpecOverrides } from '../../tools/types.js'
import { z } from 'zod'
import { SandboxTimeoutError } from '../../sandbox/errors.js'
import type { Sandbox } from '../../sandbox/base.js'
import type { BashOutput } from './types.js'
import { BashTimeoutError, BashSessionError, SANDBOX_BASH_DESCRIPTION } from './types.js'

const sandboxBashInputSchema = z.object({
  command: z.string().describe('The bash command to execute.'),
  timeout: z.number().positive().optional().describe('Timeout in seconds (default: 120).'),
})

/**
 * Input contract for the sandbox bash tool.
 *
 * Custom `inputSchema` overrides passed to {@link makeBash} must *parse to* a
 * type assignable to this shape — the callback only reads `command` and
 * `timeout`, so any extra fields produced by a custom schema are accepted and
 * ignored. This is enforced at compile time via the `TSchema` bound on
 * {@link makeBash}.
 */
export interface SandboxBashToolInput {
  /** The bash command to execute. */
  command: string
  /** Timeout in seconds (default: 120). */
  timeout?: number | undefined
}

/**
 * Execution-binding options for {@link makeBash}.
 *
 * Kept separate from {@link ToolSpecOverrides} to make the two concerns of the
 * factory explicit: *what runs the command* (this interface) vs *what the model
 * sees* (the overrides).
 */
export interface BashBindingOptions {
  /**
   * Sandbox bound at tool-creation time. If omitted, the tool resolves
   * `context.agent.sandbox` at call time instead.
   */
  sandbox?: Sandbox
}

/**
 * Options for {@link makeBash}: execution binding plus the SDK-wide
 * {@link ToolSpecOverrides} convention, as a single flat bag.
 *
 * The flat shape is intentional — it matches keyword-argument factories in the
 * Python SDK (`make_bash(sandbox=..., name=..., ...)`) and keeps call sites
 * free of nesting, while the type-level split (intersection of
 * {@link BashBindingOptions} and {@link ToolSpecOverrides}) documents which
 * fields bind execution and which override the tool spec.
 *
 * @typeParam TSchema - Type of the `inputSchema` override. Must parse to a
 *   {@link SandboxBashToolInput}-compatible output.
 */
export type MakeBashOptions<TSchema extends z.ZodType<SandboxBashToolInput> = z.ZodType<SandboxBashToolInput>> =
  BashBindingOptions & ToolSpecOverrides<TSchema>

/**
 * Create a *stateless, sandbox-routed* bash tool. Each call runs in a fresh
 * shell inside the sandbox; no state persists between calls.
 *
 * If `sandbox` is provided it is bound at creation time; otherwise the tool
 * reads `context.agent.sandbox` at call time. Used by sandbox implementations
 * in `getTools()` and by users who want a customized bash tool.
 *
 * For the *persistent host-session* variant (Node.js only), see `makeHostBash`
 * in `./bash.js`.
 *
 * A custom `inputSchema` replaces both the model-facing JSON schema and the
 * runtime validation. Its parsed output must satisfy
 * {@link SandboxBashToolInput} (compile-time enforced); extra parsed fields are
 * ignored by the callback.
 *
 * Note: prior to the {@link ToolSpecOverrides} convention, `inputSchema` was
 * typed as an unconstrained `z.ZodType`, which let schemas through whose output
 * lacked `command` and crashed at call time. The constraint makes those
 * (already broken) usages compile errors; all sound usages are unaffected.
 *
 * @example
 * ```typescript
 * // Bind a sandbox, customize the description the model sees:
 * const tool = makeBash({
 *   sandbox: mySandbox,
 *   description: 'Runs commands in the build container.',
 * })
 *
 * // Replace the input schema — validation uses the new schema, and
 * // `tool.invoke` is typed from it:
 * const audited = makeBash({
 *   sandbox: mySandbox,
 *   inputSchema: z.object({
 *     command: z.string(),
 *     reason: z.string().describe('Why this command is needed.'),
 *   }),
 * })
 * ```
 */
export function makeBash<TSchema extends z.ZodType<SandboxBashToolInput> = typeof sandboxBashInputSchema>(
  options: MakeBashOptions<TSchema> = {}
): InvokableTool<z.output<TSchema>, BashOutput> {
  const inputSchema: z.ZodType<SandboxBashToolInput> = options.inputSchema ?? sandboxBashInputSchema
  const boundSandbox = options.sandbox

  const built = tool({
    name: options.name ?? 'bash',
    description: options.description ?? SANDBOX_BASH_DESCRIPTION,
    inputSchema,
    callback: async (input, context): Promise<BashOutput> => {
      if (!context) {
        throw new Error('Tool context is required for bash operations')
      }

      const sandbox = boundSandbox ?? context.agent.sandbox
      try {
        const result = await sandbox.execute(input.command, { timeout: input.timeout ?? 120 })
        return { output: result.stdout, error: result.stderr }
      } catch (err) {
        if (err instanceof SandboxTimeoutError) throw new BashTimeoutError(err.message)
        throw new BashSessionError((err as Error).message)
      }
    },
  })

  // Safe narrowing: runtime validation uses the custom schema, whose output is
  // assignable to SandboxBashToolInput by the TSchema bound above.
  return built as InvokableTool<z.output<TSchema>, BashOutput>
}
