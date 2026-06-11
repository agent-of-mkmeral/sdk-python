/**
 * Sandbox-routed bash tool.
 *
 * Separated from bash.ts to avoid pulling Node dependencies (child_process, Buffer)
 * into sandbox implementations that import this.
 */

import { tool } from '../../tools/tool-factory.js'
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
 * Configuration surface of {@link sandboxBash}, overridable via `clone()`.
 */
export interface SandboxBashConfig {
  /**
   * The sandbox commands are routed to. When unset, the tool reads
   * `context.agent.sandbox` at call time.
   */
  sandbox?: Sandbox
}

/**
 * Stateless bash tool that routes commands through a sandbox's `execute()`.
 *
 * By default it reads the sandbox from `context.agent.sandbox` at call time.
 * Use `clone()` to bind it to a specific sandbox and/or customize its
 * metadata — this is what sandbox implementations do in `getTools()`:
 *
 * @example
 * ```typescript
 * // Bind to a concrete sandbox with an environment-specific description:
 * const dockerBash = sandboxBash.clone({
 *   sandbox: dockerSandbox,
 *   description: `${SANDBOX_BASH_DESCRIPTION} Runs in Docker container "app".`,
 * })
 *
 * // Metadata-only customization (no sandbox involvement):
 * const pipedBash = sandboxBash.clone({ description: 'Prefer pipes | over temp files.' })
 * ```
 *
 * Note: this tool is stateless (each command runs independently). For a
 * persistent host session, use the Node-only `bash` tool from `bash.ts`.
 */
export const sandboxBash = tool({
  name: 'bash',
  description: SANDBOX_BASH_DESCRIPTION,
  inputSchema: sandboxBashInputSchema,
  config: {} as SandboxBashConfig,
  callback: async (input, context, config) => {
    if (!context) {
      throw new Error('Tool context is required for bash operations')
    }

    const sandbox = config?.sandbox ?? context.agent.sandbox
    try {
      const result = await sandbox.execute(input.command, { timeout: input.timeout ?? 120 })
      return { output: result.stdout, error: result.stderr } as BashOutput
    } catch (err) {
      if (err instanceof SandboxTimeoutError) throw new BashTimeoutError(err.message)
      throw new BashSessionError((err as Error).message)
    }
  },
})

/**
 * Options accepted by the deprecated {@link makeBash} factory.
 *
 * @deprecated Use `sandboxBash.clone({ sandbox, name, description, inputSchema })` instead.
 */
export interface MakeBashOptions {
  sandbox?: Sandbox
  name?: string
  description?: string
  inputSchema?: z.ZodType
}

/**
 * Create a sandbox bash tool.
 *
 * @deprecated Use `sandboxBash.clone({ sandbox, name, description, inputSchema })` instead.
 * This factory is a thin wrapper over `sandboxBash.clone()` kept for backwards
 * compatibility and will be removed in a future release.
 */
export function makeBash(options: MakeBashOptions = {}): typeof sandboxBash {
  return sandboxBash.clone({
    ...(options.sandbox !== undefined && { sandbox: options.sandbox }),
    ...(options.name !== undefined && { name: options.name }),
    ...(options.description !== undefined && { description: options.description }),
    ...(options.inputSchema !== undefined && { inputSchema: options.inputSchema as typeof sandboxBashInputSchema }),
  })
}
