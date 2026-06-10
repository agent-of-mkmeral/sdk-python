/**
 * Sandbox-bound bash tool factory.
 *
 * Separated from bash.ts to avoid pulling Node dependencies (child_process, Buffer)
 * into sandbox implementations that import this.
 *
 * This factory does exactly one job: bind a bash tool to a sandbox. To
 * customize surface metadata (name, description), compose with
 * {@link customizeTool} from `tools/customize-tool.js`:
 * ```typescript
 * customizeTool(makeBash(sandbox), { description: 'Runs in my container.' })
 * ```
 */

import { tool } from '../../tools/tool-factory.js'
import { z } from 'zod'
import { SandboxTimeoutError } from '../../sandbox/errors.js'
import { Sandbox } from '../../sandbox/base.js'
import type { BashOutput } from './types.js'
import { BashTimeoutError, BashSessionError, SANDBOX_BASH_DESCRIPTION } from './types.js'

const sandboxBashInputSchema = z.object({
  command: z.string().describe('The bash command to execute.'),
  timeout: z.number().positive().optional().describe('Timeout in seconds (default: 120).'),
})

/**
 * Options form for {@link makeBash}. Binding-only: metadata customization
 * (name, description) is handled by `customizeTool` instead.
 */
export interface MakeBashOptions {
  /** Sandbox to bind at creation time. If omitted, resolved from `context.agent.sandbox` at call time. */
  sandbox?: Sandbox
}

/**
 * Create a sandbox bash tool bound to the given sandbox.
 *
 * If a sandbox is provided, it's bound at creation time. Otherwise, the tool
 * reads from `context.agent.sandbox` at call time. Used by sandbox
 * implementations in `getTools()`.
 *
 * To customize the tool's name or description, wrap the result with
 * `customizeTool` — this factory intentionally does binding only.
 *
 * @param sandboxOrOptions - A sandbox instance, or an options object.
 * @returns A bash tool routing execution through the bound (or agent's) sandbox.
 */
export function makeBash(sandboxOrOptions: Sandbox | MakeBashOptions = {}): ReturnType<typeof tool> {
  const boundSandbox = sandboxOrOptions instanceof Sandbox ? sandboxOrOptions : sandboxOrOptions.sandbox
  return tool({
    name: 'bash',
    description: SANDBOX_BASH_DESCRIPTION,
    inputSchema: sandboxBashInputSchema,
    callback: async (input, context) => {
      if (!context) {
        throw new Error('Tool context is required for bash operations')
      }

      const sandbox = boundSandbox ?? context.agent.sandbox
      try {
        const result = await sandbox.execute(input.command, { timeout: input.timeout ?? 120 })
        return { output: result.stdout, error: result.stderr } as BashOutput
      } catch (err) {
        if (err instanceof SandboxTimeoutError) throw new BashTimeoutError(err.message)
        throw new BashSessionError((err as Error).message)
      }
    },
  })
}
