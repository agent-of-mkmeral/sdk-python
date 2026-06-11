import { describe, expect, it } from 'vitest'
import { mkdtempSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { z } from 'zod'
import { sandboxBash, makeBash } from '../index.js'
import { SANDBOX_BASH_DESCRIPTION } from '../types.js'
import type { BashOutput } from '../types.js'
import { TestSandbox } from '../../../__fixtures__/test-sandbox.node.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import type { ToolContext } from '../../../tools/tool.js'

function createContext(): ToolContext {
  const agent = createMockAgent()
  return {
    toolUse: { name: 'bash', toolUseId: 'test-id', input: {} },
    agent,
    invocationState: {},
    interrupt: () => {
      throw new Error('interrupt not available in mock context')
    },
  }
}

function createSandbox(): TestSandbox {
  return new TestSandbox(mkdtempSync(join(tmpdir(), 'bash-clone-test-')))
}

describe.skipIf(process.platform === 'win32')('sandboxBash.clone', () => {
  it('default sandboxBash reads the sandbox from agent context', async () => {
    const context = createContext()
    const result = await sandboxBash.invoke({ command: 'echo "from context"' }, context)

    expect((result as BashOutput).output).toContain('from context')
  })

  it('clone binds a specific sandbox that wins over the context sandbox', async () => {
    const sandbox = createSandbox()
    const bound = sandboxBash.clone({ sandbox })
    const context = createContext()

    const result = await bound.invoke({ command: 'pwd' }, context)

    expect((result as BashOutput).output.trim()).toContain('bash-clone-test-')
  })

  it('clone customizes description alongside the binding (the getTools use case)', async () => {
    const sandbox = createSandbox()
    const bound = sandboxBash.clone({
      sandbox,
      description: `${SANDBOX_BASH_DESCRIPTION} Runs in test sandbox.`,
    })

    expect(bound.name).toBe('bash')
    expect(bound.description).toBe(`${SANDBOX_BASH_DESCRIPTION} Runs in test sandbox.`)
    expect(bound.toolSpec.description).toBe(`${SANDBOX_BASH_DESCRIPTION} Runs in test sandbox.`)

    const result = await bound.invoke({ command: 'echo bound' }, createContext())
    expect((result as BashOutput).output).toContain('bound')
  })

  it('metadata-only clone keeps routing through the agent context (agent-dev use case)', async () => {
    const custom = sandboxBash.clone({ description: 'Prefer pipes | over temp files.' })

    expect(custom.description).toBe('Prefer pipes | over temp files.')

    const result = await custom.invoke({ command: 'echo "still context-routed"' }, createContext())
    expect((result as BashOutput).output).toContain('still context-routed')
  })

  it('clone does not modify the sandboxBash singleton', () => {
    sandboxBash.clone({ name: 'otherBash', description: 'changed' })

    expect(sandboxBash.name).toBe('bash')
    expect(sandboxBash.description).toBe(SANDBOX_BASH_DESCRIPTION)
  })

  it('schema override drives both spec and validation', async () => {
    const sandbox = createSandbox()
    const audited = sandboxBash.clone({
      sandbox,
      inputSchema: z.object({
        command: z.string(),
        timeout: z.number().positive().optional(),
        reason: z.string().describe('Why this command is needed.'),
      }),
    })

    const schema = audited.toolSpec.inputSchema as { properties: Record<string, unknown>; required: string[] }
    expect(schema.required).toContain('reason')

    // Valid under the original schema, invalid under the replacement → rejected.
    await expect(audited.invoke({ command: 'echo hi' } as never, createContext())).rejects.toThrow()

    const result = await audited.invoke({ command: 'echo hi', reason: 'testing' }, createContext())
    expect((result as BashOutput).output).toContain('hi')
  })

  it('deprecated makeBash alias still works and routes through clone', async () => {
    const sandbox = createSandbox()
    const tool = makeBash({ sandbox, description: 'legacy factory' })

    expect(tool.description).toBe('legacy factory')
    const result = await tool.invoke({ command: 'echo legacy' }, createContext())
    expect((result as BashOutput).output).toContain('legacy')
  })
})
