import { describe, it, expect } from 'vitest'
import { z } from 'zod'
import { bash, makeHostBash } from '../bash.js'
import type { BashOutput } from '../types.js'
import type { ToolContext } from '../../../index.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'

const makeContext = (): ToolContext => {
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

describe.skipIf(process.platform === 'win32')('makeHostBash factory', () => {
  describe('defaults', () => {
    it('the exported bash singleton is a default makeHostBash() instance', () => {
      const custom = makeHostBash()
      expect(custom.name).toBe(bash.name)
      expect(custom.description).toBe(bash.description)
      expect(custom.toolSpec).toEqual(bash.toolSpec)
    })

    it('executes commands with a persistent session', async () => {
      const hostBash = makeHostBash()
      const context = makeContext()
      await hostBash.invoke({ mode: 'execute', command: 'TEST_VAR=persisted' }, context)
      const result = await hostBash.invoke({ mode: 'execute', command: 'echo "$TEST_VAR"' }, context)
      expect((result as BashOutput).output).toContain('persisted')
    })
  })

  describe('tool-spec overrides', () => {
    it('applies name and description overrides', () => {
      const buildShell = makeHostBash({ name: 'buildShell', description: 'Persistent build shell.' })
      expect(buildShell.name).toBe('buildShell')
      expect(buildShell.description).toBe('Persistent build shell.')
      expect(buildShell.toolSpec.name).toBe('buildShell')
    })

    it('keeps sessions independent between factory instances on the same agent', async () => {
      const shellA = makeHostBash({ name: 'shellA' })
      const shellB = makeHostBash({ name: 'shellB' })
      const context = makeContext()

      await shellA.invoke({ mode: 'execute', command: 'ONLY_IN_A=yes' }, context)
      const inA = await shellA.invoke({ mode: 'execute', command: 'echo "${ONLY_IN_A:-unset}"' }, context)
      const inB = await shellB.invoke({ mode: 'execute', command: 'echo "${ONLY_IN_A:-unset}"' }, context)

      expect((inA as BashOutput).output.trim()).toBe('yes')
      expect((inB as BashOutput).output.trim()).toBe('unset')
    })

    it('supports restart per factory instance', async () => {
      const hostBash = makeHostBash({ name: 'restartable' })
      const context = makeContext()
      await hostBash.invoke({ mode: 'execute', command: 'WILL_BE_LOST=1' }, context)
      const restart = await hostBash.invoke({ mode: 'restart' }, context)
      expect(restart).toBe('Bash session restarted')
      const after = await hostBash.invoke({ mode: 'execute', command: 'echo "${WILL_BE_LOST:-gone}"' }, context)
      expect((after as BashOutput).output.trim()).toBe('gone')
    })
  })

  describe('inputSchema override semantics', () => {
    const auditedSchema = z.object({
      mode: z.enum(['execute', 'restart']),
      command: z.string().optional(),
      timeout: z.number().positive().optional(),
      reason: z.string().describe('Why this command is needed.'),
    })

    it('advertises the replacement schema to the model', () => {
      const audited = makeHostBash({ inputSchema: auditedSchema })
      const schema = audited.toolSpec.inputSchema as { properties: object; required: string[] }
      expect(Object.keys(schema.properties)).toEqual(['mode', 'command', 'timeout', 'reason'])
      expect(schema.required).toEqual(['mode', 'reason'])
    })

    it('validates with the replacement schema', async () => {
      const audited = makeHostBash({ inputSchema: auditedSchema })
      const context = makeContext()
      // Valid for the DEFAULT schema, invalid for the replacement schema (reason missing).
      await expect(audited.invoke({ mode: 'execute', command: 'echo hi' } as never, context)).rejects.toThrow()
      // Conforming input executes.
      const result = await audited.invoke(
        { mode: 'execute', command: 'echo "host audited"', reason: 'unit test' },
        context
      )
      expect((result as BashOutput).output).toContain('host audited')
    })
  })
})
