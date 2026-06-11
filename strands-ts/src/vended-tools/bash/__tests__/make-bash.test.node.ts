import { describe, it, expect } from 'vitest'
import { z } from 'zod'
import { makeBash } from '../make-bash.js'
import { SANDBOX_BASH_DESCRIPTION, type BashOutput } from '../types.js'
import type { ToolContext } from '../../../index.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import { TestSandbox } from '../../../__fixtures__/test-sandbox.node.js'
import { mkdtempSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'

const makeContext = (extra?: Record<string, unknown>): ToolContext => {
  const agent = createMockAgent(extra ? { extra } : undefined)
  return {
    toolUse: { name: 'bash', toolUseId: 'test-id', input: {} },
    agent,
    invocationState: {},
    interrupt: () => {
      throw new Error('interrupt not available in mock context')
    },
  }
}

const makeTestSandbox = (): TestSandbox => new TestSandbox(mkdtempSync(join(tmpdir(), 'make-bash-test-')))

describe.skipIf(process.platform === 'win32')('makeBash factory shape', () => {
  describe('defaults', () => {
    it('uses the default name and description', () => {
      const sandboxBash = makeBash()
      expect(sandboxBash.name).toBe('bash')
      expect(sandboxBash.description).toBe(SANDBOX_BASH_DESCRIPTION)
    })

    it('advertises the default input schema (command required, timeout optional)', () => {
      const sandboxBash = makeBash()
      const schema = sandboxBash.toolSpec.inputSchema as { properties: object; required: string[] }
      expect(Object.keys(schema.properties)).toEqual(['command', 'timeout'])
      expect(schema.required).toEqual(['command'])
    })
  })

  describe('tool-spec overrides', () => {
    it('applies name and description overrides', () => {
      const sandboxBash = makeBash({ name: 'containerBash', description: 'Runs in the build container.' })
      expect(sandboxBash.name).toBe('containerBash')
      expect(sandboxBash.description).toBe('Runs in the build container.')
      expect(sandboxBash.toolSpec.name).toBe('containerBash')
    })

    it('combines binding and overrides in one flat options bag', async () => {
      const sandbox = makeTestSandbox()
      const sandboxBash = makeBash({ sandbox, name: 'myBash', description: 'custom' })
      const result = await sandboxBash.invoke({ command: 'echo "bound"' }, makeContext())
      expect((result as BashOutput).output).toContain('bound')
      expect(sandboxBash.name).toBe('myBash')
    })
  })

  describe('sandbox binding', () => {
    it('prefers the bound sandbox over context.agent.sandbox', async () => {
      const boundDir = mkdtempSync(join(tmpdir(), 'make-bash-bound-'))
      const contextDir = mkdtempSync(join(tmpdir(), 'make-bash-context-'))
      const sandboxBash = makeBash({ sandbox: new TestSandbox(boundDir) })
      const context = makeContext({ sandbox: new TestSandbox(contextDir) })

      const result = await sandboxBash.invoke({ command: 'pwd' }, context)
      // macOS tmpdir may resolve through /private — compare suffixes.
      expect((result as BashOutput).output.trim().endsWith(boundDir.split('/').pop()!)).toBe(true)
    })

    it('falls back to context.agent.sandbox when no sandbox is bound', async () => {
      const contextDir = mkdtempSync(join(tmpdir(), 'make-bash-context-'))
      const sandboxBash = makeBash()
      const context = makeContext({ sandbox: new TestSandbox(contextDir) })

      const result = await sandboxBash.invoke({ command: 'pwd' }, context)
      expect((result as BashOutput).output.trim().endsWith(contextDir.split('/').pop()!)).toBe(true)
    })
  })

  describe('inputSchema override semantics', () => {
    const auditedSchema = z.object({
      command: z.string(),
      timeout: z.number().positive().optional(),
      reason: z.string().describe('Why this command is needed.'),
    })

    it('advertises the replacement schema to the model', () => {
      const audited = makeBash({ sandbox: makeTestSandbox(), inputSchema: auditedSchema })
      const schema = audited.toolSpec.inputSchema as { properties: object; required: string[] }
      expect(Object.keys(schema.properties)).toEqual(['command', 'timeout', 'reason'])
      expect(schema.required).toEqual(['command', 'reason'])
    })

    it('validates with the replacement schema: rejects input missing custom required fields', async () => {
      const audited = makeBash({ sandbox: makeTestSandbox(), inputSchema: auditedSchema })
      // Valid for the DEFAULT schema, invalid for the replacement schema.
      await expect(audited.invoke({ command: 'echo hi' } as never, makeContext())).rejects.toThrow()
    })

    it('validates with the replacement schema: accepts and executes conforming input', async () => {
      const audited = makeBash({ sandbox: makeTestSandbox(), inputSchema: auditedSchema })
      const result = await audited.invoke({ command: 'echo "audited"', reason: 'unit test' }, makeContext())
      expect((result as BashOutput).output).toContain('audited')
      expect((result as BashOutput).error).toBe('')
    })

    it('enforces custom refinements from the replacement schema', async () => {
      const restricted = makeBash({
        sandbox: makeTestSandbox(),
        inputSchema: z.object({
          command: z.string().refine((c) => !c.includes('rm '), { message: 'rm is not allowed' }),
          timeout: z.number().positive().optional(),
        }),
      })
      await expect(restricted.invoke({ command: 'rm -rf /tmp/x' }, makeContext())).rejects.toThrow(/rm is not allowed/)
      const ok = await restricted.invoke({ command: 'echo safe' }, makeContext())
      expect((ok as BashOutput).output).toContain('safe')
    })
  })
})
