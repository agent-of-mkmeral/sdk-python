import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { z } from 'zod'
import { fileEditor, makeFileEditor, DEFAULT_FILE_EDITOR_DESCRIPTION } from '../file-editor.js'
import type { ToolContext } from '../../../index.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import { TestSandbox } from '../../../__fixtures__/test-sandbox.node.js'
import { promises as fs } from 'fs'
import * as path from 'path'
import { tmpdir } from 'os'

describe('makeFileEditor factory shape', () => {
  let testDir: string

  const makeContext = (extra?: Record<string, unknown>): ToolContext => {
    const agent = createMockAgent(extra ? { extra } : undefined)
    return {
      toolUse: { name: 'fileEditor', toolUseId: 'test-id', input: {} },
      agent,
      invocationState: {},
      interrupt: () => {
        throw new Error('interrupt not available in mock context')
      },
    }
  }

  beforeEach(async () => {
    testDir = path.join(tmpdir(), `make-file-editor-test-${Date.now()}-${Math.random().toString(36).slice(2)}`)
    await fs.mkdir(testDir, { recursive: true })
  })

  afterEach(async () => {
    try {
      await fs.rm(testDir, { recursive: true, force: true })
    } catch {
      // Ignore cleanup errors
    }
  })

  describe('defaults', () => {
    it('the exported fileEditor singleton is a default makeFileEditor() instance', () => {
      const custom = makeFileEditor()
      expect(custom.name).toBe(fileEditor.name)
      expect(custom.description).toBe(fileEditor.description)
      expect(custom.toolSpec).toEqual(fileEditor.toolSpec)
    })

    it('uses the default name and description', () => {
      const editor = makeFileEditor()
      expect(editor.name).toBe('fileEditor')
      expect(editor.description).toBe(DEFAULT_FILE_EDITOR_DESCRIPTION)
    })
  })

  describe('tool-spec overrides', () => {
    it('applies name and description overrides', () => {
      const editor = makeFileEditor({ name: 'containerEditor', description: 'Edits files in the container.' })
      expect(editor.name).toBe('containerEditor')
      expect(editor.description).toBe('Edits files in the container.')
      expect(editor.toolSpec.name).toBe('containerEditor')
    })
  })

  describe('sandbox binding', () => {
    it('prefers the bound sandbox over context.agent.sandbox', async () => {
      const boundDir = path.join(testDir, 'bound')
      const contextDir = path.join(testDir, 'context')
      await fs.mkdir(boundDir, { recursive: true })
      await fs.mkdir(contextDir, { recursive: true })
      await fs.writeFile(path.join(boundDir, 'marker.txt'), 'bound sandbox content', 'utf-8')

      const editor = makeFileEditor({ sandbox: new TestSandbox(boundDir) })
      const context = makeContext({ sandbox: new TestSandbox(contextDir) })

      const result = await editor.invoke({ command: 'view', path: path.join(boundDir, 'marker.txt') }, context)
      expect(result).toContain('bound sandbox content')
    })

    it('falls back to context.agent.sandbox when no sandbox is bound', async () => {
      const filePath = path.join(testDir, 'fallback.txt')
      await fs.writeFile(filePath, 'fallback content', 'utf-8')

      const editor = makeFileEditor()
      const context = makeContext({ sandbox: new TestSandbox(testDir) })

      const result = await editor.invoke({ command: 'view', path: filePath }, context)
      expect(result).toContain('fallback content')
    })
  })

  describe('inputSchema override semantics', () => {
    const restrictedSchema = z.object({
      command: z.literal('view').describe('Read-only: only view is allowed.'),
      path: z.string(),
      view_range: z.tuple([z.number(), z.number()]).optional(),
      file_text: z.string().optional(),
      old_str: z.string().optional(),
      new_str: z.string().optional(),
      insert_line: z.number().optional(),
    })

    it('advertises the replacement schema to the model', () => {
      const readOnly = makeFileEditor({ inputSchema: restrictedSchema })
      const schema = readOnly.toolSpec.inputSchema as {
        properties: Record<string, { const?: string; enum?: string[] }>
      }
      // Default schema advertises an enum of four commands; replacement pins it to 'view'.
      expect(schema.properties.command?.enum ?? [schema.properties.command?.const]).toEqual(['view'])
    })

    it('validates with the replacement schema: rejects commands outside the restricted schema', async () => {
      const readOnly = makeFileEditor({ sandbox: new TestSandbox(testDir), inputSchema: restrictedSchema })
      const context = makeContext()
      // 'create' is valid for the DEFAULT schema but not for the replacement schema.
      await expect(
        readOnly.invoke({ command: 'create', path: path.join(testDir, 'x.txt'), file_text: 'nope' } as never, context)
      ).rejects.toThrow()
      // And the file must not have been created.
      await expect(fs.access(path.join(testDir, 'x.txt'))).rejects.toThrow()
    })

    it('validates with the replacement schema: accepts and executes conforming input', async () => {
      const filePath = path.join(testDir, 'readable.txt')
      await fs.writeFile(filePath, 'read-only ok', 'utf-8')

      const readOnly = makeFileEditor({ sandbox: new TestSandbox(testDir), inputSchema: restrictedSchema })
      const result = await readOnly.invoke({ command: 'view', path: filePath }, makeContext())
      expect(result).toContain('read-only ok')
    })
  })
})
