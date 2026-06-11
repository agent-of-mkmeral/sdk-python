import { describe, expect, it } from 'vitest'
import { mkdtempSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { fileEditor, makeFileEditor } from '../index.js'
import { DEFAULT_FILE_EDITOR_DESCRIPTION } from '../file-editor.js'
import { TestSandbox } from '../../../__fixtures__/test-sandbox.node.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import type { ToolContext } from '../../../tools/tool.js'

function createContext(): ToolContext {
  const agent = createMockAgent()
  return {
    toolUse: { name: 'fileEditor', toolUseId: 'test-id', input: {} },
    agent,
    invocationState: {},
    interrupt: () => {
      throw new Error('interrupt not available in mock context')
    },
  }
}

describe.skipIf(process.platform === 'win32')('fileEditor.clone', () => {
  it('clone binds a specific sandbox for file I/O', async () => {
    const workDir = mkdtempSync(join(tmpdir(), 'fe-clone-test-'))
    const filePath = join(workDir, 'hello.txt')
    writeFileSync(filePath, 'hello from bound sandbox\n')

    const bound = fileEditor.clone({
      sandbox: new TestSandbox(workDir),
      description: `${DEFAULT_FILE_EDITOR_DESCRIPTION} Files are in the test sandbox.`,
    })

    expect(bound.description).toContain('test sandbox')

    const result = await bound.invoke({ command: 'view', path: filePath }, createContext())
    expect(String(result)).toContain('hello from bound sandbox')
  })

  it('clone does not modify the fileEditor singleton', () => {
    fileEditor.clone({ name: 'editor2', description: 'changed' })

    expect(fileEditor.name).toBe('fileEditor')
    expect(fileEditor.description).toBe(DEFAULT_FILE_EDITOR_DESCRIPTION)
  })

  it('deprecated makeFileEditor alias still works', async () => {
    const workDir = mkdtempSync(join(tmpdir(), 'fe-legacy-test-'))
    const filePath = join(workDir, 'legacy.txt')
    writeFileSync(filePath, 'legacy content\n')

    const tool = makeFileEditor({ sandbox: new TestSandbox(workDir), description: 'legacy factory' })

    expect(tool.description).toBe('legacy factory')
    const result = await tool.invoke({ command: 'view', path: filePath }, createContext())
    expect(String(result)).toContain('legacy content')
  })
})
