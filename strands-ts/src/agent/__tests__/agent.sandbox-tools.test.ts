import { describe, expect, it } from 'vitest'
import { Agent } from '../agent.js'
import { Sandbox } from '../../sandbox/base.js'
import type { ExecuteOptions } from '../../sandbox/base.js'
import type { ExecutionResult, FileInfo, StreamChunk } from '../../sandbox/types.js'
import { Tool } from '../../tools/tool.js'
import type { InvokableTool, Tool as ToolType } from '../../tools/tool.js'
import { tool } from '../../tools/tool-factory.js'
import { z } from 'zod'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { collectGenerator } from '../../__fixtures__/model-test-helpers.js'

/**
 * Minimal in-memory sandbox that vends a single `echo` tool. Browser-safe:
 * no Node APIs. Used to test the agent's sandbox tool registration with
 * toolPrefix (which routes through customizeTool).
 */
class ToolVendingSandbox extends Sandbox {
  vended = tool({
    name: 'echo',
    description: 'Echoes the input value.',
    inputSchema: z.object({ value: z.string() }),
    callback: (input) => `echo:${input.value}`,
  })

  override getTools(): ToolType[] {
    return [this.vended]
  }

  // eslint-disable-next-line require-yield
  async *executeStreaming(_command: string, _options?: ExecuteOptions): AsyncGenerator<StreamChunk | ExecutionResult> {
    throw new Error('not implemented')
  }
  // eslint-disable-next-line require-yield
  async *executeCodeStreaming(
    _code: string,
    _language: string,
    _options?: ExecuteOptions
  ): AsyncGenerator<StreamChunk | ExecutionResult> {
    throw new Error('not implemented')
  }
  async readFile(_path: string): Promise<Uint8Array> {
    throw new Error('not implemented')
  }
  async writeFile(_path: string, _content: Uint8Array): Promise<void> {
    throw new Error('not implemented')
  }
  async removeFile(_path: string): Promise<void> {
    throw new Error('not implemented')
  }
  async listFiles(_path: string): Promise<FileInfo[]> {
    throw new Error('not implemented')
  }
}

describe('agent sandbox-vended tools with toolPrefix', () => {
  it('registers vended tools under the prefixed name with a consistent toolSpec', async () => {
    const sandbox = new ToolVendingSandbox() // default toolPrefix = 'sandbox'
    const agent = new Agent({ model: new MockMessageModel(), sandbox })
    await agent.initialize()

    const registered = agent.toolRegistry.get('sandbox_echo')
    expect(registered).toBeDefined()
    expect(agent.toolRegistry.get('echo')).toBeUndefined()

    // Surface metadata is consistent: name, toolSpec.name, and the spec the model sees
    expect(registered!.name).toBe('sandbox_echo')
    expect(registered!.toolSpec.name).toBe('sandbox_echo')
    expect(registered!.toolSpec.description).toBe('Echoes the input value.')
    expect(registered!.toolSpec.inputSchema).toEqual(sandbox.vended.toolSpec.inputSchema)

    // The wrapper is still a Tool (registry/agent contracts rely on the Tool interface)
    expect(registered instanceof Tool).toBe(true)

    // The original vended tool is untouched
    expect(sandbox.vended.name).toBe('echo')
    expect(sandbox.vended.toolSpec.name).toBe('echo')
  })

  it('registers vended tools unprefixed when toolPrefix is undefined', async () => {
    const sandbox = new ToolVendingSandbox()
    sandbox.toolPrefix = undefined
    const agent = new Agent({ model: new MockMessageModel(), sandbox })
    await agent.initialize()

    expect(agent.toolRegistry.get('echo')).toBeDefined()
    expect(agent.toolRegistry.get('sandbox_echo')).toBeUndefined()
  })

  it('executes the prefixed vended tool end-to-end through the agent loop', async () => {
    const sandbox = new ToolVendingSandbox()
    const model = new MockMessageModel()
      .addTurn({
        type: 'toolUseBlock',
        name: 'sandbox_echo',
        toolUseId: 'tool-1',
        input: { value: 'hello' },
      })
      .addTurn({ type: 'textBlock', text: 'Done' })

    const agent = new Agent({ model, sandbox })
    const { result } = await collectGenerator(agent.stream('Use the echo tool'))

    expect(result.stopReason).toBe('endTurn')

    // The tool result message contains the delegated execution output
    const toolResultMessage = agent.messages.find((m) => m.content.some((block) => block.type === 'toolResultBlock'))
    expect(toolResultMessage).toBeDefined()
    const toolResultBlock = toolResultMessage!.content.find((block) => block.type === 'toolResultBlock')!
    expect(JSON.stringify(toolResultBlock)).toContain('echo:hello')
  })

  it('prefixed vended tool delegates direct invoke() with inner validation intact', async () => {
    const sandbox = new ToolVendingSandbox()
    const agent = new Agent({ model: new MockMessageModel(), sandbox })
    await agent.initialize()

    const registered = agent.toolRegistry.get('sandbox_echo') as InvokableTool<{ value: string }, string>
    await expect(registered.invoke({ value: 'direct' })).resolves.toBe('echo:direct')
    // Inner Zod validation still runs against the original schema
    await expect(registered.invoke({ value: 7 as unknown as string })).rejects.toThrow()
  })

  it('skips a vended tool when the user already registered the prefixed name', async () => {
    const sandbox = new ToolVendingSandbox()
    const userTool = tool({
      name: 'sandbox_echo',
      description: 'User-supplied tool that wins over the vended one.',
      inputSchema: z.object({}),
      callback: () => 'user wins',
    })
    const agent = new Agent({ model: new MockMessageModel(), sandbox, tools: [userTool] })
    await agent.initialize()

    expect(agent.toolRegistry.get('sandbox_echo')).toBe(userTool)
  })
})
