import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { tool } from '../tool-factory.js'
import { Tool, ToolStreamEvent } from '../tool.js'
import { TextBlock, ToolResultBlock } from '../../types/messages.js'
import type { InvokableTool, ToolContext, ToolStreamGenerator } from '../tool.js'
import type { ToolSpec } from '../types.js'
import { customizeTool } from '../customize-tool.js'
import { ToolValidationError } from '../../errors.js'
import { createMockContext } from '../../__fixtures__/tool-helpers.js'
import { collectGenerator } from '../../__fixtures__/model-test-helpers.js'
import type { JSONValue } from '../../types/json.js'

function createContext(name: string, input: JSONValue): ToolContext {
  return createMockContext({ name, toolUseId: 'test-123', input })
}

function makeBaseTool(): InvokableTool<{ value: string }, string> {
  return tool({
    name: 'baseTool',
    description: 'Base description',
    inputSchema: z.object({ value: z.string() }),
    callback: (input) => `echo:${input.value}`,
  })
}

describe('customizeTool', () => {
  describe('surface metadata overrides', () => {
    it('overrides name consistently across name and toolSpec.name', () => {
      const wrapped = customizeTool(makeBaseTool(), { name: 'renamedTool' })

      expect(wrapped.name).toBe('renamedTool')
      expect(wrapped.toolSpec.name).toBe('renamedTool')
      // Untouched fields pass through
      expect(wrapped.description).toBe('Base description')
      expect(wrapped.toolSpec.description).toBe('Base description')
    })

    it('overrides description consistently across description and toolSpec.description', () => {
      const wrapped = customizeTool(makeBaseTool(), { description: 'New description' })

      expect(wrapped.description).toBe('New description')
      expect(wrapped.toolSpec.description).toBe('New description')
      expect(wrapped.name).toBe('baseTool')
      expect(wrapped.toolSpec.name).toBe('baseTool')
    })

    it('overrides name and description together', () => {
      const wrapped = customizeTool(makeBaseTool(), { name: 'newName', description: 'New description' })

      expect(wrapped.name).toBe('newName')
      expect(wrapped.toolSpec.name).toBe('newName')
      expect(wrapped.description).toBe('New description')
      expect(wrapped.toolSpec.description).toBe('New description')
    })

    it('preserves the input schema in toolSpec', () => {
      const base = makeBaseTool()
      const wrapped = customizeTool(base, { name: 'renamedTool' })

      expect(wrapped.toolSpec.inputSchema).toEqual(base.toolSpec.inputSchema)
    })

    it('leaves the original tool completely untouched', () => {
      const base = makeBaseTool()
      const originalSpec = JSON.parse(JSON.stringify(base.toolSpec)) as ToolSpec

      customizeTool(base, { name: 'renamedTool', description: 'New description' })

      expect(base.name).toBe('baseTool')
      expect(base.description).toBe('Base description')
      expect(base.toolSpec).toEqual(originalSpec)
    })

    it('passes everything through when overrides are empty', () => {
      const base = makeBaseTool()
      const wrapped = customizeTool(base, {})

      expect(wrapped.name).toBe(base.name)
      expect(wrapped.description).toBe(base.description)
      expect(wrapped.toolSpec).toEqual(base.toolSpec)
      expect(wrapped).not.toBe(base)
    })
  })

  describe('validation', () => {
    it('rejects invalid tool names', () => {
      expect(() => customizeTool(makeBaseTool(), { name: 'has spaces!' })).toThrow(ToolValidationError)
      expect(() => customizeTool(makeBaseTool(), { name: '' })).toThrow(ToolValidationError)
      expect(() => customizeTool(makeBaseTool(), { name: 'x'.repeat(65) })).toThrow(ToolValidationError)
    })

    it('rejects empty description', () => {
      expect(() => customizeTool(makeBaseTool(), { description: '' })).toThrow(ToolValidationError)
    })

    it('accepts valid names with letters, digits, underscores, and hyphens', () => {
      expect(customizeTool(makeBaseTool(), { name: 'sandbox_bash-2' }).name).toBe('sandbox_bash-2')
    })
  })

  describe('instanceof and interface preservation', () => {
    it('returns an instanceof Tool', () => {
      const wrapped = customizeTool(makeBaseTool(), { name: 'renamedTool' })
      expect(wrapped instanceof Tool).toBe(true)
    })

    it('preserves invoke() for InvokableTool inputs', () => {
      const wrapped = customizeTool(makeBaseTool(), { name: 'renamedTool' })
      expect(typeof wrapped.invoke).toBe('function')
    })

    it('does not add invoke() to non-invokable tools', () => {
      class StreamOnlyTool extends Tool {
        name = 'streamOnly'
        description = 'Stream-only tool'
        toolSpec: ToolSpec = {
          name: 'streamOnly',
          description: 'Stream-only tool',
          inputSchema: { type: 'object' },
        }
        // eslint-disable-next-line require-yield
        async *stream(toolContext: ToolContext): ToolStreamGenerator {
          return new ToolResultBlock({
            toolUseId: toolContext.toolUse.toolUseId,
            status: 'success',
            content: [new TextBlock('ok')],
          })
        }
      }

      const wrapped = customizeTool(new StreamOnlyTool(), { name: 'renamed' })
      expect(wrapped instanceof Tool).toBe(true)
      expect((wrapped as Partial<InvokableTool<unknown, unknown>>).invoke).toBeUndefined()
    })
  })

  describe('delegation', () => {
    it('delegates stream() to the wrapped tool and returns the same result', async () => {
      const base = makeBaseTool()
      const wrapped = customizeTool(base, { name: 'renamedTool' })

      const { result: baseResult } = await collectGenerator(base.stream(createContext('baseTool', { value: 'hi' })))
      const { result: wrappedResult } = await collectGenerator(
        wrapped.stream(createContext('renamedTool', { value: 'hi' }))
      )

      expect(wrappedResult.status).toBe('success')
      expect(wrappedResult.content).toEqual(baseResult.content)
    })

    it('delegates stream events from the wrapped tool', async () => {
      const streamingTool = tool({
        name: 'streamer',
        description: 'Streams updates',
        inputSchema: z.object({}),
        callback: async function* () {
          yield 'progress-1'
          yield 'progress-2'
          return 'done'
        },
      })

      const wrapped = customizeTool(streamingTool, { name: 'renamedStreamer' })
      const { items: events, result } = await collectGenerator(wrapped.stream(createContext('renamedStreamer', {})))

      expect(events).toHaveLength(2)
      expect(events.every((e) => e instanceof ToolStreamEvent)).toBe(true)
      expect(result.status).toBe('success')
      expect(result.content[0]).toEqual(expect.objectContaining({ type: 'textBlock', text: 'done' }))
    })

    it('delegates invoke() including the inner validation', async () => {
      const wrapped = customizeTool(makeBaseTool(), { name: 'renamedTool' })

      await expect(wrapped.invoke({ value: 'hello' })).resolves.toBe('echo:hello')
      // Inner Zod validation still runs through the wrapper
      await expect(wrapped.invoke({ value: 42 as unknown as string })).rejects.toThrow()
    })

    it('inner validation failures surface as error results when streaming', async () => {
      const wrapped = customizeTool(makeBaseTool(), { name: 'renamedTool' })
      const { result } = await collectGenerator(wrapped.stream(createContext('renamedTool', { value: 42 })))

      expect(result.status).toBe('error')
    })

    it('passes the tool context through to the inner tool unchanged', async () => {
      const inner = makeBaseTool()
      const streamSpy = vi.spyOn(inner, 'stream')
      const wrapped = customizeTool(inner, { name: 'renamedTool' })

      const context = createContext('renamedTool', { value: 'hi' })
      await collectGenerator(wrapped.stream(context))

      expect(streamSpy).toHaveBeenCalledExactlyOnceWith(context)
    })
  })

  describe('double wrapping', () => {
    it('supports wrapping a wrapped tool', async () => {
      const base = makeBaseTool()
      const once = customizeTool(base, { name: 'firstRename' })
      const twice = customizeTool(once, { name: 'secondRename', description: 'Layered description' })

      expect(twice.name).toBe('secondRename')
      expect(twice.toolSpec.name).toBe('secondRename')
      expect(twice.description).toBe('Layered description')
      expect(twice.toolSpec.description).toBe('Layered description')
      expect(twice instanceof Tool).toBe(true)

      // Intermediate wrapper unchanged
      expect(once.name).toBe('firstRename')
      expect(once.description).toBe('Base description')
      // Original unchanged
      expect(base.name).toBe('baseTool')

      // Delegation chains all the way down
      const { result } = await collectGenerator(twice.stream(createContext('secondRename', { value: 'deep' })))
      expect(result.status).toBe('success')
      expect(result.content[0]).toEqual(expect.objectContaining({ type: 'textBlock', text: 'echo:deep' }))

      // invoke() chains too
      await expect(twice.invoke({ value: 'deep' })).resolves.toBe('echo:deep')
    })
  })
})
