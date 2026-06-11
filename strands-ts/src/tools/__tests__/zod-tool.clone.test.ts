import { describe, expect, it } from 'vitest'
import { z } from 'zod'
import { tool } from '../tool-factory.js'
import { ZodTool } from '../zod-tool.js'
import { Tool } from '../tool.js'
import { createMockContext } from '../../__fixtures__/tool-helpers.js'

function makeBaseTool() {
  return tool({
    name: 'echo',
    description: 'Echoes the value back.',
    inputSchema: z.object({ value: z.string() }),
    callback: (input) => input.value,
  })
}

describe('ZodTool.clone', () => {
  describe('metadata overrides', () => {
    it('overrides name only, keeping other metadata', () => {
      const base = makeBaseTool()
      const derived = base.clone({ name: 'shout' })

      expect(derived.name).toBe('shout')
      expect(derived.description).toBe('Echoes the value back.')
      expect(derived.toolSpec.name).toBe('shout')
      expect(derived.toolSpec.description).toBe('Echoes the value back.')
      expect(derived.toolSpec.inputSchema).toEqual(base.toolSpec.inputSchema)
    })

    it('overrides description only', () => {
      const base = makeBaseTool()
      const derived = base.clone({ description: 'Custom description.' })

      expect(derived.name).toBe('echo')
      expect(derived.description).toBe('Custom description.')
      expect(derived.toolSpec.description).toBe('Custom description.')
    })

    it('does not modify the source tool', () => {
      const base = makeBaseTool()
      base.clone({ name: 'other', description: 'changed' })

      expect(base.name).toBe('echo')
      expect(base.description).toBe('Echoes the value back.')
      expect(base.toolSpec.name).toBe('echo')
    })

    it('returns a new, independent ZodTool instance', () => {
      const base = makeBaseTool()
      const derived = base.clone({ name: 'copy' })

      expect(derived).not.toBe(base)
      expect(derived).toBeInstanceOf(ZodTool)
      expect(derived).toBeInstanceOf(Tool)
    })

    it('clone with no overrides reproduces the tool', async () => {
      const base = makeBaseTool()
      const derived = base.clone({})

      expect(derived.toolSpec).toEqual(base.toolSpec)
      expect(await derived.invoke({ value: 'hi' })).toBe('hi')
    })

    it('shares the callback — derived tool executes identically', async () => {
      const base = makeBaseTool()
      const derived = base.clone({ name: 'shout', description: 'Louder echo.' })

      expect(await derived.invoke({ value: 'hello' })).toBe('hello')
    })

    it('supports chained clones', () => {
      const base = makeBaseTool()
      const derived = base.clone({ name: 'a' }).clone({ description: 'b' }).clone({ name: 'c' })

      expect(derived.name).toBe('c')
      expect(derived.description).toBe('b')
    })

    it('streams through the agent-facing path with derived metadata', async () => {
      const base = makeBaseTool()
      const derived = base.clone({ name: 'shout' })
      const context = createMockContext({ name: 'shout', toolUseId: 't-1', input: { value: 'streamed' } })

      const generator = derived.stream(context)
      let result = await generator.next()
      while (!result.done) {
        result = await generator.next()
      }

      expect(result.value.status).toBe('success')
    })
  })

  describe('inputSchema overrides', () => {
    it('replacement schema drives the model-facing spec', () => {
      const base = makeBaseTool()
      const derived = base.clone({
        inputSchema: z.object({
          value: z.string(),
          reason: z.string().describe('Why this call is needed.'),
        }),
      })

      const schema = derived.toolSpec.inputSchema as { properties: Record<string, unknown>; required: string[] }
      expect(Object.keys(schema.properties)).toEqual(['value', 'reason'])
      expect(schema.required).toContain('reason')
    })

    it('replacement schema drives runtime validation (not just the spec)', async () => {
      const base = makeBaseTool()
      const derived = base.clone({
        inputSchema: z.object({ value: z.string(), reason: z.string() }),
      })

      // Valid under the ORIGINAL schema but invalid under the replacement → must reject.
      await expect(derived.invoke({ value: 'hi' } as never)).rejects.toThrow()
      // Valid under the replacement → executes via the shared callback.
      expect(await derived.invoke({ value: 'hi', reason: 'testing' })).toBe('hi')
    })

    it('replacement schema refinement rules fire at invoke time', async () => {
      const base = makeBaseTool()
      const derived = base.clone({
        inputSchema: z.object({ value: z.string().min(3) }),
      })

      await expect(derived.invoke({ value: 'ab' })).rejects.toThrow()
      expect(await derived.invoke({ value: 'abc' })).toBe('abc')
    })

    it('source tool validation is unaffected by a derived schema', async () => {
      const base = makeBaseTool()
      base.clone({ inputSchema: z.object({ value: z.string().min(100) }) })

      expect(await base.invoke({ value: 'short' })).toBe('short')
    })
  })

  describe('config overrides', () => {
    function makeConfigurableTool() {
      return tool({
        name: 'greeter',
        description: 'Greets.',
        inputSchema: z.object({ who: z.string() }),
        config: { greeting: 'Hello', punctuation: '!' } as { greeting: string; punctuation: string },
        callback: (input, _context, config) => `${config?.greeting}, ${input.who}${config?.punctuation}`,
      })
    }

    it('passes declared config to the callback via invoke', async () => {
      const base = makeConfigurableTool()
      expect(await base.invoke({ who: 'world' })).toBe('Hello, world!')
    })

    it('passes declared config to the callback via stream', async () => {
      const base = makeConfigurableTool()
      const context = createMockContext({ name: 'greeter', toolUseId: 't-1', input: { who: 'world' } })

      const generator = base.stream(context)
      let result = await generator.next()
      while (!result.done) {
        result = await generator.next()
      }

      expect(result.value.status).toBe('success')
      expect(JSON.stringify(result.value.content)).toContain('Hello, world!')
    })

    it('clone overrides a subset of config keys (shallow merge)', async () => {
      const base = makeConfigurableTool()
      const derived = base.clone({ greeting: 'Howdy' })

      expect(await derived.invoke({ who: 'world' })).toBe('Howdy, world!')
      // Untouched key preserved
      const derived2 = derived.clone({ punctuation: '?' })
      expect(await derived2.invoke({ who: 'world' })).toBe('Howdy, world?')
    })

    it('config override does not affect the source tool', async () => {
      const base = makeConfigurableTool()
      base.clone({ greeting: 'Howdy' })

      expect(await base.invoke({ who: 'world' })).toBe('Hello, world!')
    })

    it('combines metadata and config overrides in one call', async () => {
      const base = makeConfigurableTool()
      const derived = base.clone({ name: 'cowboyGreeter', description: 'Texan greeting.', greeting: 'Howdy' })

      expect(derived.name).toBe('cowboyGreeter')
      expect(derived.description).toBe('Texan greeting.')
      expect(await derived.invoke({ who: 'partner' })).toBe('Howdy, partner!')
    })

    it('combines config and inputSchema overrides in one call', async () => {
      const base = makeConfigurableTool()
      const derived = base.clone({
        greeting: 'Hi',
        inputSchema: z.object({ who: z.string().min(2) }),
      })

      await expect(derived.invoke({ who: 'x' })).rejects.toThrow()
      expect(await derived.invoke({ who: 'yall' })).toBe('Hi, yall!')
    })
  })

  describe('tools without declared config', () => {
    it('clone works on tools that never declared config', async () => {
      const base = makeBaseTool()
      const derived = base.clone({ name: 'renamed' })

      expect(await derived.invoke({ value: 'ok' })).toBe('ok')
    })
  })
})
