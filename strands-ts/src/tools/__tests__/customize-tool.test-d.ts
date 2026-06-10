import { describe, it, expectTypeOf } from 'vitest'
import { z } from 'zod'
import { tool } from '../tool-factory.js'
import { customizeTool, type ToolOverrides } from '../customize-tool.js'
import type { Tool } from '../tool.js'
import type { InvokableTool } from '../tool.js'

describe('customizeTool type tests', () => {
  const base = tool({
    name: 'baseTool',
    description: 'Base description',
    inputSchema: z.object({ value: z.string() }),
    callback: (input) => `echo:${input.value}`,
  })

  it('accepts valid override keys', () => {
    customizeTool(base, { name: 'renamed' })
    customizeTool(base, { description: 'new description' })
    customizeTool(base, { name: 'renamed', description: 'new description' })
    customizeTool(base, {})
  })

  it('rejects typo’d override keys on object literals (excess property check)', () => {
    // @ts-expect-error - 'nmae' is not a valid override key
    customizeTool(base, { nmae: 'renamed' })
    // @ts-expect-error - 'descripton' is not a valid override key
    customizeTool(base, { descripton: 'new description' })
    // @ts-expect-error - 'inputSchema' is deliberately NOT overridable (see ToolOverrides docs)
    customizeTool(base, { inputSchema: z.object({}) })
  })

  it('rejects wrong value types', () => {
    // @ts-expect-error - name must be a string
    customizeTool(base, { name: 42 })
    // @ts-expect-error - description must be a string
    customizeTool(base, { description: null })
  })

  it('preserves InvokableTool typing for invokable inputs', () => {
    const wrapped = customizeTool(base, { name: 'renamed' })
    // The wrapper preserves the exact input/output types of the wrapped tool,
    // including the inferred template-literal return type of the callback.
    expectTypeOf(wrapped.invoke).toEqualTypeOf(base.invoke)
    expectTypeOf(wrapped.invoke).parameter(0).toEqualTypeOf<{ value: string }>()
    expectTypeOf(wrapped.invoke).returns.resolves.toExtend<string>()
    expectTypeOf(wrapped).toExtend<InvokableTool<{ value: string }, string>>()
  })

  it('returns plain Tool for non-invokable inputs', () => {
    const plain = base as Tool
    const wrapped = customizeTool(plain, { name: 'renamed' })
    expectTypeOf(wrapped).toEqualTypeOf<Tool>()
  })

  it('ToolOverrides has exactly name and description', () => {
    expectTypeOf<keyof ToolOverrides>().toEqualTypeOf<'name' | 'description'>()
  })
})
