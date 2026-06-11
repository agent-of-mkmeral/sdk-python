import { describe, it, expectTypeOf } from 'vitest'
import { z } from 'zod'
import { tool } from '../tool-factory.js'

const echo = tool({
  name: 'echo',
  description: 'Echoes.',
  inputSchema: z.object({ value: z.string() }),
  callback: (input) => input.value,
})

const greeter = tool({
  name: 'greeter',
  description: 'Greets.',
  inputSchema: z.object({ who: z.string() }),
  config: { greeting: 'Hello' } as { greeting: string },
  callback: (input, _context, config) => `${config?.greeting}, ${input.who}`,
})

describe('ZodTool.clone type tests', () => {
  it('accepts valid metadata overrides', () => {
    const derived = echo.clone({ name: 'renamed', description: 'new desc' })
    expectTypeOf(derived.invoke).toEqualTypeOf(echo.invoke)
  })

  it('rejects typo`d metadata keys', () => {
    // @ts-expect-error - 'nmae' is not a valid override key
    echo.clone({ nmae: 'renamed' })
    // @ts-expect-error - 'descrption' is not a valid override key
    echo.clone({ descrption: 'typo' })
  })

  it('rejects config keys on tools that declared no config', () => {
    // @ts-expect-error - echo declares no config, so 'sandbox' is not a valid key
    echo.clone({ sandbox: {} })
  })

  it('accepts declared config keys with the right types', () => {
    const derived = greeter.clone({ greeting: 'Howdy' })
    expectTypeOf(derived.invoke).toEqualTypeOf(greeter.invoke)
  })

  it('rejects wrongly-typed config values', () => {
    // @ts-expect-error - greeting must be a string
    greeter.clone({ greeting: 42 })
  })

  it('rejects unknown config keys', () => {
    // @ts-expect-error - 'greting' is not a declared config key
    greeter.clone({ greting: 'Howdy' })
  })

  it('inputSchema replacement must satisfy the callback input contract', () => {
    // Extra fields are fine — output is still assignable to { value: string }
    const widened = echo.clone({
      inputSchema: z.object({ value: z.string(), reason: z.string() }),
    })
    expectTypeOf(widened.invoke).parameter(0).toEqualTypeOf<{ value: string; reason: string }>()

    echo.clone({
      // @ts-expect-error - schema output lacks 'value', incompatible with the callback contract
      inputSchema: z.object({ val: z.string() }),
    })
  })

  it('invoke() is retyped by a replacement schema', () => {
    const derived = echo.clone({
      inputSchema: z.object({ value: z.string(), count: z.number() }),
    })
    expectTypeOf(derived.invoke).parameter(0).toEqualTypeOf<{ value: string; count: number }>()
  })

  it('clone without schema override preserves the original input type', () => {
    const derived = echo.clone({ name: 'renamed' })
    expectTypeOf(derived.invoke).parameter(0).toEqualTypeOf<{ value: string }>()
  })

  it('config declarations cannot collide with metadata keys', () => {
    tool({
      name: 'bad',
      description: 'Config collides with metadata.',
      inputSchema: z.object({ x: z.string() }),
      // @ts-expect-error - 'name' cannot be a config key
      config: { name: 'collision' } as { name: string },
      callback: (input) => input.x,
    })
  })
})
