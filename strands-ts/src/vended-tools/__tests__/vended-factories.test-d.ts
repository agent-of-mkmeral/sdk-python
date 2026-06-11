/**
 * Type-level tests for the vended-tool factory convention
 * (binding options + ToolSpecOverrides).
 *
 * Verifies:
 * 1. Typo'd keys fail to compile on object literals (excess-property checking).
 * 2. inputSchema overrides are constrained: the schema's parsed output must be
 *    assignable to the factory's input contract.
 * 3. A custom schema narrows the created tool's invoke() input type.
 * 4. The documented limitation: spreads of intermediate variables bypass
 *    excess-property checking when at least one property overlaps.
 */
import { describe, it, expectTypeOf } from 'vitest'
import { z } from 'zod'
import { makeBash, type MakeBashOptions } from '../bash/make-bash.js'
import { makeHostBash } from '../bash/bash.js'
import { makeFileEditor } from '../file-editor/file-editor.js'
import type { ToolSpecOverrides } from '../../tools/types.js'
import type { Sandbox } from '../../sandbox/base.js'

declare const sandbox: Sandbox

describe('vended-tool factory type tests', () => {
  describe('excess-property checking on object literals', () => {
    it('rejects typo’d override keys on makeBash', () => {
      // @ts-expect-error - 'descriptin' is a typo of 'description'
      makeBash({ descriptin: 'oops' })
      // @ts-expect-error - 'nam' is a typo of 'name'
      makeBash({ sandbox, nam: 'bash2' })
      // @ts-expect-error - 'sandbx' is a typo of 'sandbox'
      makeBash({ sandbx: sandbox })
    })

    it('rejects typo’d override keys on makeHostBash', () => {
      // @ts-expect-error - 'descriptin' is a typo of 'description'
      makeHostBash({ descriptin: 'oops' })
      // @ts-expect-error - host bash has no execution binding; 'sandbox' is not an option
      makeHostBash({ sandbox })
    })

    it('rejects typo’d override keys on makeFileEditor', () => {
      // @ts-expect-error - 'Name' is miscapitalized
      makeFileEditor({ Name: 'editor2' })
      // @ts-expect-error - 'inputSchma' is a typo of 'inputSchema'
      makeFileEditor({ sandbox, inputSchma: z.object({}) })
    })

    it('accepts all valid keys', () => {
      makeBash({})
      makeBash({ sandbox })
      makeBash({ sandbox, name: 'bash2', description: 'd' })
      makeHostBash({ name: 'buildShell', description: 'd' })
      makeFileEditor({ sandbox, name: 'editor2', description: 'd' })
    })
  })

  describe('inputSchema contract constraints', () => {
    it('makeBash rejects schemas whose output lacks the required command field', () => {
      // @ts-expect-error - parsed output has no 'command'
      makeBash({ inputSchema: z.object({ cmd: z.string() }) })
      // @ts-expect-error - 'command' parses to number, not string
      makeBash({ inputSchema: z.object({ command: z.number() }) })
    })

    it('makeBash accepts schemas whose output satisfies { command, timeout? }', () => {
      makeBash({ inputSchema: z.object({ command: z.string() }) })
      makeBash({ inputSchema: z.object({ command: z.string(), timeout: z.number().positive().optional() }) })
      // Extra parsed fields are fine — the callback ignores them.
      makeBash({ inputSchema: z.object({ command: z.string(), reason: z.string() }) })
    })

    it('makeHostBash rejects schemas whose output lacks mode', () => {
      // @ts-expect-error - parsed output has no 'mode'
      makeHostBash({ inputSchema: z.object({ command: z.string() }) })
    })

    it('makeFileEditor rejects schemas incompatible with its command/path contract', () => {
      // @ts-expect-error - parsed output has no 'path'
      makeFileEditor({ inputSchema: z.object({ command: z.literal('view') }) })
      // @ts-expect-error - 'destroy' is not a valid command literal
      makeFileEditor({ inputSchema: z.object({ command: z.literal('destroy'), path: z.string() }) })
    })
  })

  describe('schema overrides narrow the created tool’s invoke() type', () => {
    it('default makeBash invoke() takes { command, timeout? }', () => {
      const defaultBash = makeBash({ sandbox })
      expectTypeOf(defaultBash.invoke).parameter(0).toEqualTypeOf<{
        command: string
        timeout?: number | undefined
      }>()
    })

    it('custom schema flows through to invoke()', () => {
      const audited = makeBash({
        sandbox,
        inputSchema: z.object({ command: z.string(), reason: z.string() }),
      })
      expectTypeOf(audited.invoke).parameter(0).toEqualTypeOf<{ command: string; reason: string }>()

      // @ts-expect-error - 'reason' is required by the custom schema
      void audited.invoke({ command: 'ls' })
    })
  })

  describe('documented limitation: spread of intermediate variables', () => {
    it('a typo’d key smuggled via spread with an overlapping property compiles (known TS behavior)', () => {
      const opts = { name: 'bash2', descriptin: 'typo not caught here' }
      // No @ts-expect-error: excess-property checks apply only to inline object
      // literals. This is the documented trade-off of the flat options bag.
      makeBash({ ...opts })
    })

    it('weak-type check still rejects option bags sharing no properties', () => {
      const unrelated = { descriptin: 'oops' }
      // @ts-expect-error - no overlapping properties: weak-type check fires
      makeBash(unrelated)
    })
  })

  describe('ToolSpecOverrides convention', () => {
    it('MakeBashOptions is the binding options intersected with ToolSpecOverrides', () => {
      expectTypeOf<MakeBashOptions['name']>().toEqualTypeOf<ToolSpecOverrides['name']>()
      expectTypeOf<MakeBashOptions['description']>().toEqualTypeOf<ToolSpecOverrides['description']>()
      expectTypeOf<MakeBashOptions['sandbox']>().toEqualTypeOf<Sandbox | undefined>()
    })
  })
})
