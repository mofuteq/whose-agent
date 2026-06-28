import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('frontend source boundary', () => {
  it('does not directly reference private runtime history types or private event names', () => {
    const root = join(process.cwd(), 'src')
    const source = listFiles(root)
      .filter((path) => /\.(css|tsx?|jsx?)$/.test(path))
      .map((path) => readFileSync(path, 'utf8'))
      .join('\n')
    const forbidden = [
      ['Whose', 'Agent', 'State'].join(''),
      ['Conversation', 'View'].join(''),
      ['Message', 'View'].join(''),
      ['Authority', 'Cause', 'Record'].join(''),
      ['STATE', '_SNAPSHOT'].join(''),
      ['STATE', '_DELTA'].join(''),
    ]

    for (const token of forbidden) {
      expect(source).not.toContain(token)
    }
  })
})

function listFiles(directory: string): string[] {
  return readdirSync(directory).flatMap((entry) => {
    const path = join(directory, entry)
    return statSync(path).isDirectory() ? listFiles(path) : [path]
  })
}
