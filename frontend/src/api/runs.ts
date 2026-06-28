import type { RunProjection } from '../state/types'

export async function fetchRun(runId: string): Promise<RunProjection> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`, {
    headers: { accept: 'application/json' },
  })
  if (!response.ok) {
    throw new Error('Unable to reconcile run.')
  }
  return (await response.json()) as RunProjection
}
