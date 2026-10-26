export type Phase = { id: number; name: string; state: string; detail: string }
export type Progress = { label: string; started: number; inputs: Record<string, string>; phases: Phase[] }
export type Stage = { name: string; state: string; detail: string }
export type PipelineStatus = { job: string; stages: Stage[] }
export type Output = { name: string; path: string; mtimeMs: number; sizeMb: number; lufs?: number; peak?: number }
export type Check = { id: string; label: string; ok: boolean; detail: string; fix: string }
export type Doctor = { checked: number; ok: boolean; link: string; checks: Check[] }
export type Snap = {
  progress: Progress | null
  pipeline: PipelineStatus | null
  outputs: Output[]
  doctor: Doctor | null
}
export type View = { tab: 'progress' | 'outputs' | 'doctor'; busy: string; note: string }

declare module 'claude-code' {
  interface PluginState {
    'voiceclone-progress': { snap: Shaped<Snap>; view: Shaped<View> }
  }
}
