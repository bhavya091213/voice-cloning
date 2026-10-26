import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

import type { Check, Doctor, Output, Phase, Snap, View } from '../types'

const ROOT = 'D:/VoiceClone'
const PY = `${ROOT}/.venv/Scripts/python.exe`
const VC = `${ROOT}/scripts/vc.py`
const PANE = 'voiceclone'
const snap = atom(
  { plugin: 'voiceclone-progress', key: 'snap' } as const,
  { progress: null, pipeline: null, outputs: [], doctor: null } as Snap,
  { shape: 'snap-2' },
)
const view = atom({ plugin: 'voiceclone-progress', key: 'view' } as const, { tab: 'progress', busy: '', note: '' } as View, {
  shape: 'view-1',
})

const GLYPH: Record<string, string> = { done: '✔', running: '▶', pending: '○', error: '✖', skipped: '–' }
const COLOR: Record<string, string | undefined> = { done: 'green', running: 'cyan', error: 'red', skipped: 'yellow' }
const TABS: [View['tab'], string][] = [
  ['progress', 'Progress'],
  ['outputs', 'Outputs'],
  ['doctor', 'Doctor'],
]

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await $.command.register({ name: 'vc-progress', description: 'Open the VoiceClone progress / outputs / doctor pane' })
    void $.ui.open({ id: PANE, title: 'VoiceClone' })

    let last = ''
    const metaCache: Record<string, { lufs?: number; peak?: number }> = {}
    const load = async (file: string) => {
      try {
        return JSON.parse(await $.fs.read(`${ROOT}/${file}`))
      } catch {
        return null
      }
    }
    const listOutputs = async (): Promise<Output[]> => {
      try {
        const files = (await $.fs.list(`${ROOT}/output`))
          .filter(f => f.kind === 'file' && f.name.endsWith('.wav'))
          .sort((a, b) => b.mtimeMs - a.mtimeMs)
          .slice(0, 12)
        const rows: Output[] = []
        for (const f of files) {
          const id = f.name.replace(/\.wav$/, '')
          if (!(id in metaCache)) {
            const m = await load(`work/jobs/${id}/meta.json`)
            metaCache[id] = { lufs: m?.master?.lufs, peak: m?.master?.true_peak_db }
          }
          rows.push({
            name: id,
            path: `${ROOT}/output/${f.name}`,
            mtimeMs: f.mtimeMs,
            sizeMb: Math.round(f.size / 1e5) / 10,
            ...metaCache[id],
          })
        }
        return rows
      } catch {
        return []
      }
    }
    $.clock.every(1500, async () => {
      const fresh: Snap = {
        progress: await load('work/skill_progress.json'),
        pipeline: await load('work/status.json'),
        outputs: await listOutputs(),
        doctor: await load('work/doctor.json'),
      }
      const key = JSON.stringify(fresh)
      if (key === last) return
      last = key
      await update($, snap, () => fresh)
    })

    return next(e)
  })

  on('command.run', { command: 'vc-progress' }, async $ => {
    await $.ui.open({ id: PANE, title: 'VoiceClone' })
    return { text: 'VoiceClone pane opened.' }
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text, Button } = $.ui.resolve(e)
    const s = await read($, snap)
    const v = await read($, view)
    const set = (patch: Partial<View>) => void update($, view, cur => ({ ...cur, ...patch }))

    const play = async (o: Output) => {
      set({ note: `playing ${o.name}` })
      const r = await $.process.run(['powershell', '-NoProfile', '-Command', `Start-Process -FilePath '${o.path}'`])
      if (r.exitCode !== 0) set({ note: `play failed: ${r.stderr.slice(0, 120)}` })
    }
    const runDoctor = async (fix: boolean) => {
      set({ busy: fix ? 'doctor --fix' : 'doctor', note: '' })
      try {
        const r = await $.process.run([PY, VC, 'doctor', ...(fix ? ['--fix'] : [])], { timeoutMs: 240000 })
        const d: Doctor = JSON.parse(r.stdout)
        await update($, snap, cur => ({ ...cur, doctor: d }))
        set({ busy: '', note: d.ok ? 'all checks passed' : `${d.checks.filter((c: Check) => !c.ok).length} problem(s)` })
      } catch (err) {
        set({ busy: '', note: `doctor failed: ${String(err).slice(0, 120)}` })
      }
    }

    const tabs = (
      <Box>
        {TABS.map(([id, label]) => (
          <Button key={id} variant={v.tab === id ? 'primary' : undefined} onPress={() => set({ tab: id })}>
            {label}
          </Button>
        ))}
      </Box>
    )
    const note = v.note !== '' ? <Text dimColor>{v.note}</Text> : null

    if (v.tab === 'outputs') {
      const link = s.doctor?.link ?? ''
      return (
        <Box flexDirection="column">
          {tabs}
          {s.outputs.length === 0 && <Text dimColor>No files in output\ yet.</Text>}
          {s.outputs.map(o => (
            <Box key={o.name}>
              <Button onPress={() => void play(o)}>▶ {o.name}</Button>
              <Text dimColor>
                {' '}{o.sizeMb} MB{o.lufs !== undefined ? ` · ${o.lufs} LUFS` : ''}{o.peak !== undefined ? ` · ${o.peak} dBTP` : ''}
              </Text>
            </Box>
          ))}
          {link !== '' && (
            <Button onPress={press => void $.ui.copy({ text: link, surface: press.surface })}>Copy Studio link</Button>
          )}
          {note}
        </Box>
      )
    }

    if (v.tab === 'doctor') {
      const d = s.doctor
      return (
        <Box flexDirection="column">
          {tabs}
          <Box>
            <Button onPress={() => void runDoctor(false)}>Run checks</Button>
            <Button onPress={() => void runDoctor(true)}>Run + fix</Button>
          </Box>
          {v.busy !== '' && <Text color="cyan">▶ {v.busy} running (GPU check takes a few seconds)…</Text>}
          {!d && <Text dimColor>No report yet. Run checks.</Text>}
          {d && (
            <Text color={d.ok ? 'green' : 'red'} bold>
              {d.ok ? '✔ healthy' : '✖ needs attention'} · {new Date(d.checked * 1000).toLocaleTimeString()}
            </Text>
          )}
          {d?.checks.map((c: Check) => (
            <Box key={c.id} flexDirection="column">
              <Text color={c.ok ? undefined : 'red'} dimColor={c.ok}>
                {c.ok ? '✔' : '✖'} {c.label}
                {c.ok && c.detail !== '' ? ` · ${c.detail.slice(0, 50)}` : ''}
              </Text>
              {!c.ok && <Text color="yellow">    fix: {c.fix}</Text>}
            </Box>
          ))}
          {note}
        </Box>
      )
    }

    const { progress, pipeline } = s
    if (!progress) {
      return (
        <Box flexDirection="column">
          {tabs}
          <Text dimColor>No run yet. Ask Claude to use the voiceclone skill.</Text>
        </Box>
      )
    }
    const done = progress.phases.filter((p: Phase) => p.state === 'done' || p.state === 'skipped').length
    const total = progress.phases.length
    const bar = '█'.repeat(done) + '░'.repeat(total - done)
    const active = pipeline && pipeline.stages.some(st => st.state === 'running' || st.state === 'error')

    return (
      <Box flexDirection="column">
        {tabs}
        <Text bold>{progress.label || 'VoiceClone run'}</Text>
        <Text color="cyan">{bar} {done}/{total}</Text>
        {Object.entries(progress.inputs).map(([k, val]) => (
          <Text key={k} dimColor>{k}: {val}</Text>
        ))}
        <Text> </Text>
        {progress.phases.map((p: Phase) => (
          <Box key={p.id} flexDirection="column">
            <Text color={COLOR[p.state]} dimColor={p.state === 'pending'} bold={p.state === 'running'}>
              {GLYPH[p.state] ?? '?'} {p.id}. {p.name}
            </Text>
            {p.detail !== '' && p.state !== 'pending' && <Text dimColor>    {p.detail}</Text>}
          </Box>
        ))}
        {pipeline && active && (
          <Box flexDirection="column">
            <Text> </Text>
            <Text bold>Pipeline · {pipeline.job}</Text>
            {pipeline.stages.map(st => (
              <Text key={st.name} color={COLOR[st.state]} dimColor={st.state === 'pending'}>
                {GLYPH[st.state] ?? '?'} {st.name} {st.detail}
              </Text>
            ))}
          </Box>
        )}
      </Box>
    )
  })
}
