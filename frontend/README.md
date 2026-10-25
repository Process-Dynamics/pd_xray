# pd_xray frontend

React + OpenSeadragon UI, built into `pd-xray-server` (`src/pd_xray/server/static/`). Users never need Node.

```bash
npm ci                 # exact versions from package-lock.json
npm run dev            # http://localhost:5173, API proxied to `pd-xray serve` on :8000
npm run format         # Prettier; CI-style check: npm run format:check
npm test && npm run build
```

- `src/api.ts`: server API types and calls. `src/scheduling.ts`: pure logic (latest-wins loading, levels, prefetch).
- `src/Viewer.tsx`: frames load hidden and swap in when complete, so the screen shows one frame at a time.
