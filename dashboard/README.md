# BlackICE-Mesh · Dashboard

Realtime telemetry terminal for the adversarial ML defence mesh. Built with Next.js
(App Router, Turbopack), Tailwind CSS v4, and TypeScript (strict).

## Design System Constraints

Applied from the [Design-Dungeons](https://github.com/pd241008/Design-Dungeons)
engineering playbook plus explicit UI constraints:

- **Watch-Dogs terminal aesthetic** — dark void backdrop (`#06090a`) with a faint grid,
  CRT scanlines, blinking block cursor, and `Share Tech Mono` display font.
- **Neobrutalist structures** — every frame uses harsh 2px borders, hard offset
  shadows, zero softening, and high-contrast accents (electric cyan `#00f0ff`,
  hazard red `#ff2e2e`, warning amber `#ffb000`).
- **Glassmorphism** — floating data panels (`glass-panel-floating`) with
  `backdrop-filter: blur()` over the dark background for Clean Accuracy,
  Robust Accuracy, and Attack Success Rate diagnostics.
- **Unified API envelope** — the gateway returns `{success, data, error}` with
  aligned HTTP status codes; consumed strictly in `src/lib/api.ts`.
- **No `any`** — strict typing across `src/lib/types.ts`, `src/lib/api.ts`, hooks,
  and components.

## Dev

```bash
npm run dev        # http://localhost:3000
npm run lint
npm run build
```

Set `NEXT_PUBLIC_GATEWAY_URL` (default `http://localhost:8080`) to point the
dashboard at the gateway-service.

## Layout

```text
src/
├── app/            # App Router: layout + dashboard page
├── components/     # Terminal/neobrutalist/glass UI + feature panels
├── hooks/          # useTelemetry (WebSocket streaming)
└── lib/            # api client + strict types
```
