# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.
Also includes a standalone Python/Streamlit app: **MixWise** (Marketing Mix Modeling).

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)
- **Python**: 3.11 with Streamlit, pandas, numpy, scikit-learn, plotly, scipy

## MixWise App

- **Location**: `artifacts/mixwise/app.py`
- **Config**: `artifacts/mixwise/.streamlit/config.toml`
- **Workflow**: "MixWise" — `streamlit run artifacts/mixwise/app.py --server.port 5000`
- **Design**: Dark theme (#0d0f14 bg, #13161e card, #5b8dee accent), DM Serif Display / DM Mono / DM Sans

### Pages
1. **Upload & Clean** — CSV upload or demo data, data quality report, correlation heatmap, cleaning tools
2. **Model Builder** (or Lift Calculator in Tier 3) — OLS/Ridge regression, coefficient table, predicted vs actual
3. **Priors & Adstock** — Per-channel adstock decay sliders, saturation toggles, before/after comparison
4. **ROAS Dashboard** — 4 metric cards, stacked bar chart, ROAS table, waterfall chart, budget optimizer
5. **A/B Testing** — Coming soon stub with early access button

### Data Maturity Tiers
- **Tier 1** (104+ weeks, 3+ channels): Full MMM
- **Tier 2** (26–103 weeks or 2 channels): Lite MMM with industry priors
- **Tier 3** (<26 weeks): Incrementality Mode / Lift Calculator

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.
