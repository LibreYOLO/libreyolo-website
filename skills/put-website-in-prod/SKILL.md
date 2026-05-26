---
name: put-website-in-prod
description: Deploy libreyolo-website to production on Vercel using the global vercel CLI (vercel --prod --yes). Use when the user asks to deploy, publish, push to prod, put it live, or ship the site.
---

# Put the website in production

This repo (`libreyolo-website`) is a Next.js 16 app on Vercel. Production is **`https://www.libreyolo.com`**.

Publishing means running the **global** Vercel CLI from the repo root. That is the only supported deploy path for agents and humans in this project.

## Where the site lives on Vercel

- **Scope:** `xubanceccons-projects`
- **Project:** `libreyolo-website`
- **Production domain:** `www.libreyolo.com`
- **Dashboard:** https://vercel.com/xubanceccons-projects/libreyolo-website

## Prereqs

Use the **globally installed** `vercel` CLI (v54+). **Never** use `npx vercel` or `npx vercel --prod`. The repo pins `vercel@50.1.3` in `package.json`; that old version uses an incompatible token format and fails with `Error: The specified token is not valid`.

Install globally if missing:

```bash
npm install -g vercel
```

Verify:

```bash
vercel --version   # expect 54.x or newer
```

## Login (if needed)

`vercel login` is interactive (browser device code). Do not run it from a non-interactive agent session; it will hang.

Ask the user to run it in their terminal:

> Run `vercel login` and confirm the device code in the browser. When you see "Congratulations! You are now signed in.", tell me and I will deploy.

The token persists at `~/.vercel/auth.json` (platform equivalent on Windows).

## Production deploy

From the repo root:

```bash
vercel --prod --yes
```

- **`--prod`** deploys to production and aliases to `www.libreyolo.com`.
- **`--yes`** accepts defaults non-interactively (required in agent sessions).
- Deploys the **current working tree** on disk. You do not need a git push for the site to update.
- First run links this directory to `xubanceccons-projects/libreyolo-website` via `.vercel/project.json`. Later runs reuse that link.

Success looks like:

```
▲ Production  https://libreyolo-website-<hash>-xubanceccons-projects.vercel.app
▲ Aliased     https://www.libreyolo.com
```

If you see **`Aliased`**, production is live.

## Verify

- Open `https://www.libreyolo.com` (or the page that changed) and confirm visually.
- Build logs: `vercel inspect <deployment-url>` using the Production URL from the output.
- Recent deploys: `vercel ls` (top row with `● Ready` and `Production` is live).

## Common pitfalls

| Symptom | Fix |
|--------|-----|
| `The specified token is not valid` | You used `npx vercel`. Run global `vercel --prod --yes` instead. |
| `No existing credentials found` | User must run `vercel login`. |
| CLI prompts and hangs | Add `--yes`, or ensure `.vercel/project.json` exists from a prior link. |
| Site shows old content after success | Hard-refresh (Ctrl+Shift+R) or incognito; cache, not a failed deploy. |
| `.vercel/` in `git status` | Already gitignored; do not commit it. |

## What this skill does not cover

- **Git commit / push** is separate from publishing. Only commit or push when the user asks; publishing is always `vercel --prod --yes`.
- Do not redeploy in a loop if the Vercel build is failing. Read the build log, fix the error, then deploy once.
