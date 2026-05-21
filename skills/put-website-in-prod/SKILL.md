---
name: put-website-in-prod
description: Deploy the libreyolo-website (this repo) to production on Vercel. Use when the user asks to "deploy", "push to prod", "put it live", "ship the site", or to verify a manual production deploy. Covers the Vercel CLI install, login, the specific Vercel project this repo is linked to, the correct deploy command, and the most common pitfalls.
---

# Put the website in production

This repo (`libreyolo-website`) is a Next.js 16 app deployed on Vercel. The production domain is `https://www.libreyolo.com`.

## Where the site lives on Vercel

- **Scope:** `xubanceccons-projects` (Xuban's personal Vercel account, not a team)
- **Project:** `libreyolo-website`
- **Production domain:** `www.libreyolo.com`
- **Vercel dashboard:** https://vercel.com/xubanceccons-projects/libreyolo-website

## The two deploy paths

1. **Auto-deploy via GitHub integration (default).** Pushing to `main` triggers a Vercel build automatically. Most of the time, this is all you need. Just `git push origin main` and the new commit goes live in ~1–2 minutes.
2. **Manual CLI deploy.** Use only when (a) you want to deploy local changes without pushing, (b) you need to redeploy without a new commit, or (c) the user explicitly asks for `vercel --prod`.

Confirm with the user which path they want before running the CLI deploy. Don't run a manual deploy if the user just wanted to push.

## CLI prereqs

Use the **globally-installed** `vercel` CLI (currently v54+). Do **not** use `npx vercel` — that picks up the stale `vercel@50.1.3` pinned in `package.json` as a project dependency, whose auth token format is incompatible with v54. If you see `Error: The specified token is not valid`, you almost certainly ran `npx vercel` instead of plain `vercel`.

Install globally if missing:

```bash
npm install -g vercel
```

Verify:

```bash
vercel --version   # expect 54.x or newer
```

## Login flow

`vercel login` is interactive (opens a browser URL with a device code). **Don't run it from a non-interactive Claude Code session** — it will hang waiting for the user to confirm in the browser. Instead, ask the user to run it themselves:

> Please run `vercel login` in your terminal and confirm the device code in the browser. Once it says "Congratulations! You are now signed in.", I'll continue.

If you're invoked inline (e.g. with `!`-prefixed commands), running it inline works because stdout/stderr stream to the user.

After login, the token is stored at `~/.vercel/auth.json` (or the equivalent on the platform) and persists across sessions.

## Manual production deploy

From the repo root:

```bash
vercel --prod --yes
```

- `--prod` targets the production environment (aliases to `www.libreyolo.com`).
- `--yes` accepts defaults non-interactively. First run will auto-link the local directory to the existing `xubanceccons-projects/libreyolo-website` project by name match. Subsequent runs read `.vercel/project.json` and skip the prompt.
- Without `--yes`, the CLI prompts ("Set up and deploy ~/path? Y/n", "Which scope?", "Link to existing project?") and will hang in non-interactive sessions.

Expected output ends with:

```
▲ Production  https://libreyolo-website-<hash>-xubanceccons-projects.vercel.app
▲ Aliased     https://www.libreyolo.com
```

If you see `Aliased`, the production domain is updated.

## Verifying the deploy

- Open `https://www.libreyolo.com` (or the specific page that changed) and confirm visually.
- For build details / logs: `vercel inspect <deployment-url>` using the URL from the previous step.
- For history: `vercel ls` lists recent deploys; the topmost with `● Ready` and `Production` is the live one.

## Common pitfalls

- **`Error: The specified token is not valid`** → you ran `npx vercel`, which uses the pinned v50.1.3. Use the global `vercel` command directly.
- **`No existing credentials found`** → user is logged out. Ask them to run `vercel login`.
- **`.vercel/` appearing in `git status`** → it's already in `.gitignore`; don't stage it. It holds the local project link (`project.json`) and is per-machine.
- **Deploy succeeded but `www.libreyolo.com` shows old content** → DNS / browser cache. Hard-refresh (Ctrl+Shift+R) and check from an incognito window before assuming the deploy is broken.
- **Two production builds running simultaneously** → you pushed to `main` *and* ran `vercel --prod`. Both will complete; the later finish wins. Usually harmless, but wasteful.

## When NOT to manually deploy

- Right after `git push origin main` — the auto-deploy is already running. Wait for it.
- If the user only asked you to commit/push. Pushing is not the same as deploying via CLI; don't escalate.
- If the build is currently failing on Vercel — fix the underlying error first, don't loop on manual redeploys.
