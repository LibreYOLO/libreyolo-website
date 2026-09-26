# Plan: LibreYOLO v1.6.0 documentation

Owner of the plan: Xuban. Executor: an agent working in this worktree.
Branch: `docs/v1.6.0`, cut from `main` at `19c1ca6`. Draft PR to `main`.

Target: the docs tree under `content/docs/` describes LibreYOLO 1.6.0 on the
day it ships. v1.6.0 is not tagged yet, and some PRs still have to land on the
library's `dev`, so the goal is about 90% done now and a short catch-up pass at
release time (Phase 7).

## 0. Read before touching anything

In this repo, in this order:

1. `AGENTS.md`: sitemap, `llms.txt`, no em dashes, and the translation rule.
2. `skills/write-docs-page/SKILL.md`: the page-type specs, the blocks, the
   verification rules and the pre-commit checklist. It wins over this plan on
   anything about how a page looks or reads.
3. `skills/humanizer/SKILL.md`, with the two overrides listed in
   `write-docs-page` (US spelling, no voice on reference pages).
4. `TRANSLATION_PLAN.md` and `scripts/translation/STYLE-<locale>.md`.
5. `scripts/build-registry/` (how `src/data/docs/registry.json` is built).

Inputs prepared for this work, outside the repo, in `~/release-1.6.0/`:

| File | What it is |
| --- | --- |
| `CHANGELOG-1.6.0.md` | The draft 1.6.0 changelog: 104 entries, every one with PR numbers. **This is the work list.** |
| `CHANGELOG-1.6.0-evidence.md` | Per-entry evidence (file:line at library `dev`), exact defaults, weight URLs and licenses, and a list of known discrepancies. Look up an entry's ID (M02, T07, C10 and so on) here before writing about it. |
| `v160-inventory.md`, `v160-raw.json` | An earlier, cheaper per-PR inventory. Less reliable; the changelog supersedes it. |

The library checkout is `/Users/xuban.ceccon/Documents/personal/libreyolo`.
Its working tree is someone else's branch: **never edit, switch branches,
commit or stash there.** Read the library at `dev` through git instead:
`git -C <libreyolo> fetch origin && git -C <libreyolo> show origin/dev:<path>`.
If you need a checked-out tree (the registry builder does), make a detached
worktree of `origin/dev` in your scratchpad and remove it when done.

The library's contributor docs are the best source text for several new
features. They changed in this range:
`docs/hf_hub.md`, `docs/fiftyone.md`, `docs/librellm.md`, `docs/librevla.md`,
`docs/libreground_design.md`, `docs/custom_trackers.md`,
`docs/classification_training.md`, `docs/classification_augmentation.md`,
`docs/input_profiles.md`, `docs/levjepa.md`, `docs/quantization.md`,
`docs/kernels.md`, `docs/checkpoint_schema.md`, `docs/dataset_schema.md`,
`docs/predict_sources.md`, `docs/export_support.md`, and ADRs 0019 to 0029
(LibreLLM, LibreGround, detect3d, 3D-MOOD, FCOS3D, DetAny3D, depth encodings,
albedo, Marigold V2, LibreVLA, event histograms). They are contracts written
for contributors: take facts from them, never their prose.

## 1. Rules that override your instincts

- **Facts come from code at library `origin/dev`, not from the changelog
  text.** The changelog tells you what to document; `git show origin/dev:...`
  tells you what is true. When they disagree, the code wins, and you add a
  line to "Changelog corrections" in the PR description.
- **Weights licensing:** a checkpoint carries the license its upstream
  publisher declared, and nothing more. Do not add non-commercial caveats,
  banners or notes because of the training dataset (Cityscapes, nuScenes,
  DOTA, Objects365, Adobe DIM and so on). Some library code and notices still
  carry such dataset-based caveats; do not copy them onto the site. When the
  upstream publisher declares a restrictive license (ConvNeXt V2 CC-BY-NC-4.0,
  LeVJEPA CC-BY-NC-4.0, YOLO-NAS, Moondream 3 BSL 1.1, Dome-DETR
  academic-only, EdgeCrafter `obj2coco`, DetAny3D CC-BY-NC-4.0), state it in
  the `<provenance-box>` as the skill requires.
- Never name or compare to Roboflow in prose. The model name RF-DETR is fine.
  Never name a competing library at all (skill rule).
- Checkpoint names only if they are in the registry, which is gated on the
  Hugging Face org listing. Code routes a name to a URL; that does not prove
  the file was uploaded. The evidence file flags several as "availability not
  probed" (U-Net, some TinyFormer combinations, Molmo2 8B/O-7B and others).
- Snippets live in frontmatter, run on CPU as pasted, and use only
  `SAMPLE_IMAGE` or the two shipped images. A family that cannot run on CPU
  (Marigold 4-bit, the 3D detectors with external runtimes) says so in the
  Install section and its snippet states the requirement in a comment.
- No em dashes, emoji, "experimental" or its synonyms. US spelling.
- `last_verified: "1.6.0"` on every page you create or check against 1.6.0.
  Do not bump it on pages you did not verify.

## 2. Setup (done)

- Worktree: `/Users/xuban.ceccon/Documents/personal/libreyolo-website-docs-v160`
- Branch `docs/v1.6.0` off `origin/main`, pushed, draft PR open to `main`.
- `node_modules` is a symlink to the main checkout's. `npm run dev` works
  through it; `next build` under Turbopack fails on the symlink (known, see
  the diagrams work). Validate a full build with a Vercel **preview** deploy
  only. Never `vercel --prod`: pushing and merging do not deploy, and
  production goes live only via `skills/put-website-in-prod` after the
  library release.
- Commit identity for this repo:
  `git -c user.name="Xuban" -c user.email="59646791+EHxuban11@users.noreply.github.com" commit ...`
- No `Co-Authored-By` or session lines in commits or the PR.

## 3. Phases

Work in this order. Commit after every page or small group, with a message
that names the pages. Push at the end of each phase so the draft PR shows
progress. Update the checklist in section 5 as you go; it is the resume point
if the session dies.

### Phase 1: Registry against library dev

The registry says `libreyolo_version: "1.5.0"` and has 82 families. The new
1.6.0 families are missing, so the generated blocks (`checkpoint-table`,
`export-matrix`, `provenance-box`, `citation-block`, `benchmark-table`) render
nothing for them.

1. Read `scripts/build-registry/*.mjs` to learn its inputs, then run it against
   a detached worktree of library `origin/dev`. Record the dev SHA in the
   registry's source field so Phase 7 knows what to re-run.
2. Add every new family to `lineages.mjs` where a page covers a lineage.
3. For each new family write `src/data/docs/upstream/<slug>.json` using the
   citation procedure in the skill: BibTeX copied verbatim from the authors'
   own README or `CITATION.cff`, then cross-checked against arXiv or the
   publisher. No verified BibTeX means no Citation section; never assemble one.
   License fields: upstream-declared only (rule above).
4. Diff the regenerated registry against `main`'s. Anything that changed for an
   existing family (export support, sizes, tiers) is a fact some existing page
   may now get wrong; list those families for Phase 3.

Exit: every family in section 4A resolves in the registry, and each page's
blocks render when you `curl` the page on the dev server.

### Phase 2: New pages (English only)

New English pages ship without translations; they fall back to English until
translated. Follow the page-type spec in the skill exactly: model page, task
page, workflow page, CLI page, reference page. Section 4 lists every page.

Several model pages exist already as **architecture-only stubs** from the
diagrams work (`architecture_only: true`, `families: []`, about 15 lines):
PP-YOLOE, PP-LiteSeg, QuickSRNet, ViTMatte, V-JEPA 2, PE, LaMa, HVI-CIDNet,
ShowUI, North Micro Vision. Turn them into full model pages: set `families`,
remove `architecture_only`, keep the diagram section at the end, and follow
the model-page section order. Check whether each stub already has locale
twins; if it does, see Phase 6.

Per page:

1. Read the family's `model.py`, `UNSUPPORTED_TRAIN_PARAMS`, capability flags
   and defaults at library `origin/dev`. Read the changelog entry and its
   evidence block.
2. Write the page. Prose authored, facts from blocks.
3. `curl -s localhost:3000/docs/<path>` and grep for each block's content.
   Check 390 px width and dark mode for anything with a table.
4. Run the pre-commit checklist from the skill.

### Phase 3: Update existing English pages

Section 4B lists them, each with the changelog IDs it has to absorb. Edit in
place; these URLs are permanent. For every behavior change, the page states
the 1.6.0 behavior plainly. The migration from 1.5.0 goes on the upgrade page
(4C), not scattered as "changed in 1.6.0" notes across pages.

Every page edited here has 13 locale twins, which Phase 6 has to update.
Keep edits surgical so that translation stays proportional.

### Phase 4: Release-level pages

`start/changelog.md`, `start/upgrade.md` (or `migrate.md`, whichever the site
uses for version upgrades; read both), `start/install.md` (the new extras),
`start/versions.md`, `start/weights.md`, `start/licensing.md`. Section 4C has
the details.

`versions.md` needs a decision that is **not yours**: whether the 1.5.0 v2
tree gets frozen at a versioned path (the way 1.1.0 to 1.4.0 were) when 1.6.0
replaces it. Draft the table row for 1.6.0 as current and leave the 1.5.0 row
with a `TODO(owner)` comment in the markdown, then raise it in the PR
description. Do not build archive routes.

### Phase 5: Site plumbing

- `src/data/docs/nav.json`: every new page, in the right section.
- `src/app/sitemap.js`: every new canonical page (English only until twins
  exist). Verify the generated sitemap on the dev server.
- `src/app/llms.txt/route.js`: static page list.
- Section index pages (`/docs/models`, `/docs/tasks` and so on) if they are
  hand-listed rather than generated; check.
- The homepage task explorer says "17 tasks"; 1.6.0 has 20. Update it only if
  the number is not generated from the registry, and never redesign the
  homepage (it is protected).

### Phase 6: Translations

`AGENTS.md` is absolute here: **an English page that has twins cannot change
without its twins.** For each page edited in Phases 2 to 5 that has twins,
update all 13 (`de es fr id it ja ko pl pt ru uk vi zh`) following
`STYLE-<locale>.md`, then:

```bash
node scripts/translation/validate.mjs <locale>      # every locale
node scripts/translation/sync-check.mjs             # must pass
node scripts/translation/sync-check.mjs --stamp     # only after retranslating
```

This is the biggest mechanical block. Fan it out: one subagent per locale,
each given the list of changed English pages and their diffs. Stamping without
retranslating defeats the check and is not allowed.

Brand-new pages: translate them only if time remains after everything else
passes. They are correct in English without twins.

### Phase 7: Release-time catch-up (not now; leave it scripted)

When the library tags v1.6.0:

1. `git -C <libreyolo> log <recorded dev SHA>..v1.6.0 --first-parent --oneline`
   lists what landed after this work. Document each PR the same way.
2. Re-run the registry against the tag, not dev. Diff again.
3. Grep this branch for `TODO(1.6.0)` markers (use that exact form for anything
   waiting on an unmerged library PR).
4. Replace `dev` raw links with `release` ones (raw links must never use
   `/main/`).
5. Hand off to `skills/put-website-in-prod`.

Write these steps into the PR description so whoever finishes it does not need
this file.

## 4. Page inventory

IDs in brackets are changelog entries in `~/release-1.6.0/CHANGELOG-1.6.0.md`.
This inventory is a starting point: the family and slug lists came from the
changelog, not from `lineages.mjs`. Confirm each slug against
`lineages.mjs` and the existing naming before creating a file, and add any page
the changelog implies that is missing here.

### 4A. New or stub-to-full pages

Model pages (`content/docs/models/`):

| Page | Status | Covers |
| --- | --- | --- |
| `ppyoloe.md` | stub to full | PP-YOLOE s/m/l/x, trainable [M02] |
| `tinyformer.md` | new | TinyFormer s to xl, VisDrone and Objects365-to-COCO variants, rectangle limit [M03] |
| `dekr.md` | new | Bottom-up pose, inference only [M04] |
| `ppliteseg.md` | stub to full | t50/b50/t75/b75, rectangular canvases, training [M06, T12] |
| `unet.md` | new | U-Net semantic, trainable, no export [M07] |
| `convnextv2.md` | new | 8 sizes, CC-BY-NC-4.0 weights [M08, T08] |
| `pe.md` | stub to full | Zero-shot, image/text/video embeddings, `clip_frames` [M09] |
| `vjepa2.md` | stub to full | Video embeddings and probes, `embed_tokens()` [M10] |
| `levjepa.md` | new | Clip and patch embeddings, CC-BY-NC-4.0 [M11] |
| `ben2.md` | new | Background removal [M12] |
| `quicksrnet.md` | stub to full | 2x super-resolution [M13] |
| `ddcolor.md` | new | Colorization [M14] |
| `hvi-cidnet.md` | stub to full | Low-light, `gamma`/`saturation`/`intensity` [M14] |
| `lama.md` | stub to full | Inpainting, `mask=` [M14] |
| `vitmatte.md` | stub to full | Guided matting, `trimap=` [M14] |
| `wilddet3d.md`, `3d-mood.md`, `fcos3d.md`, `detany3d.md` | new | 3D detection, intrinsics, external runtimes [M15] |
| `marigold-v2.md` | new | 9 adapters, depth/normals/albedo, 4-bit CUDA [M16, C04] |
| `smolvla.md` or one `librevla.md` | new | SmolVLA, ACT, Diffusion Policy. Decide by `lineages.mjs` [M20] |
| `molmo2.md`, `gemma-4.md`, `moondream.md` | new | VLM additions [M18] |
| `northmicrovision.md`, `showui.md` | stub to full | [M18, M17] |

Task pages (`content/docs/tasks/`): `3d-object-detection.md` (`detect3d`),
`robot-policies.md` (`act`), `albedo-estimation.md` (`albedo`) [M01]. Pick
slugs that match the existing naming (`depth-estimation`, `surface-normals`).

Workflow pages: `train/fitness-callbacks.md` [T07]; `train/vlm-fine-tuning.md`
for Qwen3-VL LoRA [M19] (check whether `train/lora.md` should hold it
instead); `predict/tracking-custom.md` only if `tasks/object-tracking.md`
cannot absorb custom trackers and image sequences cleanly [A07, A08].

Reference pages: Hugging Face Hub loading, pushing and the logger [A09];
FiftyOne [A10]; `LibreLLM` [M21]; `LibreGround` [M17]; `LibreVLA` [M20]. Check
`vlm-api.md` and `openvocab-api.md` first; extend them where the surface
belongs there instead of creating a page.

CLI pages: `wilddet3d`, `3dmood`, `fcos3d`, `detany3d` [M15]. One page per
command, per the skill.

### 4B. Existing English pages to update (all have 13 twins)

| Page | Absorbs |
| --- | --- |
| `models/rf-detr.md` | UI weights [M24], multi-class pose [T11], event histograms [A06], OpenCV resize changes metrics [C05, B03], incremented run dirs [C11, B06], rectangular DETR predict [F10] |
| `models/yolov9.md` | PGI, `max_labels` 300, momentum warmup, letterbox stamp [C10, B05], event histograms [A06], TFLite INT8 [A12] |
| `models/yolox.md` | TFLite INT8 [A12], annotated mosaic partners [C12] |
| `models/yolo-nas.md` | OBB [M05], AMP default [C09] |
| `models/d-fine.md`, `deim.md`, `rt-detr.md` | AMP default [C09, B04] |
| `models/edgecrafter.md` | `obj2coco` weights, opt-in, upstream license [S03] |
| `models/dome-detr.md` | Auto-download mirrors, academic-only [S03] |
| `models/midas.md`, `moge-2.md`, `sam.md` | Mirrors [S03]; the MiDaS page must not carry a dataset caveat |
| `models/sam-3d-body.md` | Pinned assets, gated download [B02, S02] |
| `models/dinov2.md` | Run dirs, resume fix [C11, F12], class weights [T08] |
| `models/lfm2-vl.md`, `qwen3-vl.md`, `ground-*.md` | New sizes, fine-tuning, grounding [M17, M18, M19] |
| `models/convnext.md`, `resnet.md`, `mobilenetv4.md`, `efficientnetv2.md`, `vit.md`, `swin.md`, `deit.md`, `alexnet.md`, `vgg.md` | Classification eval preprocessing and calibration [C06, F07], loss weighting where supported [T08] |
| `tasks/object-detection.md` | PP-YOLOE, TinyFormer links; `classes=`, `single_cls` pointers |
| `tasks/pose-estimation.md` | DEKR, RF-DETR multi-class, better label errors [F05] |
| `tasks/oriented-detection.md` | YOLO-NAS OBB |
| `tasks/semantic-segmentation.md` | U-Net, PP-LiteSeg, rectangular datasets [T12] |
| `tasks/image-classification.md` | ConvNeXt V2, macro metrics [A03], loss weighting [T08], crop controls [T09] |
| `tasks/embeddings.md` | PE, V-JEPA 2, LeVJEPA, video clips |
| `tasks/image-restoration.md` | QuickSRNet, DDColor, HVI-CIDNet, LaMa |
| `tasks/background-removal.md` | BEN2, ViTMatte |
| `tasks/depth-estimation.md`, `surface-normals.md` | Marigold V2, depth `encoding` [C04] |
| `tasks/object-tracking.md` | Custom trackers [A07], image sequences, `fps`, `color_format` [A08] |
| `tasks/point-detection.md`, `open-vocabulary-detection.md` | Molmo2, Moondream, Gemma 4, LibreGround |
| `train/hyperparameters.md` | `classes`, `single_cls`, `min_samples`, `class_balanced`, `average_best`, `export_check`, `precise_bn`, `cls_pw`, `class_weights`, `plot_samples` [T01 to T08, A05] |
| `train/validation.md` | `visualize` [A01], `image_metrics` [A02], macro metrics [A03], `best_conf` [A04], `plot_samples` [A05] |
| `train/augmentations.md` | Classification crop, `crop_pct`, CLI flags, `no_aug_epochs` tail [T09, C08], `mixup + cutmix <= 1` [B07] |
| `train/multi-gpu.md` | `device=[0,1]` without a main guard [T10] |
| `train/loggers.md` | Hugging Face Hub logger [A09] |
| `train/datasets.md` | `kpt_names` [T11], `classes` [T01], label clipping [F01] |
| `train/event-histograms.md` | RF-DETR as well as YOLO9, the full input profile [A06] |
| `train/performance.md` | Matcher and fused-optimizer speedups [P01, P02] only with measured numbers; otherwise one factual line |
| `predict/results.md` | `plot()` for every task, `orig_img` [A13], saved top-5 [C07] |
| `predict/sources.md` | Image sequences for tracking [A08] |
| `predict/thresholds.md` | Pointer to `best_conf` [A04] |
| `export/tflite.md` | INT8 for YOLOX and YOLO9 [A12] |
| `export/quantization.md` | `mse` and `entropy` calibration [A31], QAT overrides [C13, B08] |
| `export/onnx.md` | `onnxruntime>=1.18.0` [B01, D02] |
| `reference/results-types.md` | `Boxes3D`, `AlbedoMap`, `Actions`, `DepthMap(encoding=)` [M01, C04] |
| `reference/python-api.md` | `LibreVLA`, `LibreGround`, `LibreLLM`, Hub loading, `push_to_hub`, `Tracker`, `TrainFitnessCallback` |
| `reference/vlm-api.md` | New models, `train()`, Molmo2 `{label}` template [M18, M19] |
| `reference/kernels.md` | Triton MSDA, `LIBREYOLO_TRITON_MSDA`, hub-kernels hint [P03, P04, C15] |
| `reference/checkpoint-schema.md` | `fitness_source`, `letterbox_pad`, input profile, keypoint metadata |
| `reference/dataset-formats.md` | `kpt_names`, event histograms, LeRobot, albedo datasets |
| `reference/settings.md` | New environment variables |
| `reference/export-matrix.md` | Generated; confirm it renders the new families after Phase 1 |
| `cli/train.md`, `cli/val.md`, `cli/export.md`, `cli/predict.md`, `cli/quantize.md` | New flags: classes, visualize, plot_samples, int8, mask/trimap, calibration algorithms. Defaults read from the CLI source |

### 4C. Release-level pages

- `start/changelog.md`: the 1.6.0 entry, adapted from
  `~/release-1.6.0/CHANGELOG-1.6.0.md` to the page's existing format. Drop the
  model-diagram skill entry (A90); it is agent tooling, not a library feature.
- `start/upgrade.md` / `migrate.md`: a "1.5.0 to 1.6.0" section built from
  Breaking changes B01 to B09 and Changed C05, C09 to C11, each with its
  one-line migration.
- `start/install.md`: extras `hf`, `llm`, `fiftyone`, `ground`, `vlm-train`,
  `vla` (Python 3.12 or later), `marigold`, `molmo2` (conflicts with the newer
  Transformers stack), new core dependency `cloudpickle`, `onnxruntime` floor
  [D01 to D05].
- `start/versions.md`: see Phase 4.
- `start/weights.md`, `start/licensing.md`: new mirrors and restrictive
  upstream licenses [S03]. No dataset-derived caveats.
- `start/faq.md`, `start/troubleshooting.md`: only where a new error message
  needs an answer (unavailable weight spellings [F06], persistent-worker
  rejection [C14, B09]).

## 5. Progress checklist

Update this list in the same commit as the work. It is the resume point.

- [x] Phase 1: registry rebuilt against dev SHA `4da12d005eb41a8e694a86124c97ea5faaee9b7c`; 110 lineages / 117 families / 20 tasks; upstream JSON for all new lineages. 14 new BibTeX records copied verbatim and cross-checked, plus two reused verified citations for the grounding adapters. Citation sections omitted for unverified entries; V-JEPA 2 author mismatch is recorded for follow-up.
  - Rebuild command: `sh scripts/build-registry/rebuild.sh <detached-library-checkout> <vision-analysis/generated/verified-results.v1.json>`.
  - Existing capability changes: YOLO-NAS (OBB), FeyNobg and YOLOv2 (exports), LFM2-VL and RT-DETR (sizes); refreshed checkpoint inventories across 40 families.
  - Hub collection checks the matching asset, not just repository existence. U-Net has no verified hosted asset. PP-YOLOE and DEKR use source CDN downloads.
  - Licensing correction: ViTMatte publisher card declares Apache-2.0; dataset-derived caveats in the evidence are not copied.
  - Local Turbopack dev hits the known symlink error too; use `npm run dev -- --webpack`.
- [x] Phase 2: 30 new English pages and 12 stub-to-full pages (42 / 42), including the two existing grounding stubs. All 12 existing Japanese twins updated. Custom tracking fits the existing task page; VLM tuning has a separate workflow because its trainer and checkpoint contract differ from detector LoRA.
  - Local HTTP render checks cover all 42 pages. PP-LiteSeg export/checkpoint tables checked at 390 px in dark mode; table and code containers scroll.
  - Nav, sitemap and llms discovery wired in this phase so the added routes are indexed in the same change. Full plumbing audit remains Phase 5.
- [ ] Phase 3: existing English pages and every existing locale twin committed (61 / 66). All structural and freshness checks pass.
- [ ] Phase 4: 8 English release-level pages drafted; 103 changelog entries covered after excluding A90. Versions keeps the 1.5.0 TODO(owner), no archive route. Existing twins in progress.
- [x] Phase 5: nav, sitemap, llms.txt and generated indexes verified. Task count is generated (20); added labels for the three tasks and guarded missing artwork, without changing homepage layout.
- [x] Phase 6: 936 existing locale twins updated (Japanese 84; each other locale 71). All 13 locale validators pass, and `sync-check` passes 2,275 / 2,275. Final `--stamp` changes zero already-current hashes. Brand-new pages remain English-only.
- [x] Initial Vercel preview build passes (4,189 generated routes): https://libreyolo-website-gkdbjoexh-xubanceccons-projects.vercel.app . Its production sitemap has 3,040 unique canonical URLs and excludes untranslated fallbacks. A final preview will follow the complete translation set.
- [ ] PR description: done, not done, `TODO(1.6.0)` list, changelog corrections, owner decisions

## 6. Done means

- Every changelog entry except internal ones (A90) is documented on at least
  one page, or listed in the PR description as deliberately skipped with a
  reason.
- `sync-check.mjs` and `validate.mjs` pass for all 13 locales.
- A Vercel preview build succeeds and a sample of new pages renders their
  blocks.
- The PR stays a **draft**. The agent does not merge, does not mark it ready,
  and does not deploy to production. The owner decides after the library tag.

## 7. Open questions for the owner (put them in the PR description)

1. Freeze the 1.5.0 v2 tree at `/docs/v1.5.0`, or keep only the unversioned
   tree? (Phase 4)
2. One `librevla` page, or a page per policy?
3. Translate the brand-new pages in this PR, or in a follow-up?

## 8. Execution notes and release handoff

- Release catch-up is scripted in `scripts/build-registry/release-catch-up.sh`. It reads the owner checkout through git and uses a temporary detached worktree; it does not deploy.
- No unmerged-library-PR `TODO(1.6.0)` markers remain at the pinned dev SHA. The v1.6.0 tag, release date and delta audit remain Phase 7.
- Initial Vercel authorization failure was resolved by relinking this worktree to the authenticated `xubanceccons-projects/libreyolo-website` project. No production deployment occurred.
- Checkpoint inventory verifies matching `.pt` assets, while hosted snapshot status checks the actual repository files separately. Runtime extras and inherited train/validation capabilities are collected statically without importing optional ML runtimes.
- Citation follow-up: V-JEPA 2 upstream README and arXiv author records disagree (Mahmoud/Mido Assran and split Mojtaba Komeili metadata). No assembled or silently corrected BibTeX is published.

- Final QA: 66 frontmatter Python snippets on new/expanded pages pass syntax checks; pinned-source CPU payload smoke passes for Actions, AlbedoMap, Boxes3D, depth encodings and restoration saving. No model training, GPU benchmarking or complete heavy-model inference sweep was run.
- Mobile QA found and fixed export-table screen-reader text escaping its scroll container; the page now stays within a 390 px viewport in dark mode. Mostly-empty checkpoint input-size columns and unknown size metadata are omitted.
- Additional source corrections found during review: MiDaS needs the `midas` extra, PE needs `clip` tokenizer dependencies, and automatic 3D checkpoint acquisition needs `hf`. ConvNeXt V1 no longer claims V2 is excluded.

- Final render audit passes 42 / 42 new or expanded English pages, including each declared checkpoint, export, licensing and citation block plus canonical URL. All 13 locale validators pass (174 each; Japanese 187).
