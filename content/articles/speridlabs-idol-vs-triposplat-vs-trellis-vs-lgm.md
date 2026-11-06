---
title: "Speridlabs Idol vs TripoSplat vs TRELLIS vs LGM: which image-to-3D model to use"
description: "Four open image-to-3D models compared on one benchmark. Speridlabs Idol leads on 6 of 7 quality metrics and is the fastest of the diffusion models. Numbers, trade-offs and code for all of them."
date: 2026-11-06
author: Xuban
tags: [LibreYOLO, speridlabs-idol, triposplat, trellis, lgm, image-to-3d, gaussian-splatting, 3d-generation, benchmark]
faq:
  - q: "Which is the best open image-to-3D model: Idol, TripoSplat, TRELLIS or LGM?"
    a: "On Speridlabs' benchmark, Speridlabs Idol has the best score on 6 of the 7 quality metrics: SSIM, LPIPS and IoU at the input view, and DINOv2, CLIP and sharpness distance at seven new views. LGM has the highest PSNR, but its results are visibly blurrier, and it is the fastest at 1.1 s per object. Among the diffusion models, Idol is also the fastest, at 6.8 s on an H100."
  - q: "How is Speridlabs Idol different from TripoSplat?"
    a: "Idol builds on TripoSplat. Both output up to 262,144 Gaussians by default at 50 steps, and on the same benchmark Idol scores higher on every metric and finishes in 6.8 s instead of 9.9 s on an H100. Idol also goes up to 1,048,576 Gaussians per object."
  - q: "Can I run Idol, TripoSplat and TRELLIS from the same Python API?"
    a: "Yes. In LibreYOLO all three load through LibreGen with the same call: LibreGen(\"speridlabs-idol\"), LibreGen(\"triposplat\") or LibreGen(\"trellis\"), then model(\"chair.jpg\").save(\"chair.ply\"). Idol ships in LibreYOLO 1.7.0 and later; TripoSplat and TRELLIS are on the dev branch and land in the next release."
  - q: "Are these models free for commercial use?"
    a: "The code and weights of Speridlabs Idol are Apache-2.0, and TripoSplat, TRELLIS and LGM are MIT. Idol and TripoSplat also read images through a DINOv3 encoder under Meta's DINOv3 License, which has its own end-use restrictions, so read it before you ship a product."
---

<img src="/articles/speridlabs-idol-vs-triposplat-vs-trellis-vs-lgm/elkano-boat.webp" alt="A galleon from one video frame, generated as a 3D object by Speridlabs Idol, TripoSplat and TRELLIS, each shown from four angles" width="1910" height="720" loading="lazy" style="display:block;margin:1.5rem auto 0.5rem;width:100%" />

*One frame of a galleon in, three 3D objects out: Speridlabs Idol, TripoSplat and TRELLIS, each turned to four angles. Frame from the [elkano.dev](https://elkano.dev) video.*

You have one photo of an object and you want the whole object in 3D, back and sides included. Four open models do that and output 3D Gaussian splats: [Speridlabs Idol](https://speridlabs.com/research/idol), [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat), [TRELLIS](https://github.com/microsoft/TRELLIS) and [LGM](https://github.com/3DTopia/LGM).

Short answer: use **Speridlabs Idol**. It has the best score on 6 of the 7 quality metrics in the benchmark below, and it is the fastest of the three diffusion models. LGM is the one to pick only if you need an object in about a second and can live with a blurry back side.

## The four models

| Model | By | Released | License | Output |
| --- | --- | --- | --- | --- |
| [Speridlabs Idol](https://github.com/speridlabs/idol-sdk) | [Speridlabs](https://speridlabs.com) | 2026 | Apache-2.0 | 32,768 to 1,048,576 Gaussians |
| [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) | VAST / Tripo AI | 2026 | MIT | up to 262,144 Gaussians |
| [TRELLIS](https://github.com/microsoft/TRELLIS) | Microsoft | 2024 | MIT | Gaussians, meshes, radiance fields |
| [LGM](https://github.com/3DTopia/LGM) | 3DTopia | 2024 | MIT | 65,536 Gaussians by default |

**Speridlabs Idol** is a 0.66B-parameter rectified-flow transformer. It reads DINOv3 and FLUX.2 VAE features of the input, denoises an object latent together with the camera of the photo, and decodes it through an octree into Gaussians. It builds on TripoSplat and runs in under 8 GB of GPU memory.

**TripoSplat** is Tripo AI's single-image-to-Gaussians model, and the base Idol starts from.

**TRELLIS** is Microsoft's structured 3D latent model. It runs two flow stages, sparse structure then latents, and decodes to several 3D formats.

**LGM** (Large Multi-View Gaussian Model) first generates four views of the object with a multi-view diffusion model, then predicts Gaussians from them in one feed-forward pass.

## The benchmark

Speridlabs scored all four models on the same set of images, one H100 per model. Three things are measured: how well the object matches the photo from the photo's own viewpoint, how it holds up from seven new viewpoints, and how long a request takes. From the [Idol SDK benchmark](https://github.com/speridlabs/idol-sdk#benchmark) and the [Idol research page](https://speridlabs.com/research/idol):

| Model | Setting | PSNR ↑ | SSIM ↑ | LPIPS ↓ | IoU ↑ | DINOv2 ↑ | CLIP ↑ | Sharpness dist. ↓ | Request time ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Speridlabs Idol** | 50 steps, 262,144 Gaussians | 12.25 | **0.726** | **0.213** | **0.862** | **0.781** | **0.885** | **0.81** | 6.8 s |
| TripoSplat | 50 steps, 262,144 Gaussians | 11.76 | 0.712 | 0.225 | 0.860 | 0.743 | 0.868 | 1.03 | 9.9 s |
| TRELLIS | Paper setting, 50 + 50 steps | 11.89 | 0.716 | 0.235 | 0.845 | 0.754 | 0.868 | 0.97 | 7.1 s |
| LGM | Default, 65,536 Gaussians | **12.46**\* | 0.722 | 0.237 | 0.834 | 0.575 | 0.784 | 1.22 | **1.1 s** |

\* LGM has the highest PSNR despite visibly blurry results, because PSNR averages squared pixel error. Bold is the best in each column.

What the columns mean:

- **PSNR, SSIM, LPIPS** compare a render of the object from the input camera with the input photo. LPIPS is the one closest to what a person sees.
- **IoU** is how well the object's silhouette from that camera matches the photo's.
- **DINOv2 and CLIP** compare seven renders from new viewpoints with the input photo. A high score means the back and sides still look like the same object.
- **Sharpness distance** is how far the sharpness of those new views is from the input's. Lower means the unseen sides are as crisp as the visible one.

## What the numbers say

**Idol wins where it matters most: the sides the photo never showed.** On the three new-view metrics it leads all three other models. Its DINOv2 score is 0.781 against 0.754 for TRELLIS, the runner-up, and its sharpness distance is 0.81 against 0.97. That is a 16% drop, so the generated back of the object is clearly crisper. Anyone can make the front look right, since the front is in the photo. The back is where image-to-3D models differ, and that is where Idol is best.

**At the input view, Idol is the most faithful too.** Best SSIM, best LPIPS and best IoU. The one column it does not win is PSNR, and the footnote explains why: PSNR rewards blur, and LGM's blur buys it the top PSNR while it comes last on LPIPS.

**Idol against TripoSplat** is the cleanest comparison in the table, because Idol builds on TripoSplat and both run at 50 steps with 262,144 Gaussians. Idol is better on all seven quality metrics and takes 6.8 s instead of 9.9 s, about 31% less.

**Idol against TRELLIS:** TRELLIS is the best known of the four, with over 14,000 GitHub stars. Idol beats it on all seven quality metrics, PSNR included, and is still slightly faster, 6.8 s against 7.1 s for TRELLIS's two 50-step stages.

**Idol against LGM:** LGM is six times faster, and it is the only reason to choose it. Its new views score far below the rest (DINOv2 0.575 against 0.781 for Idol), so the object often stops looking like itself once you turn it around.

One more thing in Idol's favor: **level of detail is a setting**. Idol goes from 32,768 Gaussians (about 0.4 MB) to 1,048,576 (about 12 MB) from the same model, so the same pipeline serves a web preview and a hero asset. TripoSplat tops out at 262,144 and LGM defaults to 65,536.

A fair caveat: this is Speridlabs' benchmark of their own model. The settings for every model are listed in the table, and the code and weights of all four are public, so anyone can rerun it.

## Side by side

Numbers are one thing. Here is what the objects look like. We ran Speridlabs Idol, TripoSplat and TRELLIS through LibreYOLO on 41 images, photos of real sculptures and buildings plus examples from the Idol demo, with the same seed and 50 steps each, on one RTX 4090. Every row below is one model, turned to four angles around the object. Most of those angles were never in the photo.

We picked the six where the models differ most. On many simple objects (a donut, a mailbox, a power drill) all three do well, and Idol and TripoSplat often look close, since Idol builds on TripoSplat. Where they differ, it is usually Idol that holds up.

<img src="/articles/speridlabs-idol-vs-triposplat-vs-trellis-vs-lgm/puppy.webp" alt="Puppy by Jeff Koons, generated as a 3D object by Speridlabs Idol, TripoSplat and TRELLIS from one photo" width="1910" height="720" loading="lazy" style="display:block;margin:1.5rem auto 0.5rem;width:100%" />

*Puppy by Jeff Koons, Bilbao. Idol keeps the flowers as flowers all the way round; TRELLIS turns them into blotches. Photo: [José Ligero Loarte](https://commons.wikimedia.org/wiki/File:Puppy_de_Jeff_Koons_(2)_--_2021_--_Bilbao,_Espa%C3%B1a.jpg), CC BY-SA 4.0. This figure is shared under CC BY-SA 4.0.*

<img src="/articles/speridlabs-idol-vs-triposplat-vs-trellis-vs-lgm/loyola.webp" alt="The Sanctuary of Loyola, generated as a 3D object by Speridlabs Idol, TripoSplat and TRELLIS from one photo" width="1910" height="720" loading="lazy" style="display:block;margin:1.5rem auto 0.5rem;width:100%" />

*The Sanctuary of Loyola, Azpeitia. Idol and TripoSplat keep the façade and the stairs. TRELLIS makes up a round building with no entrance. Photo: [Jose Luis Filpo Cabana](https://commons.wikimedia.org/wiki/File:Santuario_de_Loyola._Bas%C3%ADlica.jpg), CC BY 3.0.*

<img src="/articles/speridlabs-idol-vs-triposplat-vs-trellis-vs-lgm/fur-egg-chair.webp" alt="An egg chair generated as a 3D object by Speridlabs Idol, TripoSplat and TRELLIS from one image" width="1910" height="720" loading="lazy" style="display:block;margin:1.5rem auto 0.5rem;width:100%" />

*An egg chair. TRELLIS changes the orange fabric to mustard; Idol keeps the color and the thin steel base.*

<img src="/articles/speridlabs-idol-vs-triposplat-vs-trellis-vs-lgm/fig-panda.webp" alt="A toy panda generated as a 3D object by Speridlabs Idol, TripoSplat and TRELLIS from one image" width="1910" height="720" loading="lazy" style="display:block;margin:1.5rem auto 0.5rem;width:100%" />

*A toy panda. Turned around, Idol keeps the back white and clean; TRELLIS gives it a purple sheen.*

<img src="/articles/speridlabs-idol-vs-triposplat-vs-trellis-vs-lgm/nb-microscope.webp" alt="A brass microscope generated as a 3D object by Speridlabs Idol, TripoSplat and TRELLIS from one image" width="1910" height="720" loading="lazy" style="display:block;margin:1.5rem auto 0.5rem;width:100%" />

*A brass microscope. TripoSplat leaves a grey box of stray Gaussians around the object; Idol does not.*

The four synthetic images come from the [Idol SDK demo](https://github.com/speridlabs/idol-sdk/tree/main/demo/examples) (Apache-2.0). Want to compare on your own images? The code is in [Run them yourself](#run-them-yourself).

## Speed on your GPU

Speridlabs' own numbers for the Idol SDK, at 50 steps and 262,144 Gaussians, median over 8 images:

| GPU | VRAM | Time per object |
| --- | --- | --- |
| NVIDIA B300 | 269 GB | 3.4 s |
| NVIDIA H100 | 80 GB | 6.7 s |
| NVIDIA RTX 5090 | 32 GB | 14.1 s |
| NVIDIA RTX 4090 | 24 GB | 19.1 s |
| NVIDIA RTX 3080 | 10 GB | 46.2 s |

A 10 GB consumer card runs it. The full ten-GPU table is in the [Idol SDK README](https://github.com/speridlabs/idol-sdk#across-gpus).

## Run them yourself

The official way to run Idol is the [Speridlabs Idol SDK](https://github.com/speridlabs/idol-sdk):

```bash
git clone https://github.com/speridlabs/idol-sdk && cd idol-sdk
pip install -e .
python -m idol.sample --image photo.jpg --mode real
```

Its DINOv3 and FLUX.2 encoders sit in gated Hugging Face repositories, so the first run needs two access requests and an `hf auth login`.

In LibreYOLO, Idol needs no login and is one install and three lines ([full docs page](/docs/models/speridlabs-idol)):

```bash
pip install "libreyolo[gen]"
```

```python
from libreyolo import LibreGen

model = LibreGen("speridlabs-idol")      # weights download on first use
model("chair.jpg").save("chair.ply")     # any photo or render of one object
```

The port matches the Speridlabs SDK bit for bit on CUDA, and Idol's weights download unchanged from [speridlabs/idol](https://huggingface.co/speridlabs/idol), so every download counts for Speridlabs.

To compare the models on your own images, TripoSplat and TRELLIS sit behind the same call. They are on LibreYOLO's `dev` branch and ship in the next release:

```bash
pip install "libreyolo[gen] @ git+https://github.com/LibreYOLO/libreyolo@dev"
```

```python
from libreyolo import LibreGen

for name in ["speridlabs-idol", "triposplat", "trellis"]:
    model = LibreGen(name)
    model("chair.jpg", seed=0).save(f"chair-{name}.ply")
```

Or from the shell:

```bash
libreyolo generate source=chair.jpg model=speridlabs-idol
libreyolo generate source=chair.jpg model=triposplat
libreyolo generate source=chair.jpg model=trellis
```

Every `.ply` uses the standard 3D Gaussian Splatting layout, so any splat viewer opens all three side by side. LGM is not in LibreYOLO; its own code is at [3DTopia/LGM](https://github.com/3DTopia/LGM).

## Which one to pick

- **You want the best object from one photo:** Speridlabs Idol.
- **You want it fast on a datacenter GPU:** Speridlabs Idol, 3.4 s on a B300 and 6.7 s on an H100.
- **You need several sizes of the same object:** Speridlabs Idol, 32,768 to 1,048,576 Gaussians.
- **You need a mesh rather than splats:** TRELLIS, which also decodes to meshes.
- **You need something in about a second and quality can drop:** LGM.

For most image-to-3D work in 2026, that means Speridlabs Idol.

## Licensing

- **Speridlabs Idol:** Apache-2.0, code and weights.
- **TripoSplat, TRELLIS, LGM:** MIT.
- **DINOv3 encoder** (used by Idol and TripoSplat): Meta's DINOv3 License, a custom agreement with end-use restrictions. Read it before you ship. [YOLO licenses explained](/articles/yolo-licenses-explained) covers how model licenses stack.
- **LibreYOLO:** MIT.

## Links

Speridlabs Idol:

- Speridlabs: [speridlabs.com](https://speridlabs.com)
- Research page and benchmark: [speridlabs.com/research/idol](https://speridlabs.com/research/idol)
- SDK: [github.com/speridlabs/idol-sdk](https://github.com/speridlabs/idol-sdk)
- Weights: [huggingface.co/speridlabs/idol](https://huggingface.co/speridlabs/idol)
- Demo: [huggingface.co/spaces/speridlabs/idol](https://huggingface.co/spaces/speridlabs/idol)

The others: [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat), [TRELLIS](https://github.com/microsoft/TRELLIS), [LGM](https://github.com/3DTopia/LGM).

In LibreYOLO:

- [Speridlabs Idol in LibreYOLO: one image to a 3D object](/articles/speridlabs-idol-image-to-3d), the launch article with the Basque Country gallery
- [Speridlabs Idol docs page](/docs/models/speridlabs-idol)
- [3D generation task](/docs/tasks/3d-generation): the payload, seeds, `.ply` and `.splat`
- [libreyolo generate CLI](/docs/cli/generate)
- [Depth Anything V2 in LibreYOLO](/articles/simplest-way-to-run-depth-anything-v2), another single-image model with the same call shape

Star it on GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Try Idol: [huggingface.co/spaces/speridlabs/idol](https://huggingface.co/spaces/speridlabs/idol)

## Play with the objects

The six objects from the side-by-side figures, from all three models: 18 objects you can turn around. Pick a photo, then a model, and drag to look at the back.

<idol-vs-gallery></idol-vs-gallery>
