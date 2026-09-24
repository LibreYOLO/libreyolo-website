---
title: "Every YOLO License, Explained: v1 to YOLO26 (2026 Guide)"
description: "The complete YOLO license map for 2026: YOLOv1 through YOLOv13, YOLO26, YOLO-World and YOLOE, original repos and permissive rewrites, with paper and GitHub links, and what you can actually ship in a commercial product."
date: 2026-07-11
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, gpl, apache, mit-license, yolov9, yolov10, yolo11, yolov12, yolov13, yolo26]
faq:
  - q: "Is YOLOv9 free for commercial use?"
    a: "It depends which repository you use. The paper repository (WongKinYiu/yolov9) is GPL-3.0. The same lab also ships a separate MIT-licensed implementation of YOLOv9 and YOLOv7 at MultimediaTechLab/YOLO, written after commercial users asked for a permissive option. It is a rewrite, not a relicense of the GPL files. The code license is clear; the pretrained weights carry no separate license statement in either repo, so treat weight provenance as its own question."
  - q: "Which YOLO versions are free for closed-source commercial use?"
    a: "Through at least one permissive repository: YOLOv1 to YOLOv4 (original Darknet, public domain), YOLOv7 and YOLOv9 (the lab's own MIT rewrite), YOLOX and PP-YOLOE (Apache-2.0), and the MIT reimplementations in LibreYOLO. YOLOv5, v6, v8, v10, YOLO11, v12, v13, YOLO26, YOLO-World and YOLOE have no permissive implementation as of July 2026. Check the current LICENSE file yourself before you rely on this: licenses change, and this is general information, not legal advice."
  - q: "What license is YOLOv10 under?"
    a: "AGPL-3.0. YOLOv10 is an academic release from Tsinghua University, but its code is built on the Ultralytics codebase, so it inherits AGPL-3.0. There is no permissive reimplementation."
  - q: "What license are YOLOv12 and YOLOv13 under?"
    a: "Both are AGPL-3.0, confirmed in their LICENSE files. Like YOLOv10, they are academic papers whose reference code is built on the Ultralytics repository, so the AGPL carries over regardless of who wrote the paper. Third-party pages that list YOLOv13 as Apache-2.0 are wrong."
  - q: "When did Ultralytics switch YOLO to AGPL-3.0?"
    a: "On 14 April 2023, not in 2022 as is widely repeated. The LICENSE file in ultralytics/yolov5 has only ever been changed three times, and the AGPL commit (34cf749, PR #11359) is dated 2023-04-14. The last 2022 release, YOLOv5 v7.0, still shipped GPL-3.0. The ultralytics/ultralytics repo was relicensed the same day. YOLOv8 therefore launched in January 2023 under GPL-3.0: PyPI releases 8.0.0 through 8.0.76 declare GPL-3.0, and 8.0.80 (16 April 2023) is the first to declare AGPL-3.0."
  - q: "Is YOLO26 free for commercial use?"
    a: "Not for closed-source products. YOLO26 ships under AGPL-3.0 with a paid Enterprise License as the alternative, the same terms as YOLOv8 and YOLO11."
  - q: "Is the original Darknet YOLO really public domain?"
    a: "Yes. Joseph Redmon's Darknet LICENSE states 'Darknet is public domain. Do whatever you want with it.' AlexeyAB's YOLOv4 fork carries the same public domain text. The catch is that the PyTorch ports people actually use carry their own licenses, from Apache-2.0 to AGPL-3.0 to no license at all."
  - q: "Can I use the YOLO-NAS pretrained weights commercially?"
    a: "No. The super-gradients code is Apache-2.0, but the official YOLO-NAS weights ship under a separate license that says you 'may not use the Software for any commercial use, including in connection with any models used in a production environment.'"
  - q: "Do neural network weights inherit the license of the training code?"
    a: "It is unsettled. Ultralytics states that AGPL-3.0 covers 'the training code and the models produced by that training code,' and as the copyright holder they set the terms for what they publish. Whether a court would agree that weights you train yourself are a derivative work of the training code has never been tested. Treat published AGPL weights as encumbered, and be careful about loading them into a permissive reimplementation."
  - q: "Why can different implementations of the same YOLO have different licenses?"
    a: "Copyright protects expression, not ideas (17 U.S.C. 102(b)). An architecture described in a paper can be independently implemented and licensed however its author chooses. What you cannot do is copy GPL or AGPL source files and relicense them. Note that this is a copyright argument only: independent implementation is not a defense against patents."
---

Every year there are more YOLOs, and every year the licensing question gets asked again, usually five minutes before a product decision. The confusing part is that "YOLO" is not one project. It is a brand name shared by twenty-odd model families from different authors, and here is the detail most license guides miss: **the license belongs to a repository, not to a model.** The same YOLO version can exist as a GPL repo and an MIT repo at the same time, from the same authors. YOLOv9 does, and it changes the whole decision.

This guide maps the landscape repository by repository, with links to each paper and codebase, current as of July 2026. Every license below was checked against the actual LICENSE file in the actual repository, because a surprising amount of what is written about YOLO licensing (including in the guides that rank highest for it) is wrong.

If you only want the Ultralytics AGPL question answered, we cover that in depth in [Is YOLO Free for Commercial Use?](/articles/yolo-commercial-license). This one is the full map.

**Two disclosures before we start.** First, we maintain LibreYOLO, an MIT-licensed library that competes with several of the projects below, and the last section of this article is about it. That is a reason to check our work, not to take it on faith, which is why every license claim here links to the primary source. Second, this article is general information about published license texts and public statements, current as of the date above. It is not legal advice and is not a substitute for a lawyer in your jurisdiction. Licenses and maintainers' positions change. Read the current LICENSE file of any repository before you make a shipping decision.

## The license table

"Permissive route" means: a repository implementing this model that you can use in a closed-source commercial product, without open-sourcing your application and without paying for a license.

| Model | Year | Original code | Permissive route | Paper |
| --- | --- | --- | --- | --- |
| YOLOv1 | 2015 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (public domain) | The original itself | [arXiv](https://arxiv.org/abs/1506.02640) |
| YOLOv2 | 2016 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (public domain) | The original itself | [arXiv](https://arxiv.org/abs/1612.08242) |
| YOLOv3 | 2018 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (public domain) | The original; beware the AGPL PyTorch port | [arXiv](https://arxiv.org/abs/1804.02767) |
| YOLOv4 | 2020 | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) (public domain) | The original; [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) (Apache-2.0) | [arXiv](https://arxiv.org/abs/2004.10934) |
| YOLOv5 | 2020 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) (AGPL-3.0) | None | no paper |
| YOLOv6 | 2022 | [meituan/YOLOv6](https://github.com/meituan/YOLOv6) (GPL-3.0) | None | [arXiv](https://arxiv.org/abs/2209.02976) |
| YOLOv7 | 2022 | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, same lab) | [arXiv](https://arxiv.org/abs/2207.02696) |
| YOLOv8 | 2023 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0; GPL-3.0 at launch, see below) | None (the Keras port was dropped, see below) | no paper |
| YOLOv9 | 2024 | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, same lab) | [arXiv](https://arxiv.org/abs/2402.13616) |
| YOLOv10 | 2024 | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) (AGPL-3.0) | None | [arXiv](https://arxiv.org/abs/2405.14458) |
| YOLO11 | 2024 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | None | no paper |
| YOLOv12 | 2025 | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12) (AGPL-3.0) | None | [arXiv](https://arxiv.org/abs/2502.12524) |
| YOLOv13 | 2025 | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13) (AGPL-3.0) | None | [arXiv](https://arxiv.org/abs/2506.17733) |
| YOLO26 | 2026 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | None | [arXiv](https://arxiv.org/abs/2606.03748) |
| YOLOX | 2021 | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) | The original itself | [arXiv](https://arxiv.org/abs/2107.08430) |
| PP-YOLOE | 2022 | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) (Apache-2.0) | The original, from PaddleDetection (not PaddleYOLO) | [arXiv](https://arxiv.org/abs/2203.16250) |
| YOLO-NAS | 2023 | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) (Apache-2.0 code, non-commercial weights) | Code yes, official weights no | no paper |
| YOLO-World | 2024 | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) (GPL-3.0) | None | [arXiv](https://arxiv.org/abs/2401.17270) |
| YOLOE | 2025 | [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) (AGPL-3.0) | None | [arXiv](https://arxiv.org/abs/2503.07465) |
| MMYOLO | 2022 | [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo) (GPL-3.0, stalled since 2024) | None | no paper |

Read it column by column and one thing becomes obvious: the version number tells you nothing about the license. The repository does.

## Same model, two licenses: the YOLOv9 case

YOLOv9 is the cleanest proof that "what license is YOLOvN?" is the wrong question, and the timeline is worth telling precisely because it is the only time in YOLO history that community pressure actually changed an outcome.

- **18 February 2024:** [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) goes public alongside the [paper](https://arxiv.org/abs/2402.13616).
- **22 February 2024:** a user opens issue #10 asking what the license is. Chien-Yao Wang (WongKinYiu) replies "I think it should be GPL3," and adds the GPL-3.0 file four days later.
- **26 February 2024:** a second user opens [issue #82, "An Apache/MIT rewrite"](https://github.com/WongKinYiu/yolov9/issues/82). Over the next two and a half weeks, a dozen commercial users pile in.
- **14 March 2024:** Wang posts in that thread: *"Okay I create a new repo for mit rewrite... I think I can handle most of implementation of architectures and loss functions. And need someone to give great help about dataloader and ddp training."*

That repo, first named `yolov9mit`, is today **[MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)**: MIT-licensed, copyright "Kin-Yiu, Wong and Hao-Tang, Tsui," describing itself as *"the official implementation of YOLOv7 and YOLOv9, YOLO-RD."* The bulk of the rewrite was done not by the volunteers in the thread but by Hao-Tang Tsui, Wang's labmate, who has written roughly 90 percent of its commits.

So YOLOv9 is simultaneously "not shippable" and "shippable" depending on which repo you clone. Same architecture, same lab, two licenses.

**Before you bet a product on it, three caveats the repo will not tell you on the front page:**

1. **It is still maturing.** An [open issue](https://github.com/MultimediaTechLab/YOLO/issues/231) from February 2026 reports that the shipped checkpoints do not reproduce the paper's COCO numbers and carry noticeably more parameters than the paper claims. Segmentation and keypoint training are [still unimplemented](https://github.com/MultimediaTechLab/YOLO/issues/232). Nothing has merged to main since the v1.0 tag in December 2025.
2. **The weights carry no separate license statement.** The repo's MIT LICENSE conventionally covers its release assets, and the technical evidence points to checkpoints trained by this project rather than copied from the GPL repo: the files differ in size and parameter count from the GPL repo's checkpoints. But no document states that the .pt files are MIT, and a community [request for a formal audit against AGPL contamination](https://github.com/MultimediaTechLab/YOLO/issues/51) is still open. That is a gap in the paperwork: nobody has claimed "verified clean," and nobody has alleged a problem.
3. **It is a different codebase**, not a drop-in replacement for the GPL repo's scripts.

None of that makes it unusable. It makes it a project to evaluate, not a checkbox to tick.

## Era 1: the public domain years (YOLOv1 to YOLOv4)

Joseph Redmon released [YOLOv1](https://arxiv.org/abs/1506.02640), [YOLOv2](https://arxiv.org/abs/1612.08242) (published as the YOLO9000 paper) and [YOLOv3](https://arxiv.org/abs/1804.02767) inside his Darknet framework, and the [LICENSE file](https://github.com/pjreddie/darknet/blob/master/LICENSE) is famous for being three lines long:

> 0. Darknet is public domain.
> 1. Do whatever you want with it.
> 2. Stop emailing me about it!

That is a genuine public domain dedication. [YOLOv4](https://arxiv.org/abs/2004.10934), maintained by Alexey Bochkovskiy in a [Darknet fork](https://github.com/AlexeyAB/darknet), carries the *same* file, not the Unlicense that several license guides attribute to it. GitHub's own classifier gives up on it and reports "Other," which is worth knowing if your compliance tooling reads that field: a scanner will flag it as unidentified rather than as a clean permissive license, even though the text could hardly be more permissive.

Darknet itself is not quite dead, either. Redmon's repo has been idle since 2022, but AlexeyAB's README now points to [hank-ai/darknet](https://github.com/hank-ai/darknet) as the recommended, actively maintained C/C++ successor.

**The trap is the ports.** Almost nobody trains in Darknet in 2026; people reach for a PyTorch reimplementation, and those carry their own licenses:

- [ultralytics/yolov3](https://github.com/ultralytics/yolov3), the most popular YOLOv3 port, is **AGPL-3.0**. A public domain architecture, redistributed under the strictest common license in this space. The license you get depends entirely on whose port you pip-installed.
- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) is **GPL-3.0**.
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) is **Apache-2.0**, so a permissive route for v4 survives into PyTorch.
- [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4), from the YOLOv4 co-author, has **no LICENSE file at all**. That is worse than copyleft: with no license, no rights are granted, and the default is all rights reserved.

Check the license of the repository you cloned, not the version number on the paper.

## Era 2: GPL-3.0 (YOLOv6, v7, v9)

[YOLOv6](https://github.com/meituan/YOLOv6) (Meituan), [YOLOv7](https://github.com/WongKinYiu/yolov7) and [YOLOv9](https://github.com/WongKinYiu/yolov9) shipped as GPL-3.0.

GPL-3.0 is copyleft: if you distribute a modified version, or combine GPL code with yours into a single program, you must convey the complete corresponding source of that combined work under GPL-3.0. Two boundaries matter. **Mere aggregation** means separate, independent programs merely shipped together are not automatically combined. And the trigger is **distribution**: unlike AGPL, if you never ship the combined work, plain GPL imposes no disclosure duty, which is why some companies keep a GPL model strictly behind their own API.

That path is narrower than it sounds, and it fails in three ways people rarely plan for. The moment the model lands on a customer device or an on-prem install, you are distributing. "Internal only" stops being true when a copy leaves your legal entity: handing a build to a contractor, an outsourced dev team or a separately incorporated affiliate is distribution, even though copying it around inside one company is not. And the shield only holds if *everything* reachable from that API is GPL or more permissive, because a single AGPL component anywhere in the served work pulls the whole thing into Section 13 regardless of what the GPL file itself says.

**YOLOv7 and YOLOv9 have the MIT escape hatch described above.** YOLOv6 does not, and the strongest evidence for that comes from Baidu: PaddleDetection is Apache-2.0, and its maintainers deliberately quarantined YOLOv5, YOLOv6, YOLOv7 and YOLOv8 into a separate GPL-3.0 repo, [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO), stating that those models' code "will not be merged into PaddleDetection." When a company as large as Baidu builds a wall to keep YOLOv6 out of its permissive framework, that tells you what the license is worth.

## Era 3: AGPL-3.0 (YOLOv5, v8, v10, 11, v12, v13, 26, YOLOE)

Ultralytics moved to AGPL-3.0 on **14 April 2023**, not in 2022 as is very widely repeated (including by guides that rank on the first page for this question). Everything since ([YOLOv8](https://github.com/ultralytics/ultralytics), YOLO11 and [YOLO26](https://arxiv.org/abs/2606.03748)) has been AGPL-3.0, with a paid Enterprise License as the alternative. Their [license page](https://ultralytics.com/license) says the Enterprise tier covers "YOLO26, earlier YOLO versions, and any future YOLO models," so the policy is deliberate and forward-looking. Pricing is not public.

That date is checkable, and worth checking, because the popular version of this story is wrong in a way that matters:

- The `LICENSE` file in `ultralytics/yolov5` has been touched exactly three times in its history. The third is commit [`34cf749`](https://github.com/ultralytics/yolov5/commit/34cf749958d2dd3ed1205f6bb07e0f20f6e2372d), "Update LICENSE to AGPL-3.0 (#11359)," dated **2023-04-14**, which replaces `GNU GENERAL PUBLIC LICENSE` with `GNU AFFERO GENERAL PUBLIC LICENSE` across 101 files.
- The last YOLOv5 release of 2022, **v7.0 (22 November 2022), still shipped GPL-3.0**, as do v6.2, v6.1 and v6.0 before it. So do the thousands of forks that stopped syncing before April 2023: pick any of them and GitHub still reports GPL-3.0 today.
- The same day, 41 minutes later, `ultralytics/ultralytics` got [the same treatment](https://github.com/ultralytics/ultralytics/commit/2c6fc0a4443b9cf805ef17b1cfdd71a98693b4d4) (PR #2031).

Which produces the fact almost nobody knows: **YOLOv8 did not launch under AGPL.** It shipped in January 2023 under **GPL-3.0** and was relicensed three months later. The PyPI record is unambiguous: `ultralytics` 8.0.0 (10 January 2023) through 8.0.76 (13 April 2023) all declare `GPL-3.0`. Version 8.0.80, uploaded on 16 April 2023, is the first to declare `AGPL-3.0`.

This is not a loophole. A licence granted is not retroactively revoked, so those old versions remain available on the terms they were published under. But they are three years stale, unmaintained and unpatched, GPL-3.0 is still copyleft (you have swapped a network clause for a distribution clause, not escaped copyleft), and Ultralytics' commercial position is broader than the licence text anyway. The useful takeaway is narrower and more general: **a licence attaches to the version you received, not to the project's name.** Pin your dependencies and record what the licence said on the day you took the copy, because the project can change it tomorrow and your compliance story depends on which one you actually got.

AGPL-3.0 is GPL plus Section 13, the network clause: *if you modify the Program*, users who interact with your modified version over a network must be offered its source. That closes the SaaS path plain GPL leaves open.

Note the qualifier, because most articles drop it. AGPL Section 0 defines "modify" as copying from or adapting the work *in a fashion requiring copyright permission*. Running unmodified training code over your own dataset is executing the program, not adapting its source, in the same way that compiling your code with an unmodified GPL compiler does not modify the compiler. So training alone is not automatically the trigger, and anyone who tells you it is has skipped the definition.

What does put you inside the license's scope is combining or embedding the package into your own codebase so that the two ship or serve as one program. The FSF's own reading is that linking a GPL or AGPL work with other modules makes a combined work; whether a court would extend that as far as a Python import has never been tested. Note also that Ultralytics' own [commercial terms](https://ultralytics.com/license) require an Enterprise License for SaaS platforms, APIs and cloud systems that use YOLO behind the scenes, so the vendor's position is broader than the bare license text. Treat integration into a product you serve or ship as the risk; do not assume that training in isolation is.

On YOLO26 specifically, the dates get muddled everywhere, so: it was **previewed in September 2025** at YOLO Vision, **actually released in January 2026** (the `ultralytics` 8.4.0 release, whose notes say "Ultralytics YOLO26 has arrived"), and the **paper landed in June 2026**. It lives in the main `ultralytics/ultralytics` repo; `ultralytics/yolo26` is only a landing page.

**The part that surprises people:** the academic YOLOs of the last two years are AGPL too, and not because Ultralytics wrote them.

- [YOLOv10](https://arxiv.org/abs/2405.14458) is from Tsinghua University. Its README says *"The code base is built with ultralytics."*
- [YOLOv12](https://arxiv.org/abs/2502.12524): *"The code is based on ultralytics."*
- [YOLOv13](https://arxiv.org/abs/2506.17733): *"The code is based on Ultralytics."*
- [YOLOE](https://arxiv.org/abs/2503.07465), the open-vocabulary "see anything" model from the YOLOv10 group, is likewise AGPL-3.0 and built on Ultralytics.

Copyleft is inherited: a derivative of AGPL code has to be conveyed under AGPL, so the moment one of these forks is shipped or served, the license comes with it. (Modify it and keep it strictly to yourself and no obligation attaches, which is why research use is unaffected.) The research is independent; the license is not. Since 2024, publishing "the next YOLO" as an Ultralytics derivative has become the default workflow, which means the AGPL now propagates automatically up the version numbers.

If you see a page claiming YOLOv13 is Apache-2.0, it is wrong. The LICENSE file, the GitHub API and the model card all say AGPL-3.0.

### The YOLOv8 permissive route that no longer exists

Many guides still online, including an earlier draft of this one, point at KerasCV's Apache-2.0 YOLOv8 as the clean escape from AGPL YOLOv8. **You cannot rely on it today.** Here is the public record:

- In the KerasCV discussion [Clarifications regarding YOLOv8 licensing](https://github.com/keras-team/keras-cv/discussions/2032), a contributor wrote that "many parts are derived form ultralytics." No maintainer answered the thread, so the question of how independent the implementation was has never been settled either way in public.
- The KerasHub issue [add YOLOV8](https://github.com/keras-team/keras-hub/issues/1760) was closed in July 2025 by a Keras maintainer with: "Hey all!! Because of license issues, we have decided to drop this." Two people then asked what that meant for existing commercial users of the KerasCV version. Neither got a reply.
- KerasCV is now archived, and KerasHub, its maintained successor, **does not ship YOLOv8**.

We do not know whether those two things are connected, and we are not asserting that anything improper happened. That is not a foundation to build a product on.

## The permissive outliers: YOLOX, PP-YOLOE, YOLO-NAS

Not every YOLO joined the copyleft train.

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** (Megvii, 2021) is Apache-2.0 with no carve-outs, code and weights. It is still the cleanest classic YOLO for commercial use, though the repository has had no commits since mid-2025, and a [question about bundling the pretrained weights in a commercial app](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865) has been open since March 2026 with no reply. The license text is unambiguous. There is just nobody currently around to answer questions about it. We wrote about [running YOLOX with LibreYOLO](/articles/yolox-with-libreyolo).
- **[PP-YOLOE](https://arxiv.org/abs/2203.16250)** (Baidu) is Apache-2.0 **inside PaddleDetection**. Take it from there, not from PaddleYOLO, which contains a near-identical config directory and is GPL-3.0. Same model, same company, two repos, two licenses. It is the YOLOv9 situation in reverse.
- **[YOLO-NAS](https://github.com/Deci-AI/super-gradients)** (Deci, 2023) is the one that catches people. The code is Apache-2.0; the official weights are not. Their [separate license](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md) states you *"may not use the Software for any commercial use, including in connection with any models used in a production environment."* Since NVIDIA acquired Deci in 2024 we have found no new feature work in the repository, and the original weight URLs no longer resolve (the checkpoints now live at a different host). A workaround sometimes suggested, that training the architecture from scratch leaves you Apache-clean, is plausible but has not been confirmed by Deci or NVIDIA, and it sits awkwardly against license text that reaches "any components comprising the model." More on YOLO-NAS [here](/articles/yolo-nas-with-libreyolo).
- **[MMYOLO](https://github.com/open-mmlab/mmyolo)** is sometimes listed as Apache-2.0 in older guides. It is not: MMYOLO is GPL-3.0 (its sibling MMDetection is the Apache one), and it has had no commits since July 2024.
- **[YOLO-World](https://github.com/AILab-CVC/YOLO-World)** (Tencent) is **GPL-3.0**, not AGPL and not permissive. Worth stating because open-vocabulary YOLOs are often assumed to be permissive by association with the CLIP-style ecosystem they borrow from.

Those cases cover the three most common licensing mistakes in this space: assuming the weights follow the code license, assuming everything from one organization shares one license, and assuming a model's license from its name.

## Licenses cover code, not ideas (but watch the patents)

One principle is the legal basis for every permissive route above: **copyright protects expression, not ideas.** That is not a slogan, it is black-letter law on both sides of the Atlantic.

In the US it is [17 U.S.C. 102(b)](https://www.law.cornell.edu/uscode/text/17/102), which excludes any "idea, procedure, process, system, method of operation, concept, principle, or discovery" from copyright protection, going back to *Baker v. Selden* (1879). In the EU, which is the law that governs us and probably a good number of readers, Article 1(2) of the [Software Directive (2009/24/EC)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32009L0024) says that "ideas and principles which underlie any element of a computer program" are not protected, and the Court of Justice confirmed in *SAS Institute v World Programming* ([C-406/10](https://curia.europa.eu/juris/liste.jsf?num=C-406/10), 2012) that a program's functionality, its programming language and its data formats are not themselves protected expression. Only the code is.

An architecture described in an arXiv paper is a published idea. Anyone may implement it. What you cannot do is copy or adapt someone's GPL/AGPL source files and relicense the result. An independent implementation, written by people working from the paper rather than the source, is a new work whose author picks the license.

Two honest caveats most vendors skip:

- **"Independent" has to be true.** The defensible version of this is a clean-room process: the people writing the code work from the paper and a spec, not from the copyleft source. Reading the GPL repo closely and then "rewriting it" is not the same thing, and the difference matters if anyone ever looks.
- **This is a copyright argument, not a patent one.** Independent invention is *not* a defense against a patent. Clean-room reimplementation clears the copyright question and leaves the patent question exactly where it was. There is no widely asserted patent hanging over the YOLO architectures today, but "no one has sued yet" is not the same as "there is nothing to sue over."

## The weights are a second license

The most expensive mistake in this whole area is auditing the code and forgetting the checkpoint. Three layers carry licenses, and they can all differ:

**1. The code.** Everything above.

**2. The weights.** Sometimes explicit, as with YOLO-NAS. Sometimes asserted: Ultralytics' [license page](https://ultralytics.com/license) states that AGPL-3.0 "covers the training code and the models produced by that training code," and extends that to models you train yourself with their code. As the copyright holder of what they publish, they can set the terms for their own checkpoints. Whether weights *you* produce are legally a derivative work of the training code is a genuinely open question: there is no case law, and it cuts against the FSF's own general rule that a program's output is not automatically covered by the program's copyright. Unsettled is not the same as safe, though. In practice, treat published AGPL weights as encumbered.

The sharp edge for anyone taking this article's advice: **the GPL YOLOv7 and YOLOv9 repos say nothing at all about their weights.** Their checkpoints are release assets in a GPL-3.0 repository with no separate grant. Loading those .pt files into an MIT reimplementation gets you clean code and a checkpoint whose status nobody has clarified. If the license is the reason you switched, use weights the permissive project trained itself, or train your own.

**3. The training data.** COCO's *annotations* are CC BY 4.0, so the labels are fine. The *images* are not COCO's to license: they are Flickr photos under a patchwork of individual terms, which the COCO consortium explicitly does not own. Most of the industry treats this as low risk and moves on, but "COCO is fine" is a statement about the annotations only. Beyond COCO it gets stricter fast: plenty of aerial, medical and driving datasets are non-commercial, and a model's permissive code license does not launder them.

Audit all three layers, every time.

## A note on MIT versus Apache-2.0

We ship an MIT-licensed library, so it would be convenient to tell you MIT is simply the best license. The honest comparison is more interesting.

MIT is not zero-obligation: you must keep the copyright notice and license text in copies you distribute. That is a real condition, just a cheap one. And Apache-2.0 has something MIT does not: an **express patent grant** from every contributor, plus a retaliation clause that terminates the grant of anyone who sues over the work. MIT is silent on patents. For a company whose main fear is patent exposure, Apache-2.0 is arguably the *safer* of the two, and much of the permissive detection ecosystem (YOLOX, RT-DETR, D-FINE, DEIM) is Apache-2.0 for exactly that reason.

What MIT buys you is simplicity and compatibility. Neither license will ever force you to publish your source, which is the axis this article is actually about. If someone tells you MIT versus Apache is the important decision, they are selling something. The important decision is permissive versus copyleft.

## So which YOLO can you actually use?

- **Closed-source commercial product:** YOLOv7 or YOLOv9 through the lab's [MIT repo](https://github.com/MultimediaTechLab/YOLO), YOLOX or PP-YOLOE (from PaddleDetection) among the originals, or a maintained MIT framework like LibreYOLO if you would rather not assemble and audit this yourself. Avoid anything GPL or AGPL unless you will open your whole application or buy the Enterprise license.
- **Open-source project:** anything, as long as your license is compatible. If your project is AGPL, the Ultralytics line is a natural fit, and the newest research (YOLOv13, YOLO26, YOLOE) lands there first.
- **Research and benchmarking:** licenses barely constrain you. Use whatever the paper you are comparing against used.
- **The "we'll deal with it later" plan:** does not work. Unwinding an AGPL dependency after your product is built means retraining, revalidating and re-exporting everything, including the weights. Pick the repository, and therefore the license, first.

For a comparison of the frameworks themselves rather than their licenses, see [Best Ultralytics Alternatives in 2026](/articles/best-ultralytics-alternatives).

## The MIT option: what LibreYOLO ships

Disclosure: LibreYOLO is our project, and its license is the reason this article exists.

**[LibreYOLO](https://github.com/LibreYOLO/libreyolo) is one MIT-licensed codebase** covering the detectors below behind a single API, with training and ONNX, TensorRT, OpenVINO and NCNN export built in. No copyleft, no network clause, no enterprise tier.

Here is the full object detection lineup with each family's upstream and its license, so you can see exactly where each permissive route comes from:

| In LibreYOLO | Model | Upstream | Upstream license |
| --- | --- | --- | --- |
| `LibreYOLO2`, `LibreYOLO3`, `LibreYOLO4` | The Darknet-era YOLOs, in PyTorch | [pjreddie/darknet](https://github.com/pjreddie/darknet), [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) | Public domain |
| `LibreYOLO7` | YOLOv7 | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (the lab's own rewrite, not the GPL repo) |
| `LibreYOLO9`, `LibreYOLO9E2E` | YOLOv9, plus an NMS-free end-to-end variant | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (the lab's own rewrite, not the GPL repo) |
| `LibreYOLO9P2` | YOLOv9 with a stride-4 head for small objects | LibreYOLO original | MIT |
| `LibreYOLOX` | YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| `LibreYOLONAS` | YOLO-NAS, detect and pose | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) | Apache-2.0 code. The restricted upstream weights are **not** redistributed |
| `LibreRTDETR`, `LibreRTDETRv2` | RT-DETR and v2 | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 |
| `LibreRTDETRv4` | RT-DETRv4 | [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | Apache-2.0 |
| `LibreRFDETR` | RF-DETR, detect, segment, pose, OBB | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | Apache-2.0 for N/S/M/L. XL/2XL are under Roboflow's Platform Model License, so we ship only the Apache sizes |
| `LibreDFINE` | D-FINE | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 |
| `LibreDEIM` | DEIM | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 |
| `LibreDEIMv2` | DEIMv2 | [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | Apache-2.0 code. The larger sizes use Meta's DINOv3 backbone under Meta's own non-OSI licence, which allows redistribution but **bars military, nuclear, espionage and weapons use**. Those sizes are published as `other`, not Apache. Read it before you ship anything dual-use |
| `LibreRTMDet` | RTMDet ([without MMDetection](/articles/rtmdet-without-mmdetection)) | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | Apache-2.0 (the Apache sibling, not the GPL-3.0 mmyolo) |
| `LibrePICODET` | PP-PicoDet | Architecture from [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), checkpoints via the [Picodet_Pytorch](https://github.com/Bo396543018/Picodet_Pytorch) re-port | Apache-2.0 (both) |
| `LibreEC` | EdgeCrafter, detect, pose, segment | [Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) | Apache-2.0 |
| `LibreFOMO` | Centroid detector for microcontroller-class hardware | LibreYOLO original | MIT |
| `LibreGroundingDINO` | Grounding DINO, open-vocabulary (inference) | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 |
| `LibreOWLv2` | OWLv2, open-vocabulary (inference) | Google, via `transformers` | Apache-2.0 |

Two things that column is deliberately honest about.

**No GPL or AGPL source is copied into the codebase.** Every family above is built from a public domain, MIT or Apache-2.0 upstream, which is the rule that keeps the framework MIT. Where an architecture has both a permissive and a copyleft home, we build from the permissive one: our YOLOv9 and YOLOv7 trace to the lab's MIT repo rather than the GPL originals, and our RTMDet to Apache-licensed MMDetection rather than GPL-licensed MMYOLO. YOLO-World is absent from the list, despite being asked for regularly, because it is GPL-3.0 and there is nowhere permissive to take it from.

**Permissive is not the same as unrestricted, and we would rather say so than let you discover it in an audit.** DEIMv2's larger sizes inherit Meta's DINOv3 terms, including a prohibition on military, nuclear, espionage and weapons use, which is a real constraint if you work anywhere near dual-use. A handful of research-preview checkpoints are trained on non-commercial datasets (DOTA, VisDrone) and are labelled `cc-by-nc` on Hugging Face rather than quietly shipped as if they were free. And the same caution we apply to everyone else's weights applies to ours: our YOLO9 checkpoints are converted from the MIT lab repo's own checkpoints, so they inherit whatever is true of those, which as noted above is "asserted, not formally audited." The MIT license covers the code we wrote. Every checkpoint carries the terms of what it was trained on and from, published per weight.

Beyond detection, the same API covers instance and semantic segmentation, pose, oriented boxes, classification (MobileNetV4, ConvNeXt, EfficientNetV2, ResNet, CLIP), monocular depth, image restoration and gaze estimation.

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
```

Same YOLO workflow, none of the license homework.

Star it on GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Docs: [libreyolo.com/docs](https://libreyolo.com/docs)

---

*Trademarks and affiliation: YOLO, YOLOv8, YOLO11, YOLO26 and Ultralytics are trademarks of their respective owners, including Ultralytics Inc. LibreYOLO is an independent project and is not affiliated with, sponsored by, or endorsed by Ultralytics or any other organisation named in this article. Product and repository names are used only to identify and compare the software discussed.*

*This article is general information, not legal advice, current as of 11 July 2026, and written by one of the vendors in the comparison. Verify the current license before you ship.*
