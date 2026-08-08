---
title: Citation
seo_title: "Cite LibreYOLO and the upstream authors"
description: "How to cite LibreYOLO in a paper, and how to cite the authors of the model family you ran. Both belong in the same methods section."
lead: "A complete LibreYOLO citation has two parts: the library, and the published work behind the model family that produced the result."
keywords: [cite libreyolo, libreyolo bibtex, libreyolo citation cff, model citation, computer vision citation]
last_verified: "1.5.0"
---

## Citing LibreYOLO

The repository publishes its citation metadata as
[`CITATION.cff`](https://github.com/LibreYOLO/libreyolo/blob/release/CITATION.cff),
not as a BibTeX block. GitHub reads that file and offers a Cite this repository
button on the repository page, which generates APA and BibTeX from it. Take the
entry from there rather than typing one.

The file in full:

```yaml
cff-version: 1.2.0
message: "If you use LibreYOLO in your research or software, please cite it as below."
title: "LibreYOLO"
type: software
authors:
  - family-names: Ceccon
    given-names: Xuban
  - name: "The LibreYOLO contributors"
license: MIT
url: "https://github.com/LibreYOLO/libreyolo"
repository-code: "https://github.com/LibreYOLO/libreyolo"
```

It carries no version and no release date on purpose.
[`RELEASING.md`](https://github.com/LibreYOLO/libreyolo/blob/release/RELEASING.md)
tells maintainers never to bump, date or retitle `CITATION.cff` or
`.zenodo.json` during a release, so that every citation lands on one record
instead of scattering across versions. Report the version you ran in your own
text, and leave the citation alone.

## Citing the model family

LibreYOLO is a port. Running `LibreRFDETRm.pt` means running RF-DETR, and the
people who wrote RF-DETR are the ones a reviewer expects to see credited.
Citing the library on its own attributes their work to the wrong project.

Everything needed sits on the family's page. The Upstream row in the header
names the original work and the organization behind it, and links the paper and
the source repository. The Citation section further down holds the BibTeX.

That BibTeX is copied verbatim from the authors' own citation block, normally
the Citation section of the upstream README or a `CITATION.cff`, and it renders
with a link back to the block it came from so you can check it against the
source. It is never assembled from paper metadata. An entry rebuilt by hand
fails quietly and expensively: a dropped coauthor, the wrong venue, the wrong
entry type, a year belonging to the preprint. Preprints also get accepted, so
an entry may be an `@inproceedings` even when the version you read was on arXiv.

Copy the block as it stands. If your bibliography style needs a different entry
type, convert the entry rather than retyping it, and keep the author list in
its original order.

## What a methods section needs

Three things make a LibreYOLO result reproducible and correctly attributed:

- The library, cited from `CITATION.cff`, together with the version you ran.
  `libreyolo version` prints it, along with the Python, torch and CUDA versions
  it is running against.
- The upstream work, cited from the Citation section of the family's page.
- The exact checkpoint filename, such as `LibreRFDETRm.pt`. Sizes within a
  family behave differently, and several families publish checkpoints trained
  on different datasets under the same prefix, so the family name alone does
  not identify what ran.

Attribution is also a license term for much of what LibreYOLO publishes.
Apache-2.0 and the CC BY family both require the notice to travel with the
weights you redistribute, which is a separate obligation from citing a paper.
See [licensing](/docs/licensing) for which terms apply to which checkpoint.
