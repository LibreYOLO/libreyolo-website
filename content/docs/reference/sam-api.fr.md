---
title: API de segmentation guidable
seo_title: "API LibreSAM\_: prompts, alias et signatures"
description: >-
  La fabrique LibreSAM, ses alias de tailles, les prompts par point, bounding
  box et texte conceptuel, le cycle set_image qui encode une fois, et les
  fonctionnalités non prises en charge par ce niveau.
lead: >-
  LibreSAM est la fabrique de segmentation guidable. Une passe forward nécessite
  un prompt propre à l'image et transmis au moment de l'appel, ce niveau possède
  donc sa propre interface de prédiction au lieu de passer par le runner
  d'inférence sans prompt.
keywords:
  - LibreSAM
  - segmentation guidable
  - prompt point SAM
  - prompt bounding box SAM
  - set_image
  - tout segmenter
  - extra sam libreyolo
last_verified: 1.5.0
verification: >-
  Alias de fabrique, tailles et dépôts lus dans libreyolo/models/sam/model.py,
  sam2.py, edgetam.py, sam3.py, libreyolo/models/mobilesam/model.py et
  libreyolo/models/picosam3/model.py. Contrat des prompts et valeurs par défaut
  lus dans libreyolo/models/sam/base.py. Intention de conception lue dans
  docs/adr/0007-libresam-contract.md, le tout en v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[sam]'
  usage:
    - label: Prompts par point et bounding box
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        r = model.predict(SAMPLE_IMAGE, points=[900, 370], labels=[1])
        print(r.masks.xy)
        print(r.boxes.xyxy)

        r = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 200, 200])
        print(len(r))
    - label: 'Encoder une fois, guider plusieurs fois'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")
        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[500, 375], labels=[1])
        b = model.predict(bboxes=[100, 100, 200, 200])
        print(len(a), len(b))

        model.reset_image()
source_hash: 18e8206c10ce17fd
---

## Installer

Ce niveau nécessite l'extra `sam`.

<code-tabs name="install" />

## La fabrique

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model` est un alias de taille, et non un chemin. `**kwargs` atteint le
constructeur de la famille, qui accepte `device` et `multimask`. Un alias
inconnu lève `ValueError` et le message énumère tous les alias connus.

<code-tabs name="usage" />

## Alias

| Famille | Alias | Tailles | Poids |
|---|---|---|---|
| SAM-1 | `base`, `large`, `huge`, `b`, `l`, `h`, `sam-base`, `sam-large`, `sam-huge`, `sam_b`, `sam_l`, `sam_h` | `base`, `large`, `huge` | `facebook/sam-vit-base`, `-large`, `-huge` |
| SAM-2 | `sam2-tiny`, `sam2-small`, `sam2-base-plus`, `sam2-baseplus`, `sam2-large`, et les formes courtes `sam2-t`, `sam2-s`, `sam2-bp`, `sam2-l`, `sam2_t`, `sam2_s`, `sam2_bp`, `sam2_l` | `tiny`, `small`, `base-plus`, `large` | `LibreYOLO/LibreSAM2tiny`, `-small`, `-base-plus`, `-large` |
| EdgeTAM | `edgetam`, `edge-tam`, `edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`, `sam-3`, `sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`, `mobilesam-tiny`, `mobilesam_t`, `mobile-sam`, `mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`, `picosam3-pico`, `picosam3_pico`, `pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

La valeur par défaut est `base`. SAM-1, SAM-2, EdgeTAM et MobileSAM
s'exécutent sur un canevas nominal de 1024 pixels, SAM 3 sur 1008 et PicoSAM3
sur 96.

Les poids SAM 3 sont protégés. Ils sont téléchargés depuis `facebook/sam3`
sous la SAM License personnalisée de Meta, qui n'est ni MIT ni Apache-2.0, et
LibreYOLO ne les redistribue pas. Acceptez les conditions sur la page du dépôt
et authentifiez-vous auprès de Hugging Face avant le chargement. Le chargeur
consigne d'abord l'avis.

Les classes de familles sont également exportées. `LibreSAM1`, `LibreSAM2`,
`LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM` et `LibrePicoSAM3` peuvent donc
être construites directement avec `size=`.

## predict

```python
model.predict(
    source=None,
    *,
    points=None,
    bboxes=None,
    labels=None,
    masks=None,
    text=None,
    conf=None,
    multimask=None,
    max_det=300,
    device=None,
    color_format="auto",
    points_per_side=None,
) -> Results
```

| Argument | Valeur par défaut | Signification |
|---|---|---|
| `source` | `None` | Image à segmenter\u00a0; `None` réutilise l'image mise en cache par `set_image()` |
| `points` | `None` | Prompt par point en coordonnées de pixels |
| `bboxes` | `None` | Prompt par bounding box sous la forme `[x1, y1, x2, y2]`, ou liste de bounding boxes avec un masque par bounding box |
| `labels` | `None` | Étiquettes des points, `1` positif et `0` négatif, dont la forme correspond à `points`\u00a0; toutes positives si elles sont omises |
| `masks` | `None` | Réservé\u00a0; transmettre une valeur lève `NotImplementedError` |
| `text` | `None` | Prompt conceptuel\u00a0; SAM 3 uniquement |
| `conf` | `None` | Seuil minimal de l'IoU prédite du masque |
| `multimask` | `None` | Renvoyer tous les masques d'ambiguïté par prompt\u00a0; utilise par défaut le réglage de construction |
| `max_det` | `300` | Limite du nombre de masques renvoyés |
| `device` | `None` | Déplacer le modèle pour cet appel et les suivants, en invalidant les embeddings en cache |
| `color_format` | `"auto"` | Indication de format des couleurs pour les tableaux en mémoire |
| `points_per_side` | `None` | Densité de la grille du mode tout segmenter\u00a0; valeur par défaut 32 |

La valeur renvoyée est un objet `Results` ordinaire qui contient `masks`,
ainsi que des `boxes` ajustées dérivées de ces masques, avec la classe `0`
nommée `"object"`.

## Formes des prompts

`points` accepte les formes imbriquées `[x, y]` pour un objet,
`[[x, y], ...]` pour N objets et `[[[x, y], ...], ...]` pour des points
regroupés par objet. Les tableaux Numpy fonctionnent partout où une liste est
acceptée. Les coordonnées sont des pixels ordinaires sur l'image source.

Omettre tous les prompts spatiaux exécute le mode tout segmenter, un générateur
automatique de masques sur grille doté d'un seuil d'IoU prédite et d'une
déduplication par IoU des bounding boxes. La valeur par défaut 32 de
`points_per_side` exécute environ 1024 passes de décodeur, ce qui est lent sur
CPU\u00a0; réduisez-la pour un usage interactif. Le générateur omet le filtrage par
score de stabilité, le multi-crop et la déduplication par IoU des masques. Il
s'agit donc d'une approximation du chemin guidé et non d'un équivalent exact.

## Confiance

`conf` filtre selon l'IoU de masque prédite, qui mesure la qualité du masque et
non une confiance de détection. `None` conserve chaque masque du chemin guidé
et applique le seuil de grille de la famille dans le mode tout segmenter. `0.0`
désactive le filtrage dans les deux modes.

Sur le chemin texte de SAM 3, `conf` correspond plutôt au score de détection
Promptable Concept Segmentation. `None` y signifie le seuil standard 0.3 et
`0.0` conserve tous les candidats.

## Prompts textuels

`text=` est propre à SAM 3\u00a0; toutes les familles à prompts spatiaux lèvent
`NotImplementedError` pour ce paramètre. Le texte est mutuellement exclusif
avec les points et les bounding boxes. Le dictionnaire `names` renvoyé associe
la classe `0` au concept demandé. Un appel textuel avec `source=None` réencode
l'image en cache, car le tracker et l'encodeur conceptuel ne partagent pas de
cache.

Le mot-clé `exemplars=` est réservé à une future extension d'exemplaires
d'images et n'est pas implémenté.

## Cycle d'encodage unique

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image` exécute une fois le lourd encodeur d'image et met les embeddings
en cache, chaque appel `predict()` ultérieur avec `source=None` est donc peu
coûteux. Les deux méthodes renvoient le modèle pour permettre le chaînage des
appels. Transmettre `device=` à `predict` déplace le modèle et invalide le
cache.

## PicoSAM3

PicoSAM3 accepte uniquement `bboxes=`. Les prompts par point, texte, masque,
multimask et tout segmenter lèvent une erreur. La bounding box est agrandie de
10\u00a0% et traverse un réseau ROI de 96 pixels. PicoSAM3 est l'unique famille du
niveau qui s'exporte, uniquement vers ONNX.

## Fonctionnalités non prises en charge

`train()`, `val()` et `track()` lèvent `NotImplementedError` sur chaque famille
du niveau. Les masques guidables n'ont aucun ensemble fixe de classes auquel
être comparés, la mAP n'a donc aucun sens ici. `export()` lève une erreur pour
SAM-1, SAM-2, SAM 3, EdgeTAM et MobileSAM.

Les chemins vidéo et mémoire de SAM-2, SAM 3 et EdgeTAM sont hors périmètre
pour cette version, tout comme les exemplaires d'images SAM 3 et les prompts
par masque.
