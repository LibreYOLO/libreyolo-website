---
title: FeyNobg
families:
  - feynobg
seo_title: 'FeyNobg : suppression d''arrière-plan dans LibreYOLO'
description: >-
  Utilisez FeyNobg dans LibreYOLO pour la suppression d'arrière-plan et le
  matting alpha, une variante approfondie de BiRefNet signée Feyn Inc.
  Installez, prédisez et validez.
lead: >-
  Un modèle de suppression d'arrière-plan de Feyn Inc. qui approfondit
  l'architecture de BiRefNet et la réentraîne. LibreYOLO fournit l'inférence et
  la validation pour la tâche matte de FeyNobg.
keywords:
  - FeyNobg
  - suppression d'arrière-plan python
  - segmentation dichotomique d'images
  - matte alpha
  - image matting
  - détourage automatique
  - nobg
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFeyNobgl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Détourage
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8 : le RGB source plus le matte en canal alpha.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFeyNobgl-matte.pt")

        # Un répertoire contenant images/ et un dossier de mattes détecté
        # automatiquement (mattes/, matte/, gt/, masks/, mask/ ou alpha/)
        # marche aussi à la place d'un YAML de dataset.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
source_hash: 45de3b578d7ebbf2
---

## Installation

FeyNobg ne demande aucun extra optionnel. Tout ce qu'il importe fait partie de
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Le checkpoint est téléchargé depuis l'organisation LibreYOLO sur Hugging Face
au premier usage, puis mis en cache localement, comme pour toute autre famille,
même s'il n'est pas encore listé dans le tableau des checkpoints de cette page.

<code-tabs name="predict" />

Un résultat de matte ne porte aucune bounding box ; `result.matte` est un
tableau dense `(H, W)` float32 dans `[0, 1]`, 1 pour un premier plan complet et
0 pour un arrière-plan complet. Contrairement à un masque binaire, le matte doux
conserve le détail des bords anticrénelés, cheveux et fourrure compris.
`result.cutout()` compose l'image source avec ce canal alpha dans un tableau
RGBA, et `result.save(path)` (ou `save=True` sur l'appel de prédiction) l'écrit
directement dans un PNG à fond transparent. Le modèle tourne sur un canevas
natif fixe de 1024x1024 ; une autre résolution n'est pas prise en charge, parce
que les tables de positions relatives du backbone Swin y sont liées et qu'un
écart les interpole mal au lieu de lever une erreur. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Variantes

Une seule taille publiée, `l`, un backbone de niveau Swin-L. FeyNobg reprend
l'architecture de BiRefNet et approfondit son troisième étage Swin de 18 à 24
blocs avant de la réentraîner, si bien que le portage LibreYOLO réutilise le
chemin forward, le prétraitement et le contrat de sortie à logit unique de
BiRefNet ; la prédiction, la validation et la gestion des checkpoints se
comportent comme dans la famille `birefnet`.

## Valider

`val()` renvoie deux métriques sur un dossier apparié image/matte, toutes deux
dans `[0, 1]` et indépendantes de la résolution : la MAE, l'erreur absolue
moyenne par rapport à l'alpha de vérité terrain (plus c'est bas, mieux c'est),
et la S-measure (Fan et al., ICCV 2017), une similarité structurelle qui
valorise la conservation de la forme du sujet et de ses trous, ce que la MAE par
pixel seule ignore (plus c'est haut, mieux c'est). La validation passe par le
`predict` du modèle lui-même, elle utilise donc exactement le prétraitement de
la famille.

<code-tabs name="val" />

La validation se fait en inférence seule. La bibliothèque `nobg` en amont
fournit du code d'entraînement sous Apache-2.0 ; faire du fine-tuning
aujourd'hui, c'est entraîner là-bas puis convertir le résultat avec le script de
conversion propre à LibreYOLO, et non appeler `train()` sur cette famille, qui
lève une erreur au lieu de lancer un entraîneur partiel.

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
