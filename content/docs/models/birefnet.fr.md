---
title: BiRefNet
families:
  - birefnet
seo_title: "BiRefNet\_: suppression d'arrière-plan et matting dans LibreYOLO"
description: >-
  Utilisez BiRefNet dans LibreYOLO pour la suppression d'arrière-plan et la
  segmentation dichotomique d'images. Installez, prédisez, validez et exportez
  le checkpoint general.
lead: >-
  Un réseau à référence bilatérale qui prédit un matte alpha doux séparant un
  sujet de son arrière-plan. LibreYOLO fournit l'inférence et la validation pour
  la tâche matte de BiRefNet.
keywords:
  - BiRefNet
  - suppression d'arrière-plan python
  - détourage automatique d'image
  - matte alpha
  - image matting
  - supprimer le fond d'une photo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreBiRefNetl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Détourage
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8 : le RGB source plus le matte en canal alpha.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Un répertoire contenant images/ et un dossier de mattes détecté
        # automatiquement (mattes/, matte/, gt/, masks/, mask/ ou alpha/)
        # marche aussi à la place d'un YAML de dataset.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreBiRefNetl-matte.pt format=onnx
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme n'importe quel checkpoint et renvoie le
        # même objet Results.
        model = LibreYOLO("LibreBiRefNetl-matte.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: 1af1bd7f4f905081
---

## Installation

BiRefNet ne demande aucun extra optionnel. Tout ce qu'il importe fait partie
de l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

Un résultat de matte ne porte aucune bounding box ; `result.matte` est un
tableau dense `(H, W)` float32 dans `[0, 1]`, 1 pour un premier plan complet
et 0 pour un arrière-plan complet. Contrairement à un masque binaire, le matte
doux conserve le détail des bords anticrénelés, cheveux et fourrure compris.
`result.cutout()` compose l'image source avec ce canal alpha dans un tableau
RGBA, et `result.save(path)` (ou `save=True` sur l'appel de prédiction)
l'écrit directement dans un PNG à fond transparent. Le modèle tourne sur un
canevas natif fixe de 1024x1024 ; une autre résolution n'est pas prise en
charge, parce que les tables de positions relatives du backbone Swin y sont
liées et qu'un écart les interpole mal au lieu de lever une erreur. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Variantes

Un seul checkpoint publié, `l`, le modèle BiRefNet-general de niveau Swin-L et
le modèle par défaut orienté qualité en amont. Le code de la famille prend
aussi en charge un niveau léger Swin-T, `t`, mais aucune conversion LibreYOLO
n'en est encore publiée.

## Valider

`val()` renvoie deux métriques sur un dossier apparié image/matte, toutes deux
dans `[0, 1]` et indépendantes de la résolution : la MAE, l'erreur absolue
moyenne par rapport à l'alpha de vérité terrain (plus c'est bas, mieux c'est),
et la S-measure (Fan et al., ICCV 2017), une similarité structurelle qui
valorise la conservation de la forme du sujet et de ses trous, ce que la MAE
par pixel seule ignore (plus c'est haut, mieux c'est). La validation passe par
le `predict` du modèle lui-même, elle utilise donc exactement le prétraitement
de la famille.

<code-tabs name="val" />

La validation se fait en inférence seule ; le fine-tuning est une suite
documentée plutôt qu'une fonctionnalité livrée (voir Prédire pour la contrainte
de résolution exacte dont hériterait tout futur entraîneur).

## Exporter

<export-matrix />

Un artefact exporté se recharge via `LibreYOLO()` selon son extension de
fichier, si bien qu'un fichier `.onnx` se comporte comme un checkpoint et
renvoie le même `Results`. TorchScript est le chemin validé ; la conversion
ONNX fonctionne mais n'a pas passé la même barre de parité.
[L'export](/docs/export) liste les arguments que chaque format accepte, ainsi
que les extras que quelques-uns ajoutent.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
