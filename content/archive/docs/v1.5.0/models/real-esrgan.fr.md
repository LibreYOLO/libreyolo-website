---
title: Real-ESRGAN
families:
  - realesrgan
seo_title: 'Real-ESRGAN : super-résolution d''images dans LibreYOLO'
description: >-
  Utilisez Real-ESRGAN dans LibreYOLO pour une super-résolution d'images
  pratique en 4x, 2x et dans une variante 4x rapide. Installez, prédisez,
  validez et exportez.
lead: >-
  Un upscaler pratique de super-résolution aveugle, entraîné sur des
  dégradations synthétiques plutôt que seulement sur un sous-échantillonnage
  bicubique. LibreYOLO fournit l'inférence et la validation pour ses checkpoints
  4x, 2x et 4x rapide.
keywords:
  - Real-ESRGAN
  - RRDBNet
  - SRVGGNetCompact
  - super-résolution d'images
  - restauration d'images
  - super-résolution aveugle
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRealESRGANx4-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 'Par tuiles, pour les grandes images'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # tile divise la passe en tuiles qui se chevauchent et fusionne les
        # raccords ; tile_pad est la marge ajoutée autour de chaque tuile avant
        # son recadrage. Ce sont des arguments Python, pas des options de CLI.
        result = model("large-photo.jpg", tile=512, tile_pad=10, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: >
        libreyolo val model=LibreRealESRGANx4-restore.pt
        data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRealESRGANx4-restore.pt")


        # Sans valeur, imgsz utilise une petite taille de patch interne, pas
        votre

        # résolution de travail. Indiquez la taille réellement fournie au modèle

        # par votre déploiement.

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRealESRGANx4-restore.pt format=onnx
        imgsz=512
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory utilise le suffixe du fichier : un artefact exporté se
        charge

        # comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreRealESRGANx4-restore.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.restored.array.shape)
source_hash: f0efb4f65d38e22d
---

## Installer

Real-ESRGAN ne nécessite aucun extra facultatif. Tous ses imports figurent dans
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

Un résultat de restauration ne contient aucune boîte ; `result.restored` est
une image RGB dense uint8 de forme `(H, W, 3)`, sur un canevas égal à
`Results.restore_scale` fois l'entrée dans chaque dimension. `save=True` écrit
directement cette image au lieu d'un tracé annoté. L'entrée est convertie en RGB
et tout canal alpha est supprimé. Une source trop grande pour la mémoire
disponible peut être divisée avec `tile` et `tile_pad`, qui fusionnent les
raccords des tuiles dans la sortie. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et le traitement des
résultats.

## Variantes

Trois checkpoints sont nommés selon leur facteur d'agrandissement. `x4` est
RRDBNet (`RealESRGAN_x4plus`), avec 23 blocs denses residual-in-residual, et
constitue le choix de qualité par défaut en 4x. `x2` emploie la même architecture
RRDBNet en 2x. `x4t` est SRVGGNetCompact (`realesr-general-x4v3`), un générateur
plus petit et plus rapide, conçu pour la vidéo et une utilisation à plus faible
latence en 4x. Le modèle polyvalent amont fournit aussi un réseau apparié de
niveau de débruitage, fusionné lors de l'inférence. Ce réglage d'intensité ne fait
pas partie de ce portage, qui exécute le générateur `x4t` de base.

## Valider

`val()` mesure le PSNR et le SSIM entre la sortie restaurée et une image cible
nette. Les deux sont calculés en RGB sur le canevas d'origine, sans recadrage de
bord ni redimensionnement. SSIM utilise une fenêtre gaussienne 11x11 avec un
sigma de 1.5, puis établit la moyenne sur les trois canaux de couleur.

<code-tabs name="val" />

L'argument de dataset est un YAML qui associe un dossier d'images d'entrée
dégradées à un dossier d'images cibles nettes de résolution correspondante ;
consultez les [formats de datasets](/docs/reference/dataset-formats) pour les
clés exactes.

## Exporter

<export-matrix />

Un artefact exporté se recharge par `LibreYOLO()` à partir de son suffixe de
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint et
renvoie le même objet `Results`. La page [Export](/docs/export) répertorie les
arguments acceptés par chaque format et les options supplémentaires de certains
d'entre eux.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
