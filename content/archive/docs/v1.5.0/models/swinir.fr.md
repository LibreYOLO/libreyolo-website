---
title: SwinIR
families:
  - swinir
seo_title: "SwinIR\_: super-résolution d'images 4x dans LibreYOLO"
description: >-
  Utilisez SwinIR dans LibreYOLO pour la super-résolution d'images 4x.
  Installez, prédisez, validez et exportez les checkpoints lightweight, medium
  et large.
lead: "SwinIR est un réseau Swin Transformer pour la restauration d'images. LibreYOLO fournit l'inférence et la validation de ses checkpoints de super-résolution 4x\_: le générateur léger officiel, ainsi que les générateurs medium et large pour images réelles."
keywords:
  - SwinIR
  - Swin Transformer
  - super-résolution image
  - restauration d'images
  - residual Swin Transformer block
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwinIRm-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 'Par tuiles, pour les grandes images'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRl-restore.pt")

        # tile divise la passe forward en tuiles superposées et fusionne leurs
        # raccords ; tile_pad est le halo ajouté autour de chaque tuile avant
        # de la recadrer. Ce sont deux arguments nommés propres à Python,
        # et non des options CLI.
        result = model("large-photo.jpg", tile=512, tile_pad=16, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwinIRm-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSwinIRm-restore.pt")


        # Lorsque imgsz est omis, sa valeur par défaut est la taille d'un petit
        patch interne,

        # pas votre résolution de travail. Transmettez donc la taille réellement

        # envoyée au modèle par votre déploiement.

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwinIRm-restore.pt format=onnx imgsz=512
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La fabrique détermine le chemin selon le suffixe du fichier ; un
        artefact exporté se charge

        # donc comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreSwinIRm-restore.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.restored.array.shape)
source_hash: 87fc3d5524480eec
---

## Installer

SwinIR ne nécessite aucun extra facultatif. Tous ses imports sont inclus dans
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et
mis en cache localement.

<code-tabs name="predict" />

Un résultat de restauration ne contient aucune bounding box\u00a0;
`result.restored` est une image RGB uint8 dense de forme `(H, W, 3)`, sur un
canevas 4x plus grand que l'entrée dans chaque dimension. `save=True` écrit
directement cette image plutôt qu'une visualisation annotée. L'entrée est
complétée jusqu'à un multiple de 8 et non redimensionnée, la prédiction
s'exécute donc à la résolution propre de la photo. Une source dépassant la
mémoire disponible peut être divisée avec `tile` et `tile_pad`, qui refusionnent
les raccords des tuiles dans la sortie. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Variantes

Trois tailles, toutes limitées à un agrandissement 4x. `s` est le générateur
léger officiel, avec quatre étapes residual Swin Transformer block (RSTB) et
un upsampling direct par pixel shuffle. `m` et `l` sont les générateurs medium
et large pour images réelles. Ils comportent six et neuf étapes RSTB ainsi
qu'un upsampler par plus proche voisin suivi d'une convolution, conçu pour les
dégradations réelles plutôt que pour le seul sous-échantillonnage bicubique.

## Valider

`val()` mesure le PSNR et le SSIM entre la sortie restaurée et une image cible
propre. Tous deux sont calculés en RGB sur le canevas d'origine, sans recadrage
des bords ni redimensionnement. Le SSIM utilise une fenêtre gaussienne 11x11
avec un sigma de 1.5, dont la moyenne est calculée sur les trois canaux de
couleur.

<code-tabs name="val" />

L'argument dataset est un fichier YAML qui associe un répertoire d'images
d'entrée dégradées à un répertoire d'images cibles propres de résolution
correspondante. Consultez les [formats de datasets](/docs/reference/dataset-formats)
pour connaître les clés exactes.

## Exporter

<export-matrix />

Un artefact exporté se recharge dans `LibreYOLO()` grâce au suffixe de son
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint
et renvoie les mêmes `Results`. ExecuTorch et tous les formats indiqués comme
bloqués dans la matrice ne sont pas disponibles pour cette famille\u00a0; ONNX,
TorchScript, TensorRT, OpenVINO et TFLite le sont. La page
[Export](/docs/export) énumère les arguments acceptés par chaque format ainsi
que les extras ajoutés par certains.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
