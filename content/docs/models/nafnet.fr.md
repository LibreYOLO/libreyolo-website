---
title: NAFNet
families:
  - nafnet
seo_title: 'NAFNet : débruiter, entraîner et exporter sous MIT'
description: >-
  Utilisez NAFNet dans LibreYOLO pour le débruitage et la restauration d'images.
  Installez, prédisez, entraînez, validez et exportez le checkpoint SIDD sous
  licence MIT.
lead: >-
  NAFNet est un réseau convolutionnel de restauration d'images qui retire les
  fonctions d'activation non linéaires d'un bloc UNet classique et les remplace
  par une multiplication élément par élément. LibreYOLO le prend en charge pour
  une tâche, la restauration, avec un checkpoint publié de débruitage d'images
  réelles entraîné sur SIDD.
keywords:
  - NAFNet
  - restauration d'images
  - débruitage d'images
  - défloutage d'images
  - réseau sans activation non linéaire
  - SIDD
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model("noisy.jpg", save=True)

        restored = result.restored
        print(restored.array.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg
        save=True
    - label: Enregistrer l'image restaurée
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model.predict("noisy.jpg")

        result.restored.save("denoised.png")
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 imgsz=256 batch=16 lr0=1e-3
    - label: Provenance du checkpoint
      language: python
      code: >
        from libreyolo import LibreYOLO


        # La dégradation et le dataset sont inscrits dans le checkpoint
        enregistré ;

        # ils ne modifient pas ce qui est entraîné.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
    - label: Multi-GPU
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() renvoie un simple dict, pas un objet
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.export(format="onnx", imgsz=256)
        model.export(format="tensorrt", imgsz=256, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=onnx
        imgsz=256

        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=tensorrt
        imgsz=256 half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO


        # La factory utilise le suffixe du fichier : un artefact exporté se
        charge

        # comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")

        result = model("noisy.jpg")


        result.restored.save("denoised.png")
source_hash: 9bae9f82bee741bf
---

## Installer

NAFNet ne nécessite aucun extra facultatif. Tous ses imports figurent dans
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé comporte un champ pour cette famille, `restored`, une
image RGB dense HWC uint8 sur le canevas d'origine ; aucune boîte ne doit être
parcourue. `save=True` écrit directement cette image restaurée sur le disque au
lieu de tracer une annotation sur l'entrée. `conf`, `iou` et `max_det` sont
acceptés pour maintenir la même signature que toutes les autres familles, mais
n'ont aucun effet puisque la restauration ne produit aucune détection à filtrer.
Consultez la [prédiction](/docs/predict) pour les sources, le streaming et le
traitement des résultats.

## Variantes

Deux largeurs partagent cette architecture : `s` (largeur 32) et `l` (largeur
64), toutes deux construites autour d'un patch d'entraînement de 256 px. La
prédiction et la validation s'exécutent à la résolution native de l'image quelle
que soit la taille, avec un remplissage limité au facteur de sous-échantillonnage
du réseau. Seule la largeur `l` est actuellement publiée, sous la forme d'un
checkpoint de débruitage d'images réelles entraîné sur SIDD.

## Entraîner

NAFNet effectue le fine-tuning sur vos propres paires d'images dégradées et
nettes : un YAML de dataset pointe vers un dossier `inputs/<split>/` d'images
dégradées et un dossier `targets/<split>/` de cibles nettes, appariées par le
radical du nom de fichier. `degradation` et `dataset` sont des chaînes
facultatives enregistrées dans le checkpoint sauvegardé à des fins de
provenance ; elles ne participent pas à l'entraînement.

<code-tabs name="train" />

Avec les réglages par défaut, l'entraîneur exécute 100 époques avec AdamW à
`lr0=1e-3`, un batch de 16, des recadrages de 256 px et un early stopping après
50 époques sans amélioration du PSNR. Cette famille ne propose aucun chemin
LoRA : `lora=True` lève une erreur au lieu de s'exécuter, car `NAFNetTrainer`
n'active jamais le fine-tuning par adaptateur.

Pendant l'entraînement, le réseau s'exécute avec un simple pooling moyen global.
Le pooling local fenêtré réservé à l'inférence de NAFNet (Test-time Local
Converter) est détaché avant la première époque, puis réattaché une fois
l'entraînement terminé, car la rétropropagation à travers un pooling local à
fenêtre fixe ne correspondrait pas à l'utilisation du checkpoint lors de
l'inférence.

Consultez l'[entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire avec `metrics/PSNR` et `metrics/SSIM`, calculés
en RGB sur l'ensemble du canevas valide : SSIM utilise une fenêtre gaussienne
11x11 avec un sigma de 1.5, et la valeur `fitness` employée pour sélectionner le
meilleur checkpoint est le PSNR. `data` pointe vers le même format de dataset
d'images appariées que celui utilisé pour l'entraînement.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge par `LibreYOLO()` à partir de son suffixe de
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint et
renvoie le même objet `Results`, où `restored` contient l'image de sortie. NAFNet
s'exporte à une résolution spatiale fixe : `imgsz` doit être divisible par le
facteur de sous-échantillonnage du réseau (16 pour les deux largeurs de
l'architecture), et seule la dimension de batch est dynamique lorsque
`dynamic=True` ; la hauteur et la largeur sont fixées au moment de l'export.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
