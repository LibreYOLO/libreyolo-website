---
title: LingBot-Vision
families:
  - lingbotvision
seo_title: 'LingBot-Vision : la segmentation sémantique dans LibreYOLO'
description: >-
  Utilisez LingBot-Vision dans LibreYOLO pour la segmentation sémantique sur un
  backbone ViT sous Apache-2.0. Installez, prédisez, entraînez, validez et
  exportez, tailles s/b/l.
lead: >-
  LingBot-Vision est une famille de backbones vision transformer
  auto-supervisés, entraînés par masked modeling centré sur les contours pour la
  perception spatiale dense, publiée par Robbyant. LibreYOLO associe le backbone
  à une tête dense et le prend en charge pour une seule tâche, la segmentation
  sémantique.
keywords:
  - LingBot-Vision
  - segmentation sémantique python
  - vision transformer
  - pré-entraînement auto-supervisé
  - Robbyant
  - prédiction dense
  - backbone ViT apache 2.0
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLingBotVisions-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (linear probe)
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Backbone gelé par défaut, conformément au protocole d'évaluation
        # d'amont. Seule la tête dense 1x1 s'entraîne.
        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(data="my-dataset.yaml", epochs=20, imgsz=512, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 imgsz=512 batch=16
    - label: Fine-tune complet
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(
            data="my-dataset.yaml", epochs=20, imgsz=512, batch=16,
            freeze_backbone=False,
        )
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLingBotVisions-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="coreai", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLingBotVisions-sem.pt format=onnx imgsz=512
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme un checkpoint et renvoie le même Results.
        model = LibreYOLO("LibreLingBotVisions-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c47b33fdc6fa1139
---

## Installation

LingBot-Vision ne demande aucun extra optionnel. Tout ce qu'il importe fait
partie de l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

`result.semantic_mask` porte la carte dense des classes : `.data` est un
tenseur `(H, W)` d'ids de classe à la taille d'origine de l'image, et
`.classes` liste les ids de classe réellement présents. `result.boxes` vaut
`None`, puisqu'il n'y a pas de détections par instance. `conf` et `iou` sont
acceptés par parité d'API mais ne changent rien à la sortie, puisque le modèle
renvoie une classe par pixel plutôt que des détections à filtrer. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Variantes

Trois tailles publiées, s, b et l, distillées depuis un enseignant ViT-g/16 de
1.1 milliard de paramètres. L'enseignant lui-même, taille `g`, se charge et se
prête au fine-tuning dans LibreYOLO, mais LibreYOLO n'héberge pas de checkpoint
`g` propre.

<checkpoint-table />

## Entraîner

`train()` fait du fine-tuning sur un checkpoint publié. La recette par défaut
est le linear probe du rapport d'amont : le backbone ViT est gelé et seule la
tête dense 1x1 s'entraîne, ce qui correspond à la façon dont les poids hébergés
par LibreYOLO ci-dessus ont été produits. Passez `freeze_backbone=False` pour
faire du fine-tuning sur tout le réseau à la place, et attendez-vous à baisser
`lr0` en conséquence.

<code-tabs name="train" />

Voir [l'entraînement](/docs/train) pour les datasets, l'augmentation de données,
le multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` : mIoU et exactitude par
pixel, mesurées sur n'importe quel dataset au format sur lequel vous avez
entraîné.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge via `LibreYOLO()` selon son extension de
fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un
checkpoint et renvoie le même `Results`. [L'export](/docs/export) liste les
arguments que chaque format accepte.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box>

La version d'amont documente son ViT comme construit sur l'architecture
DINOv2/DINOv3 publiée par Meta AI. Robbyant distribue son implémentation sous
Apache-2.0, et ce portage LibreYOLO a été réalisé uniquement à partir du dépôt
Robbyant, jamais à partir du code DINOv2 ou DINOv3 de Meta.

</provenance-box>

## Citation

<citation-block />
