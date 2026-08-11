---
title: ConvNeXt
families:
  - convnext
seo_title: "ConvNeXt\_: entraîner, valider et exporter sous Apache-2.0"
description: >-
  Utilisez ConvNeXt dans LibreYOLO pour la classification d'images.
  Installation, prédiction, fine-tuning avec LoRA, validation et export de
  LibreConvNeXt tiny/small/base.
lead: "ConvNeXt est un classifieur d'images construit entièrement à partir de convolutions standard, modernisé bloc par bloc depuis un ResNet vers les choix de conception d'un vision transformer. LibreYOLO le prend en charge pour une seule tâche\_: la classification."
keywords:
  - ConvNeXt
  - ConvNeXt tiny
  - classification d'images python
  - réseau entièrement convolutif
  - classifieur ImageNet pré-entraîné
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreConvNeXtt-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 epochs=5
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreConvNeXtt-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreConvNeXtt-cls.pt format=onnx
        libreyolo export model=LibreConvNeXtt-cls.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme n'importe quel checkpoint et renvoie le
        # même objet Results.
        model = LibreYOLO("LibreConvNeXtt-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 1682cc69cf2925e6
---

## Installation

ConvNeXt ne demande aucun extra optionnel. Tout ce qu'il importe fait partie
de l'installation de base.

```bash
pip install libreyolo
```

Le fine-tuning par adaptateurs avec `lora=True` fait exception et demande
l'extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui que renvoie chaque famille, si bien que
passer à un autre modèle tient en une ligne. Un classifieur ne porte ni boîtes
ni masques : `result.probs` contient la prédiction sur l'image entière, avec
`top1`, `top5`, `top1conf` et `top5conf`. `conf`, `iou` et `max_det` sont
acceptés par parité d'API mais n'ont aucun effet, puisqu'il n'y a rien à
seuiller ni à supprimer sur un simple vecteur de probabilités. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Variantes

Trois tailles, tiny/small/base, toutes entraînées et évaluées de la même
manière : en choisir une revient donc à arbitrer directement entre nombre de
paramètres et exactitude. La tâche est fixe : chaque taille ne couvre que la
classification. Le nom du fichier de poids se termine par `-cls.pt` à toutes
les tailles, et c'est ce suffixe que lit la factory pour router vers cette
famille ; aucun argument `task=` n'est nécessaire.

## Entraîner

Le fine-tuning part du backbone ImageNet publié et reconstruit
automatiquement la couche de classification finale au nombre de classes du
dataset cible.

<code-tabs name="train" />

Sans réglage particulier, l'entraînement fait 100 époques à `lr0=1e-3` avec
AdamW, un batch de 64 et un early stopping après 50 époques sans amélioration.
`data` accepte une racine de dataset (`train/` et `val/`, un dossier par
classe), un nom court connu comme `imagenette160`, ou une URL `.zip`. Les
blocs de ConvNeXt portent les MLP `nn.Linear` dont LoRA a besoin, donc
`lora=True` est pris en charge ici, et injecte des adaptateurs dans les MLP
des blocs plutôt que de faire du fine-tuning sur l'ensemble du backbone.

Voir [l'entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/`. Pour la classification, il
s'agit de l'exactitude top-1 et top-5 sur le split de validation.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge via `LibreYOLO()` selon son extension de
fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un
checkpoint et renvoie le même `Results`. [L'export](/docs/export) liste les
arguments que chaque format accepte, ainsi que les extras que quelques-uns
ajoutent.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box>

Seul ConvNeXt V1 est fourni dans cette famille. Les petits checkpoints
pré-entraînés de ConvNeXt-V2 sont sous CC-BY-NC 4.0 et sont délibérément
exclus, puisqu'un poids non commercial ne peut pas être redistribué au sein
d'une bibliothèque MIT/commerciale.

</provenance-box>

## Citation

<citation-block />
