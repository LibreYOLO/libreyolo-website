---
title: MobileNetV4
families:
  - mobilenetv4
seo_title: 'MobileNetV4 : entraîner, valider et exporter sous Apache-2.0'
description: >-
  Utilisez MobileNetV4 dans LibreYOLO pour la classification d'images.
  Installez, prédisez, affinez, validez et exportez LibreMobileNetV4
  small/medium/large.
lead: >-
  MobileNetV4 est un classifieur d'images conçu pour le matériel mobile et edge.
  Son bloc Universal Inverted Bottleneck réunit plusieurs conceptions
  antérieures de blocs mobiles dans une structure unique pouvant faire l'objet
  d'une recherche. LibreYOLO le prend en charge pour une seule tâche : la
  classification.
keywords:
  - MobileNetV4
  - MobileNetV4 conv
  - classification d'images
  - inférence mobile
  - classifieur edge
  - classifieur ImageNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMobileNetV4s-cls.pt source=cat.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreMobileNetV4s-cls.pt data=imagenette160
        epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreMobileNetV4s-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMobileNetV4s-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMobileNetV4s-cls.pt format=onnx

        libreyolo export model=LibreMobileNetV4s-cls.pt format=tensorrt
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory utilise le suffixe du fichier : un artefact exporté se
        charge

        # comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreMobileNetV4s-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 4a9a1b392ffb136d
---

## Installer

MobileNetV4 ne nécessite aucun extra facultatif. Tous ses imports figurent dans
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est le même pour toutes les familles, remplacer le
modèle ne demande donc de modifier qu'une ligne. Un classifieur ne comporte ni
boîtes ni masques : `result.probs` contient la prédiction de l'image entière,
avec `top1`, `top5`, `top1conf` et `top5conf`. `conf`, `iou` et `max_det` sont
acceptés pour la parité de l'API, mais n'ont aucun effet puisqu'un seul vecteur
de probabilités ne contient rien à seuiller ou à supprimer. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et le traitement des
résultats.

## Variantes

Trois tailles, small/medium/large, reposent exclusivement sur des convolutions :
cette famille exclut les variantes hybrides qui ajoutent l'attention Mobile MQA.
Le choix d'une taille constitue un compromis direct entre nombre de paramètres
et exactitude. La tâche est fixe : chaque taille couvre uniquement la
classification. Le nom du fichier de poids se termine par `-cls.pt` pour chaque
taille, et la factory utilise ce suffixe pour sélectionner cette famille ; aucun
argument `task=` n'est nécessaire.

## Entraîner

Le fine-tuning part du backbone ImageNet publié et reconstruit automatiquement
la dernière couche de classification selon le nombre de classes du dataset cible.

<code-tabs name="train" />

Avec les réglages par défaut, l'entraîneur exécute 100 époques avec `lr0=1e-3`,
AdamW, un batch de 64 et un early stopping après 50 époques sans amélioration.
`data` accepte la racine d'un dataset (`train/` et `val/`, un dossier par classe),
un nom court connu comme `imagenette160` ou l'URL d'un fichier `.zip`. `lora=True`
n'est pas pris en charge ici ; son utilisation lève une erreur, car LoRA dans
LibreYOLO cible les composants transformer dotés de couches `nn.Linear`, absentes
des blocs UIB de cette famille.

Consultez l'[entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/`. Pour la classification, il
s'agit de l'exactitude top-1 et top-5 sur la partition de validation.

<code-tabs name="val" />

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
