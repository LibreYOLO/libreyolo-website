---
title: ResNet
families:
  - resnet
seo_title: 'ResNet : entraîner, valider et exporter sous Apache-2.0'
description: >-
  Utilisez ResNet dans LibreYOLO pour la classification d'images. Installez,
  prédisez, affinez, validez et exportez LibreResNet18/34/50/101.
lead: >-
  ResNet est un classifieur d'images composé de blocs résiduels, des connexions
  de saut qui permettent d'ajouter bien davantage de couches à un réseau sans la
  perte d'exactitude que subissent les empilements convolutionnels profonds
  ordinaires. LibreYOLO le prend en charge pour une seule tâche : la
  classification.
keywords:
  - ResNet
  - ResNet50
  - classification d'images
  - apprentissage résiduel
  - réseaux résiduels profonds
  - classifieur ImageNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
        libreyolo export model=LibreResNet50-cls.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory utilise le suffixe du fichier : un artefact exporté se
        charge

        # comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreResNet50-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: e2f46c73716af1b7
---

## Installer

ResNet ne nécessite aucun extra facultatif. Tous ses imports figurent dans
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

Quatre profondeurs sont entraînées et évaluées de la même manière. Le choix de
l'une d'elles constitue donc un compromis direct entre nombre de paramètres et
exactitude. La tâche est fixe : chaque taille couvre uniquement la
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
LibreYOLO cible les composants transformer dotés de couches `nn.Linear`,
absentes de ResNet.

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

## Citation

<citation-block />
