---
title: RTMDet
families:
  - rtmdet
seo_title: 'RTMDet dans LibreYOLO : prédire, entraîner et exporter'
description: >-
  Exécutez RTMDet dans LibreYOLO pour la détection d'objets et la segmentation
  d'instances RTMDet-Ins. Installez, prédisez, entraînez, validez et exportez
  sous Apache-2.0.
lead: >-
  RTMDet est un détecteur à une étape qui prédit à partir d'un prior fondé sur
  un point par emplacement de grille, sans ancres, au moyen d'une tête dont les
  convolutions sont partagées entre les niveaux de caractéristiques. LibreYOLO
  le prend en charge pour la détection et la segmentation d'instances
  RTMDet-Ins.
keywords:
  - RTMDet
  - détection d'objets
  - segmentation d'instances
  - RTMDet-Ins
  - détection sans ancres
  - mmdetection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRTMDets.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Segmentation d'instances
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Le suffixe -seg du nom sélectionne la tête de masque RTMDet-Ins,
        # aucun argument task n'est donc nécessaire ici.
        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTMDets.pt data=my-dataset.yaml
    - label: Segmentation d'instances
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # masques
        print(metrics["metrics/mAP50-95(B)"])   # boîtes
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, imgsz=640, batch=16, lr0=0.004,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreRTMDets.pt data=my-dataset.yaml imgsz=640
        epochs=300 batch=16 lr0=0.004
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRTMDets.pt format=onnx imgsz=640

        libreyolo export model=LibreRTMDets.pt format=tensorrt imgsz=640
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory utilise le suffixe du fichier : un artefact exporté se
        charge

        # comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreRTMDets.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 2f5033bdc1c3c931
---

## Installer

RTMDet ne nécessite aucun extra en plus du paquet de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est le même pour toutes les familles, remplacer le
détecteur ne demande donc de modifier qu'une ligne. Un nom de fichier contenant
`-seg` sélectionne de lui-même la tâche RTMDet-Ins, et `result.masks` contient
alors les masques d'instances en plus des boîtes. `conf` définit le seuil de
confiance et `iou` le seuil NMS. Consultez la [prédiction](/docs/predict) pour
les sources, le streaming et le traitement des résultats.

## Variantes

Cinq tailles, de `t` à `x`, partagent une même architecture à une résolution
d'entrée commune. Cette famille ne comporte aucun tableau de benchmark ici :
comparez les tailles selon celle des fichiers de checkpoint dans le tableau
ci-dessous.

## Entraîner

<code-tabs name="train" />

La détection s'entraîne par `train()`. Les composants QualityFocalLoss, GIoU et
DynamicSoftLabelAssigner sont portés depuis mmdetection en amont. La passe et
l'export ONNX lui sont équivalents bit à bit, avec un post-traitement qui
correspond à la sortie de mmdet à moins de 0.001 mAP sur des sous-ensembles de
val2017.

Comme l'indique la propre docstring de `train()`, les éléments suivants n'ont pas
été vérifiés : la convergence du fine-tuning sur un petit dataset, la parité avec
l'article pour l'entraînement à partir de zéro, le comportement multi-GPU, le
débit de Mosaic et MixUp mis en cache, la transition stricte du pipeline amont en
deux étapes, ainsi que les remplacements paramétrés de weight decay qui
annulent le decay sur les paramètres de normalisation et de biais.

RTMDet-Ins ne possède aucun chemin d'entraînement. Appeler `train()` sur un
checkpoint `-seg`, ou avec `task="segment"`, lève `NotImplementedError` ; la
segmentation d'instances prend uniquement en charge l'inférence et la validation.

`train()` accepte également un argument `pretrained`, mais sa valeur n'est
jamais lue dans la méthode : l'entraînement reprend toujours à partir des poids
avec lesquels le modèle a été construit, et `pretrained=False` ne réinitialise
donc pas le réseau.

Avec les autres réglages par défaut, l'entraîneur exécute 300 époques avec AdamW
à `lr0=0.004` et `weight_decay=0.05`, un warmup d'une époque selon un planning
cosinus, tandis que Mosaic et MixUp sont désactivés pour les 20 dernières
époques.

Consultez l'[entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurés sur tout dataset dans le format
utilisé pour l'entraînement.

<code-tabs name="val" />

Avec un checkpoint `-seg`, la clé simple `metrics/mAP50-95` contient le score
des masques. La même exécution rapporte aussi les boîtes sous `(B)` et les
masques sous `(M)`, afin que les deux soient disponibles en une seule passe.

## Exporter

<export-matrix />

La détection s'exporte vers la plupart des formats, tandis que la segmentation
d'instances ne s'exporte actuellement vers aucun d'eux ; la matrice ci-dessus
reflète cette distinction. Un artefact de détection exporté se recharge par
`LibreYOLO()` à partir de son suffixe de fichier. Un fichier `.onnx` ou `.engine`
se comporte donc comme un checkpoint et renvoie le même objet `Results`.
L'exécution du graphe dans un runtime brut, sans installation de LibreYOLO, est
également prise en charge, mais vous devez alors écrire vous-même le prétraitement
et le post-traitement.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
