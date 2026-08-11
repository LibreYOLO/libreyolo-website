---
title: DINOv2
families:
  - dinov2
seo_title: "DINOv2 dans LibreYOLO\_: segmentation sémantique, classification et embeddings"
description: >-
  Utilisez DINOv2 dans LibreYOLO pour la segmentation sémantique, la
  classification et l'embedding d'image entière, sur le backbone
  DINOv2-with-Registers. Apache-2.0 de bout en bout.
lead: "DINOv2 est un vision transformer auto-supervisé entraîné par Meta AI pour produire des caractéristiques d'images polyvalentes sans étiquettes. LibreYOLO encapsule son backbone DINOv2-with-Registers pour trois tâches\_: la segmentation sémantique, la classification et l'embedding d'image entière."
keywords:
  - DINOv2
  - DINOv2 with registers
  - apprentissage auto-supervisé
  - vision transformer
  - segmentation sémantique python
  - embedding d'images
  - extraction de caractéristiques
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Sémantique
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.dinov2.model import LibreDINOv2


        # Aucun checkpoint hébergé par LibreYOLO n'existe pour cette famille :

        # ceci télécharge le backbone Apache-2.0 DINOv2-with-Registers-small

        # depuis l'org Hugging Face de Meta. La tête dense reste initialisée

        # aléatoirement tant que vous ne l'entraînez pas (voir Entraîner plus
        bas).

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)

        result = model(SAMPLE_IMAGE)


        mask = result.semantic_mask

        print(mask.data.shape, mask.classes)
    - label: Classification
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # nb_classes= est le nombre de classes de votre dataset ; la tête
        # linéaire reste initialisée aléatoirement tant que vous ne
        # l'entraînez pas.
        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
    - label: Embedding
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Contourne toutes les têtes de tâche : le backbone seul suffit,
        # donc aucun fine-tuning n'est nécessaire pour que ce soit utile.
        model = LibreDINOv2(size="s", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (1, D), normalisé L2
    - label: Embedding d'un batch
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Wrapper pratique : lance predict() et empile chaque ligne dans un
        # seul tenseur (N, D).
        features = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(features.shape)
  train:
    - label: Sémantique
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Classification
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Multi-GPU
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(
            data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4,
            device="0,1",
        )
  val:
    - label: Sémantique
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Classification
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
  export:
    - label: Sémantique
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.export(format="onnx")
    - label: Classification
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.export(format="onnx")
    - label: Embedding
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        model.export(format="tflite")
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory route selon le suffixe du fichier : un artefact exporté se
        # charge comme n'importe quel checkpoint et renvoie le même objet
        # Results. L'export nomme le fichier d'après la tâche, ici
        # LibreDINOv2s-sem.onnx.
        model = LibreYOLO("LibreDINOv2s-sem.onnx")
        result = model(SAMPLE_IMAGE)
source_hash: 4256e0a0398e5aaf
---

## Installation

LibreDINOv2 ne s'enregistre que si `transformers` est installé, la même
dépendance optionnelle dont RF-DETR a besoin pour son backbone DINOv2 ; il
demande donc le même extra.

```bash
pip install "libreyolo[rfdetr]"
```

## Prédire

LibreYOLO ne publie aucun checkpoint LibreDINOv2. Construisez le wrapper
directement au lieu de charger un fichier : `model_path=None` (la valeur par
défaut) télécharge au premier usage le backbone Apache-2.0
`facebook/dinov2-with-registers-small` de Meta depuis Hugging Face. `task=`
choisit ce qui tourne par-dessus.

<code-tabs name="predict" />

`task="semantic"` et `task="classify"` ajoutent une tête dense ou linéaire
par-dessus le backbone ; cette tête est initialisée aléatoirement et n'est
utile qu'une fois que vous l'avez entraînée (voir [Entraîner](#train)).
`task="embed"` saute toutes les têtes et renvoie le token CLS final normalisé
du backbone comme une seule ligne pour l'image entière dans
`result.embeddings`, si bien qu'il ne demande aucun entraînement.
`result.boxes` vaut toujours `None` : aucune des trois tâches ne produit de
détections par instance. Voir [la prédiction](/docs/predict) pour les sources,
le streaming et le traitement des résultats.

## Variantes

`size` choisit la largeur du projecteur de style RF-DETR posé par-dessus le
backbone, pas le backbone lui-même : toutes les tailles partagent le même
encodeur DINOv2-S (small). La segmentation sémantique tourne sur la grille de
patches carrée native de DINOv2 ; la classification et l'embedding tournent à
la résolution de classification, plus petite, utilisée pour entraîner le
linear probe.

## Entraîner

`task="semantic"` et `task="classify"` s'entraînent tous les deux ;
`task="embed"` n'a aucune tête dépendante des classes à ajuster et lève
`NotImplementedError` si vous appelez `train()` dessus.

<code-tabs name="train" />

Les arguments nommés principaux sont ici `batch_size` et `lr`, pas `batch` et
`lr0` utilisés par la plupart des autres familles ; `batch` et `lr0` restent
acceptés et sont redirigés vers eux, mais passer les deux lève une erreur de
conflit. `output_dir=` (par défaut `"runs/train"`) remplace `project=`/`name=`
comme moyen principal de placer un run, même si passer directement
`project=`/`name=` fonctionne toujours. Voir [l'entraînement](/docs/train)
pour les datasets, l'augmentation de données, le multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` : mIoU et exactitude par
pixel pour `task="semantic"`, exactitude top-1 et top-5 pour
`task="classify"`. `task="embed"` n'a aucune vérité terrain à laquelle se
mesurer et lève `NotImplementedError` si vous appelez `val()` dessus.

<code-tabs name="val" />

## Exporter

<export-matrix />

Chaque tâche prend en charge un sous-ensemble de formats différent, indiqué
ci-dessus. Un artefact exporté se recharge via `LibreYOLO()` d'après son
suffixe de fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte
comme un checkpoint et renvoie les mêmes `Results`. [L'export](/docs/export)
liste les arguments que chaque format accepte.

<code-tabs name="export" />

## Licence

<provenance-box>

La ligne « Poids » ci-dessus nomme la licence qui s'applique, Apache-2.0, mais
rien n'est réellement republié sous l'org Hugging Face de LibreYOLO pour cette
famille : LibreYOLO n'héberge aucun checkpoint LibreDINOv2 qui lui soit
propre. Ce que `LibreDINOv2(model_path=None)` télécharge, c'est le dépôt
`facebook/dinov2-with-registers-small` de Meta, intact.

</provenance-box>

## Citation

<citation-block />
