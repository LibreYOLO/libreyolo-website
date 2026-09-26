---
title: SegFormer
families:
  - segformer
seo_title: "SegFormer\_: segmentation sémantique dans LibreYOLO"
description: "Utilisez SegFormer dans LibreYOLO pour la segmentation sémantique ADE20K, des tailles b0 à b5. Installez, prédisez, entraînez et exportez\_; les poids pré-entraînés sont réservés à un usage non commercial."
lead: >-
  SegFormer est un transformer de segmentation sémantique qui associe un
  encodeur hiérarchique Mix Transformer (MiT) à une tête de décodage légère
  entièrement composée de MLP. Il évite ainsi les décodeurs lourds et les
  encodages positionnels fixes dont avaient besoin les transformers de
  segmentation antérieurs. LibreYOLO le prend en charge pour une seule tâche, la
  segmentation sémantique, dans six tailles.
keywords:
  - SegFormer
  - segmentation sémantique
  - Mix Transformer
  - MiT
  - transformer segmentation
  - ADE20K
  - prédiction dense
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSegformerb0-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (fine-tuning)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: À partir de zéro
      language: python
      code: >
        from libreyolo.models.segformer.model import LibreSegformer


        # Aucun model_path : initialisation aléatoire, aucun téléchargement.
        C'est la

        # seule voie vers des poids exempts de la clause non commerciale des
        checkpoints pré-entraînés.

        model = LibreSegformer(size="b0", nb_classes=150)

        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512

        libreyolo export model=LibreSegformerb0-sem.pt format=tensorrt imgsz=512
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La fabrique détermine le chemin selon le suffixe du fichier ; un
        artefact exporté se charge

        # donc comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreSegformerb0-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: c236895b991beabf
---

## Installer

SegFormer ne nécessite aucun extra facultatif. Tous ses imports sont inclus
dans l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et
mis en cache localement.

<code-tabs name="predict" />

`result.semantic_mask` contient la carte de classes dense\u00a0: `.data` est un
tenseur `(H, W)` d'identifiants de classe à la taille de l'image d'origine, et
`.classes` énumère les identifiants de classe réellement présents.
`result.boxes` vaut `None`, puisqu'il n'existe aucune détection par instance.
`conf` et `iou` sont acceptés pour assurer la parité de l'API, mais ils ne
modifient pas la sortie\u00a0: le modèle renvoie une classe par pixel et non des
détections par instance à filtrer ou dédupliquer. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Variantes

Six tailles, de b0 à b5, élargissent et approfondissent l'encodeur Mix
Transformer à chaque étape tout en conservant la même conception de tête de
décodage entièrement composée de MLP.

<checkpoint-table />

## Entraîner

Par défaut, `train()` effectue le fine-tuning d'un checkpoint publié. Si vous
ne transmettez aucun `model_path` à `LibreSegformer(...)`, il construit au
contraire un encodeur et une tête initialisés aléatoirement et les entraîne à
partir de zéro. C'est la seule voie vers des poids dépourvus de la restriction
non commerciale des checkpoints pré-entraînés (consultez la section
[Licence](#licensing)).

<code-tabs name="train" />

Sans configuration supplémentaire, le trainer suit la recette ADE20K de
l'article SegFormer\u00a0: AdamW au learning rate de base du backbone, une tête de
décodage entraînée à un learning rate 10x supérieur, un weight decay partout
sauf sur LayerNorm et la convolution positionnelle Mix-FFN, ainsi qu'un
planning de décroissance linéaire avec warmup. La convergence des grandes
tailles, de b3 à b5, n'a pas été validée de bout en bout.

Consultez l'[entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/`\u00a0: mIoU et exactitude par
pixel, mesurées sur tout dataset au format utilisé pour l'entraînement.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge dans `LibreYOLO()` grâce au suffixe de son
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint
et renvoie les mêmes `Results`. La page [Export](/docs/export) énumère les
arguments acceptés par chaque format.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box>

L'encodeur et la tête de décodage de LibreSegformer sont un portage PyTorch de
l'implémentation SegFormer sous Apache-2.0 de Hugging Face Transformers, et
non de NVlabs/SegFormer\u00a0: le dépôt d'origine de NVIDIA n'a jamais été lu ni
copié, et n'est cité ici que pour attribuer l'article à ses auteurs. Seuls les
checkpoints pré-entraînés ci-dessus sont soumis à la restriction non
commerciale de NVIDIA\u00a0; l'architecture et le code propre à LibreYOLO restent
entièrement sous licence MIT.

</provenance-box>

## Citation

<citation-block />
