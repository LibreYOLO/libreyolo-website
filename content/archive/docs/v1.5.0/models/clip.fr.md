---
title: CLIP
families:
  - clip
seo_title: "CLIP dans LibreYOLO\_: classifier en zero-shot et calculer des embeddings"
description: "Utilisez CLIP dans LibreYOLO pour la classification d'images en zero-shot et les embeddings image/texte. Aucun entraînement\_: set_classes() définit l'ensemble des étiquettes à l'exécution."
lead: >-
  CLIP est un modèle à deux tours qui évalue une image face à des prompts
  textuels plutôt que face à un ensemble d'étiquettes figé. LibreYOLO le prend
  en charge pour la classification en zero-shot et les embeddings image/texte,
  sans aucune étape d'entraînement.
keywords:
  - CLIP
  - OpenCLIP
  - classification zero-shot
  - embedding d'images python
  - recherche d'images par texte
  - vocabulaire ouvert
  - LAION-2B
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Sans appel à set_classes(), la prédiction CLI utilise les 1 000 noms

        # de classes ImageNet chargés par défaut avec le modèle.

        libreyolo predict model=LibreCLIPb32-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Embeddings image et texte
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        image_embed = model(SAMPLE_IMAGE).embeddings.data
        text_embed = model.embed_text("a photo of a forklift")

        # Tous deux sont normalisés L2 : un produit scalaire donne le cosinus.
        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        # data est une racine ImageFolder avec un split train/ ; ses noms de
        # dossiers deviennent les prompts de classes zero-shot de ce run.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCLIPb32-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # Les étiquettes set_classes() actuelles et la résolution d'entrée sont
        # figées dans le graphe. Réexportez après avoir changé l'une ou l'autre.
    - label: CLI
      language: bash
      code: |
        # Aucun appel à set_classes() ici : cela fige donc les 1 000 classes
        # ImageNet par défaut chargées avec le modèle.
        libreyolo export model=LibreCLIPb32-cls.pt format=onnx
    - label: Export des embeddings
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" trace la seule tour image ; aucune classe requise.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: ac7cfd75ad6c0fa7
---

## Installation

CLIP demande son propre extra, qui installe les paquets dont son tokenizer BPE embarqué a besoin pour reproduire exactement les identifiants de tokens.

```bash
pip install "libreyolo[clip]"
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en cache localement.

<code-tabs name="predict" />

`set_classes()` est la primitive qui fait de ce modèle un classifieur à vocabulaire ouvert : elle insère chaque étiquette dans chaque template de prompt, encode puis moyenne les résultats, et met en cache la matrice `[K, D]` obtenue comme tête de classification, si bien qu'elle n'est pas recalculée pour chaque image. Rappelez-la à tout moment pour changer de classes. Sans aucun appel, LibreCLIP se charge avec les 1 000 noms de classes d'ImageNet-1k déjà définis.

Avec `task="embed"`, la prédiction renvoie un vecteur image normalisé L2 par entrée au lieu de probabilités de classes, et `embed_text()` renvoie des lignes de texte normalisées dans le même espace vectoriel, si bien qu'un simple produit scalaire entre les deux donne la similarité cosinus. `iou` n'a d'effet sur aucune des deux tâches ; il n'y a pas d'étape de NMS. Voir [la prédiction](/docs/predict) pour les sources, le streaming et le traitement des résultats.

## Valider

`val()` lit les noms des dossiers de classes sous un split `train/` de type ImageFolder, appelle `set_classes()` avec eux, puis mesure l'exactitude zero-shot top-1 et top-5. L'exactitude dépend de la façon dont les noms de classes se lisent comme prompts, et non d'une mise à jour des poids, puisqu'il n'y a rien à entraîner. La validation ne couvre que `task="classify"` ; `task="embed"` n'a pas de validateur sur dataset.

<code-tabs name="val" />

## Exporter

<export-matrix />

L'export inscrit l'état courant du modèle dans un graphe figé. Pour `task="classify"`, les dernières étiquettes définies par `set_classes()` et la résolution au moment de l'export sont figées dans une couche linéaire finale, si bien que le graphe ONNX ou TensorRT exporté est un classifieur d'images `[B, K]` ordinaire, sans tour texte ni tokenizer ; réexportez après avoir changé les classes ou la taille. L'export en `task="embed"` trace la seule tour image. Les deux exigent l'opset ONNX 14 ou supérieur, que l'exporteur définit par défaut.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille. Les deux sont convertis depuis les checkpoints d'OpenCLIP entraînés sur LAION-2B (`ViT-B-32` et `ViT-B-16`), et non depuis un entraînement sur COCO.

<checkpoint-table />

Les données d'entraînement LAION-2B ont un historique documenté de contenus pédocriminels (CSAM) (Stanford Internet Observatory, décembre 2023). LAION a depuis publié Re-LAION, une réédition nettoyée ; si vous réhébergez ces poids ailleurs, préférez les checkpoints dérivés de Re-LAION lorsqu'ils existent.

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
