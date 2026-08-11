---
title: SigLIP2
families:
  - siglip2
seo_title: "SigLIP2 dans LibreYOLO\_: classification zero-shot et embeddings"
description: >-
  Utilisez SigLIP2 dans LibreYOLO pour la classification d'images zero-shot et
  les embeddings d'images ou de textes, avec un score sigmoïde multi-label.
  Aucun entraînement nécessaire.
lead: >-
  SigLIP2 est un modèle à deux tours qui compare une image à des prompts
  textuels avec une sigmoïde indépendante par classe, au lieu d'un softmax
  partagé sur un ensemble d'étiquettes fixe. LibreYOLO le prend en charge pour
  la classification zero-shot et les embeddings d'images ou de textes, sans
  étape d'entraînement.
keywords:
  - SigLIP2
  - SigLIP 2
  - classification zero-shot
  - image embedding
  - text embedding
  - vocabulaire ouvert
  - modèle multilingue
  - sigmoid loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Sans appel à set_classes(), la prédiction CLI utilise les 1 000 noms
        de

        # classes ImageNet chargés par défaut avec le modèle.

        libreyolo predict model=LibreSigLIP2b16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Score sigmoïde multi-label
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        model.set_classes(["a dog", "a cat", "outdoors"], multi_label=True)

        r = model(SAMPLE_IMAGE)


        # Probabilités indépendantes par classe : plusieurs classes, ou aucune,

        # peuvent obtenir un score élevé simultanément. Le softmax (par défaut)
        les

        # normalise plutôt en distribution mono-label, comme le comportement de
        LibreCLIP.

        for i, name in model.names.items():
            print(name, float(r.probs.data[i]))
    - label: Embeddings d'images et de textes
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")

        image_embed = model(SAMPLE_IMAGE).embeddings.data

        text_embed = model.embed_text("a photo of a forklift")


        # Tous deux sont normalisés L2 ; un simple produit scalaire donne donc
        la similarité cosinus.

        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSigLIP2b16-cls.pt")


        # data est la racine d'un ImageFolder avec un sous-ensemble train/ ; les
        noms de

        # dossiers deviennent les prompts de classes zero-shot pour cette
        exécution.

        metrics = model.val(data="imagenette160")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSigLIP2b16-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        model.set_classes(["a forklift", "an empty aisle", "a spill"])

        model.export(format="onnx")


        # Les étiquettes set_classes() actuelles et la résolution d'entrée sont
        figées

        # dans le graphe. Réexportez après avoir changé l'une ou l'autre.
        multi_label

        # doit valoir False (la valeur par défaut) au moment de l'export.
    - label: CLI
      language: bash
      code: |
        # Aucun appel à set_classes() ici, les 1 000 classes ImageNet chargées
        # avec le modèle par défaut sont donc figées dans le graphe.
        libreyolo export model=LibreSigLIP2b16-cls.pt format=onnx
    - label: Export d'embeddings
      language: python
      code: >
        from libreyolo import LibreYOLO


        # task="embed" trace uniquement la tour d'image ; aucune classe
        nécessaire.

        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")

        model.export(format="onnx")
source_hash: f992655747fd8819
---

## Installer

SigLIP2 nécessite son propre extra, qui installe le package SentencePiece
utilisé par son tokenizer multilingue.

```bash
pip install "libreyolo[siglip2]"
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et
mis en cache localement.

<code-tabs name="predict" />

`set_classes()` est l'unique primitive qui transforme ce modèle en
classificateur à vocabulaire ouvert\u00a0: elle insère chaque étiquette dans tous
les templates de prompts, encode les résultats, calcule leur moyenne et met
en cache la matrice `[K, D]` obtenue comme tête de classification. Celle-ci
n'est donc pas recalculée pour chaque image. Appelez de nouveau cette méthode
à tout moment pour changer de classes. Sans appel, LibreSigLIP2 est chargé avec
les 1 000 noms de classes ImageNet-1k déjà définis.

SigLIP attribue un score indépendant à chaque classe\u00a0:
`logit = scale * (image . text) + bias`. Par défaut, cet ensemble de logits
passe malgré tout par un softmax, ce qui produit une distribution mono-label
cohérente avec le comportement `top1`/`top5` de LibreCLIP. Le passage de
`multi_label=True` à `set_classes()` (ou lors de la construction) bascule au
contraire vers des probabilités sigmoïdes indépendantes. Plusieurs classes,
ou aucune, peuvent alors obtenir un score élevé sur une même image. Le
tokenizer est un modèle SentencePiece multilingue (vocabulaire Gemma), les
noms de classes dans d'autres langues que l'anglais fonctionnent donc de la
même manière.

Avec `task="embed"`, la prédiction renvoie un vecteur d'image normalisé L2 par
entrée au lieu de probabilités de classes, et `embed_text()` renvoie des lignes
de texte normalisées dans le même espace vectoriel. Un simple produit scalaire
entre les deux correspond donc à la similarité cosinus. `iou` n'a d'effet sur
aucune des deux tâches\u00a0; il n'y a aucune étape de NMS. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Valider

`val()` lit les noms des dossiers de classes dans un sous-ensemble `train/`
ImageFolder, appelle `set_classes()` avec ces noms, puis mesure l'exactitude
zero-shot top-1 et top-5 avec un score softmax. L'exactitude dépend de la
manière dont les noms de classes sont interprétés comme prompts, et non d'une
mise à jour des poids, puisqu'il n'y a rien à entraîner. La validation couvre
uniquement `task="classify"`\u00a0; `task="embed"` n'a pas de validateur de dataset.

<code-tabs name="val" />

## Exporter

<export-matrix />

L'export fige l'état actuel du modèle dans un graphe fixe. Pour
`task="classify"`, les dernières étiquettes définies par `set_classes()` et la
résolution au moment de l'export sont incorporées dans une couche linéaire
finale avec le facteur d'échelle et le biais appris. Le graphe exporté est donc
un classificateur d'images `[B, K]` ordinaire, dépourvu de tour de texte et de
tokenizer\u00a0; exportez-le de nouveau après avoir modifié les classes ou la
taille. L'export en mode `multi_label=True` n'est pas implémenté\u00a0; repassez-le
d'abord à `False`. L'export avec `task="embed"` trace uniquement la tour
d'image. Les deux nécessitent l'opset ONNX 14 ou ultérieur, que l'exporteur
définit par défaut.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille. Tous deux sont convertis
à partir des checkpoints Apache-2.0 de Google `siglip2-base-patch16-256` et
`siglip2-so400m-patch14-384`, et non à partir d'un entraînement COCO.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />

