---
title: Segmentation guidée
seo_title: Segmentation guidée dans LibreYOLO
description: >-
  Transformer un point, une boîte ou un concept textuel en masque d'objet dans
  LibreYOLO. Charger SAM, SAM 2, SAM 3, EdgeTAM, MobileSAM ou PicoSAM3 avec
  LibreSAM.
lead: >-
  La segmentation guidée transforme un clic en masque : vous pointez un objet ou
  tracez une boîte autour de lui, et le modèle renvoie son contour. Dans
  LibreYOLO, ce n'est pas une clé de tâche distincte, mais un niveau de modèles
  chargé avec la fabrique LibreSAM, dont les résultats sont des Results de
  segmentation ordinaires.
keywords:
  - segmentation guidée
  - segmentation interactive
  - segment anything python
  - prompt point
  - prompt boîte
  - SAM python
  - masque depuis un clic
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompts par points et boîtes
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # Un point est [x, y] en pixels ; les étiquettes sont 1 pour positif, 0
        pour négatif.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # polygones

        print(result.boxes.xyxy)    # boîtes serrées dérivées des masques


        # Un prompt par boîte produit un masque par boîte.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: 'Encoder une fois, guider plusieurs fois'
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # set_image exécute une fois l'encodeur d'image lourd et le met en
        cache.

        model.set_image(SAMPLE_IMAGE)

        first = model.predict(points=[640, 420], labels=[1])

        second = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
    - label: Tout segmenter
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # Sans prompt, une grille de points couvre l'image. La grille par défaut

        # de 32 par côté représente environ 1024 passes du décodeur, lentes sur
        CPU.

        result = model.predict(SAMPLE_IMAGE, points_per_side=8)

        print(len(result.masks))
    - label: Masques d'ambiguïté
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # Un point peut désigner une manche, une chemise ou une personne.

        # multimask=True renvoie les trois masques tout ou partie au lieu du
        meilleur.

        result = model.predict(
            SAMPLE_IMAGE, points=[640, 420], labels=[1], multimask=True
        )

        print(len(result.masks))
source_hash: bb70ff24e6c0a767
---

## Définition

La segmentation guidée reçoit une image et un prompt spatial, puis renvoie le
masque de ce que désigne ce prompt. Rien n'est classé : il n'existe aucune
liste de classes et `result.boxes` contient des boîtes serrées dérivées des
masques, et non des détections à part entière. `result.masks` contient les
données des masques et `result.masks.xy` leurs polygones.

Le prompt constitue l'interface. `points` contient les coordonnées `[x, y]` en
pixels, à raison d'un ensemble par objet, et `labels` indique si chaque point
est positif (1, inclure ceci) ou négatif (0, exclure ceci). `bboxes` suit la
forme `[x1, y1, x2, y2]`, avec un masque par boîte. Les points et les boîtes
peuvent être combinés. Ils sont alors associés par objet et doivent avoir la
même longueur. L'absence de tout prompt exécute le parcours de segmentation
complète, une grille de points sur l'image.

Un point unique est ambigu par nature. Cliquer sur une manche peut désigner la
manche, la chemise ou la personne. `multimask=True` renvoie donc les trois
masques tout ou partie par prompt, au lieu du seul meilleur masque. `conf`
filtre selon l'IoU prédite par le modèle, un score de qualité du masque, et non
selon une confiance de détection.

LibreYOLO ne possède aucune clé de tâche `promptable`. Ce niveau s'enregistre
comme `segment`, la même clé que celle de la segmentation d'instances. La forme
de l'appel le distingue et explique sa propre fabrique, `LibreSAM()`, parallèle
à `LibreYOLO()`, `LibreOpenVocab()` et `LibreVLM()`. Une seule signature
`predict(image)` ne peut pas exprimer la boucle pour laquelle ces modèles sont
conçus. `set_image()` exécute une fois l'encodeur d'image et met les embeddings
en cache. Chaque appel ultérieur à `predict()` avec `source=None` ne paie que
le décodage du prompt, et `reset_image()` vide le cache. L'encodeur d'image
représente l'essentiel du coût et ne s'exécute qu'une fois par image. Un second
prompt sur la même image l'ignore donc entièrement.

## Modèles

Six familles se chargent par alias avec `LibreSAM`.

[SAM](/docs/models/sam) est la famille par défaut, dans les tailles `base`,
`large` et `huge`, également nommées `b`, `l` et `h`.

[SAM 2](/docs/models/sam-2), sous les noms `sam2-tiny`, `sam2-small`,
`sam2-base-plus` et `sam2-large`. LibreYOLO prend en charge son parcours image.

[SAM 3](/docs/models/sam-3), sous le nom `sam3`, est la seule famille qui
accepte un prompt de concept textuel. `text="yellow school bus"` renvoie toutes
les instances correspondantes. Transmettre `text=` à une autre famille
déclenche une erreur qui nomme SAM 3. Ses poids proviennent de Meta sous la
licence SAM personnalisée plutôt que sous la licence MIT de LibreYOLO, et
l'accès au dépôt est soumis à autorisation. Acceptez les conditions sur la page
du modèle et authentifiez-vous avec `hf auth login` avant le premier
téléchargement. Lisez la page [SAM 3](/docs/models/sam-3) avant de le déployer.

[EdgeTAM](/docs/models/edgetam), sous le nom `edgetam`, est une variante de
SAM 2 destinée aux appareils. LibreYOLO prend en charge son parcours image.

[MobileSAM](/docs/models/mobilesam), sous le nom `mobilesam`, remplace
l'encodeur ViT-H de SAM par un TinyViT distillé.

[PicoSAM3](/docs/models/picosam3), sous le nom `picosam3`, est un CNN compact
pour les régions guidées par des boîtes sur les capteurs en périphérie. Les
prompts par boîtes constituent ici l'intégralité du contrat. Les points, le
texte, les masques, le mode multimasque et la segmentation complète déclenchent
tous une erreur qui renvoie vers SAM 2 ou SAM 3.

L'extra de ce niveau couvre les quatre familles chargées par `transformers` :

```bash
pip install "libreyolo[sam]"
```

MobileSAM et PicoSAM3 sont des portages LibreYOLO natifs et ne nécessitent
aucune installation de `transformers` pour fonctionner.

## Prédire

<code-tabs name="predict" />

`source` et `set_image()` sont deux possibilités différentes, pas une
séquence. Transmettez une image à `predict()` pour un appel unique, ou appelez
d'abord `set_image()`, puis `predict(source=None)` pour chaque prompt.
Transmettre `device=` à `predict()` déplace le modèle pour cet appel et tous les
suivants, et invalide tout embedding mis en cache.

La segmentation complète est le mode coûteux. `points_per_side` vaut 32 par
défaut, soit environ 1 024 passes du décodeur sur l'image. Réduisez cette valeur
pour toute utilisation interactive sur CPU. Dans ce mode, un argument `conf`
omis applique le seuil de grille propre à la famille, tandis que le parcours
guidé conserve tous les masques si `conf` est omis. Transmettez `conf=0.0` pour
désactiver le filtrage dans les deux modes, et `max_det` pour limiter le nombre
de masques renvoyés.

Cette version ne prend pas en charge les prompts par masques. `masks=` déclenche
donc une erreur au lieu d'être ignoré. `track()` déclenche également une erreur
pour tout le niveau. Comme il s'agit de segmenteurs d'images, exécutez
`predict()` sur chaque image. Consultez la page [prédiction](/docs/predict) pour
les sources et la gestion des résultats.

## Entraîner

Aucune famille de ce niveau ne s'entraîne dans LibreYOLO. `train()` déclenche
une erreur. Effectuez le fine-tuning dans le projet amont, puis chargez les
poids obtenus.

## Valider

Ce niveau ne possède aucun validateur et `val()` déclenche une erreur. Un masque
guidé ne possède aucun ensemble fixe de classes auquel le comparer. Les
métriques habituelles de détection et de segmentation n'ont donc aucune clé sur
laquelle s'appuyer. Évaluer un masque guidé consiste à le comparer à un masque
de référence que vous fournissez, avec les prompts qui vous intéressent.

## Exporter

L'exportation sort du périmètre du niveau entier et `export()` déclenche une
erreur, à une exception près. [PicoSAM3](/docs/models/picosam3) exporte son CNN
de région brut 96 x 96 vers ONNX sous la forme
`roi_image -> mask_logits`. Le recadrage de la boîte et le redimensionnement du
masque vers les coordonnées de l'image restent en Python. Toutes les autres
familles s'exécutent avec `predict()` dans PyTorch. Consultez la page
[exportation](/docs/export) pour les formats disponibles ailleurs dans la
bibliothèque.
