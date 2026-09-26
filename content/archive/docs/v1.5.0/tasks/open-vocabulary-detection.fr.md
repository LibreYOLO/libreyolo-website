---
title: Détection à vocabulaire ouvert
seo_title: Détection à vocabulaire ouvert dans LibreYOLO
description: >-
  Détecter des objets à partir d'un vocabulaire textuel dans LibreYOLO. Charger
  Grounding DINO, OWLv2, OMDet-Turbo ou OV-DEIM par LibreOpenVocab et définir
  les classes à l'exécution.
lead: >-
  La détection à vocabulaire ouvert remplace la liste fixe des classes d'un
  checkpoint par des mots que vous choisissez au moment de l'appel. Dans
  LibreYOLO, ce n'est pas une tâche distincte : il s'agit de la tâche detect
  proposée par un niveau de modèles séparé, chargé avec la fabrique
  LibreOpenVocab au lieu de LibreYOLO.
keywords:
  - détection vocabulaire ouvert
  - détection objets zero shot
  - détection ensemble ouvert
  - grounding dino python
  - owlv2
  - omdet turbo
  - détection prompt texte
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        print(result.names)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Changer de vocabulaire
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("owlv2-b16")


        # set_classes est persistant : la valeur reste jusqu'à l'appel suivant.

        # Les étiquettes doivent être uniques après passage en minuscules et
        retrait des articles.

        model.set_classes(["a red backpack", "traffic cone"])

        result = model.predict(SAMPLE_IMAGE)


        model.set_classes(["bicycle wheel"])

        result = model.predict(SAMPLE_IMAGE)
    - label: Seuil textuel de Grounding DINO
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf filtre selon le score de la boîte, text_threshold selon le score
        # des tokens de la phrase décodée. Tous deux valent 0.25 par défaut
        # lorsqu'ils sont omis. Seul Grounding DINO accepte text_threshold.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
source_hash: 17197cf4d80f3d6f
---

## Définition

La détection à vocabulaire ouvert renvoie des `Results` de détection ordinaires :
boîtes, confiances et indices de classes, avec `result.names` qui associe ces
indices aux chaînes demandées. Seule l'origine de la liste des classes change.
Un détecteur conventionnel est entraîné sur un ensemble fixe de catégories et
ne peut jamais produire une catégorie extérieure à cet ensemble. Ces modèles
reçoivent le vocabulaire sous forme de texte pendant l'inférence. Il suffit donc
d'appeler `set_classes(["forklift", "safety cone"])` pour en faire les classes.

LibreYOLO ne possède aucune clé de tâche `open-vocabulary`. Ces modèles
déclarent `SUPPORTED_TASKS = ("detect",)` comme tout autre détecteur. Leur
parcours de chargement les distingue : ce sont des snapshots Hugging Face
plutôt que des checkpoints de dictionnaires d'état LibreYOLO. Ils restent donc
en dehors de la fabrique `LibreYOLO()` et sont construits par
`LibreOpenVocab()`. Cette fabrique est parallèle à `LibreSAM()` et
`LibreVLM()`, et ne remplace pas `LibreYOLO()`.

Les scores sont de véritables scores de détection, et non une légende générée
puis analysée. Chaque famille compare les régions de l'image à l'embedding
textuel de chaque prompt.

## Modèles

Ce niveau comprend quatre familles, toutes limitées à la prédiction. Chargez
n'importe laquelle par son alias avec `LibreOpenVocab`.

[Grounding DINO](/docs/models/grounding-dino), d'IDEA Research, dans les tailles
`t` et `b`. Il s'agit de la famille par défaut du niveau et de la seule qui
accepte `text_threshold`, un second seuil appliqué au score des tokens de la
phrase décodée.

[OWLv2](/docs/models/owlv2), de Google Research, dans les tailles `b16` et
`l14`. Il compare les régions de l'image à des embeddings textuels issus d'un
encodeur de type CLIP.

[OMDet-Turbo](/docs/models/omdet-turbo), d'Om AI Lab, dans une seule taille
`t`. Il dissocie les embeddings de classes d'un prompt de tâche linguistique.
C'est la seule famille de cette page qui supprime les boîtes qui se chevauchent
pendant son propre post-traitement. L'argument `iou=` est donc respecté.

[OV-DEIM](/docs/models/ov-deim), dans les tailles `s`, `m` et `l`, est un
détecteur de type DETR qui associe les requêtes du décodeur aux embeddings
textuels d'une tour MobileCLIP incluse. Cette association biunivoque avec
sélection top-K n'applique aucune NMS.

Les poids d'OV-DEIM constituent le cas restreint de ce niveau. Les poids du
détecteur sont sous licence CC BY-NC 4.0 et interdisent l'usage commercial. La
tour de texte incluse est soumise à la licence Machine Learning Research Model
d'Apple, qui n'autorise que la recherche. Le checkpoint `l` ajoute un
fine-tuning du backbone DINOv3-S sous la licence DINOv3 de Meta. Les trois
textes de licence sont fournis dans le dépôt des poids, et la bibliothèque
journalise le même résumé lorsqu'elle résout les poids, avant de construire le
modèle. Lisez la page [OV-DEIM](/docs/models/ov-deim) avant de le déployer.

Ce niveau nécessite un extra :

```bash
pip install "libreyolo[openvocab]"
```

Il couvre `transformers` et `timm` pour les trois familles encapsulées, ainsi
que les paquets `huggingface_hub`, `safetensors`, `regex` et `ftfy`
nécessaires au portage natif d'OV-DEIM.

Un second niveau accepte également un vocabulaire textuel. `LibreVLM()` charge
des modèles génératifs vision-langage, tels que
[Qwen3-VL](/docs/models/qwen3-vl) et [Florence-2](/docs/models/florence-2), puis
convertit leur sortie dans les mêmes `Results`. Il partage l'interface
`set_classes()`. La différence tient à la production des boîtes : les familles
de cette page sont des détecteurs discriminatifs qui émettent directement des
scores, tandis que le niveau VLM les génère.

## Prédire

<code-tabs name="predict" />

`set_classes()` reçoit une liste non vide de chaînes d'étiquettes et la conserve
jusqu'à l'appel suivant. Les étiquettes doivent être uniques après leur passage
en minuscules et la suppression des articles initiaux. `"a bus"` et `"bus"` ne
peuvent donc pas coexister dans un même vocabulaire. Les expressions de
plusieurs mots sont des étiquettes comme les autres. Chaque famille transforme
la liste en sa propre entrée textuelle avant la tokenisation. `"traffic cone"`
constitue donc une requête différente de `"cone"`.

Trois arguments de prédiction se comportent ici différemment de ceux d'un
détecteur natif. `imgsz=` est refusé, car le processeur gère le
redimensionnement de ces familles. `augment=True` est refusé, car l'augmentation
au moment du test sort du périmètre de ce niveau. `iou=` ne s'applique qu'à la
famille dont le processeur effectue sa propre suppression. Lorsqu'aucune
suppression n'a lieu, sa transmission produit un avertissement et l'argument
est ignoré.

Lorsque `conf` est omis, il prend la valeur par défaut propre à la famille
chargée plutôt que la valeur habituelle de 0,25 de `predict()`. Cette valeur
varie au sein du niveau. Définissez-la explicitement lorsque vous comparez deux
familles sur une même image.

`track()` déclenche une erreur pour tout ce niveau. Exécutez plutôt `predict()`
sur chaque image. Consultez la page [prédiction](/docs/predict) pour les
sources, le streaming et la gestion des résultats.

## Entraîner

Aucune famille de ce niveau ne s'entraîne dans LibreYOLO. `train()` déclenche
une erreur. Effectuez le fine-tuning dans le projet amont, puis chargez les
poids obtenus. Le vocabulaire transmis à `set_classes()` est le seul réglage
qui change ce qu'un modèle chargé détecte.

## Valider

Ce niveau ne possède aucun validateur et `val()` déclenche une erreur. La
validation à vocabulaire ouvert exige un validateur dédié, car le validateur de
détection standard transmet directement les tenseurs d'images au modèle, tandis
que ces familles ont besoin d'entrées conditionnées par du texte, construites
en parallèle.

## Exporter

L'exportation sort du périmètre de ce niveau et `export()` déclenche une erreur.
Ces modèles s'exécutent avec `predict()` dans PyTorch.
