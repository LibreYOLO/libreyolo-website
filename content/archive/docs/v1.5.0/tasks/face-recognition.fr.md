---
title: Reconnaissance faciale
seo_title: Reconnaissance faciale dans LibreYOLO
description: >-
  Détecter, encoder et identifier des visages dans LibreYOLO. Enregistrer une
  galerie, comparer deux images et effectuer une correspondance par similarité
  cosinus, depuis Python ou la CLI.
lead: >-
  La reconnaissance faciale est la tâche embed appliquée aux visages. Un
  détecteur localise et aligne chaque visage, une tête de reconnaissance renvoie
  un vecteur normalisé en L2 par visage, puis l'identité est déterminée par
  similarité cosinus avec des références enregistrées plutôt que par une liste
  de classes fixe.
keywords:
  - reconnaissance faciale python
  - embedding visage
  - vérification visage
  - galerie visages
  - arcface onnx
  - tâche embed libreyolo
  - similarité cosinus visages
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Les noms librefacerec-* sont dirigés vers la famille d'embeddings de

        # visages quel que soit le suffixe du fichier. À la première
        utilisation,

        # ils se téléchargent avec le détecteur par défaut depuis Hugging Face.

        model = LibreYOLO("librefacerec-l.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)             # boîtes de visages (N, 4)

        print(result.embeddings.data.shape)  # (N, D), une ligne par visage

        print(result.embeddings.dim)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=photo.jpg
    - label: Comparer deux images
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("librefacerec-l.onnx")


        # Exécute la détection et l'embedding sur les deux images, puis compare

        # leur visage le plus fiable. La similarité cosinus se situe dans [-1,
        1].

        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)

        print(outcome["similarity"], outcome["same_person"])
    - label: Enregistrer une galerie et identifier
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("faces.npz")

        result = model("group_photo.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # name vaut None sous le seuil
    - label: Enregistrer et identifier depuis la CLI
      language: bash
      code: >
        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=faces.npz

        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg
        gallery=faces.npz
    - label: Fournir vos propres boîtes de visages
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("librefacerec-l.onnx")


        # face_boxes ignore entièrement la détection ; face_detector accepte une

        # fonction, un modèle de détection LibreYOLO ou une instance de
        FaceDetector.

        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        print(result.embeddings.data.shape)
source_hash: d7dfcb6f812ebb2d
---

## Définition

La reconnaissance faciale renvoie un vecteur par visage, et non une étiquette.
La prédiction comporte deux étapes : un détecteur localise chaque visage et ses
cinq repères, le recadrage est déformé vers un alignement canonique 112 x 112,
puis une tête de reconnaissance émet un embedding normalisé en L2.

`result.embeddings` est une charge utile `Embeddings` de forme `(N, D)`, dont
les lignes sont alignées avec `result.boxes`. La ligne `i` décrit donc le visage
de la boîte `i`. Comme les lignes sont des vecteurs unitaires, la similarité
cosinus est un produit scalaire. `embeddings.similarity()` la calcule en un
appel avec un autre `Embeddings` ou avec une matrice entière.

Nommer un visage constitue une étape distincte. Une `Gallery` contient des
vecteurs de référence nommés. Transmettre `gallery=` à `predict()` ajoute
`result.identities`, aligné par ligne avec les embeddings et contenant pour
chaque visage un nom et son meilleur score cosinus. Le nom d'un visage sous le
seuil de correspondance reste `None`. Le nom le plus proche mais inférieur au
seuil n'est jamais substitué.

La clé de tâche canonique de la bibliothèque est `embed`.
`face-recognition`, `facial-recognition`, `reid` et `face` sont tous normalisés
vers celle-ci. `task="face-recognition"` et `task="embed"` sélectionnent donc
la même tâche. Les visages constituent la forme régionale de cette tâche plus
large. La page sur les [embeddings](/docs/tasks/embeddings) couvre les formes
d'image entière et de texte, l'API partagée `Embeddings`, `Identities` et
`Gallery`, ainsi que les modèles qui produisent des vecteurs sans rien détecter.

## Modèles

[LibreFaceRec](/docs/models/librefacerec) est la famille de cette tâche. Elle
réunit deux artefacts ONNX derrière un même appel :
`librefacerec-l.onnx`, une tête de reconnaissance iResNet100 qui produit des
embeddings de dimension 512, et `librefacerec-det.onnx`, le détecteur de visages
par défaut à cinq repères, issu d'OpenCV Zoo. Tous deux sont téléchargés depuis
l'organisation LibreYOLO sur Hugging Face à la première utilisation. Tout autre
fichier ONNX conforme à la convention ArcFace (entrée alignée 112 x 112, sortie
`(N, D)`) peut remplacer la tête de reconnaissance. Transmettez son chemin au
lieu d'un nom `librefacerec-*`.

La clé de tâche `embed` dépasse le cadre des visages.
[CLIP](/docs/models/clip), [SigLIP2](/docs/models/siglip2) et
[DINOv2](/docs/models/dinov2) prennent aussi en charge `task="embed"` et
renvoient un vecteur pour l'image entière. Il s'agit alors de recherche
d'images, pas d'identité faciale. Ces modèles partagent l'API `Gallery` et
`Embeddings`, ce qui permet de réutiliser le workflow d'enregistrement et de
mise en correspondance ci-dessous, mais ils ne détectent ni n'alignent les
visages.

La tête de reconnaissance s'exécute avec `onnxruntime`, absent de l'installation
de base :

```bash
pip install "libreyolo[onnx]"
```

## Prédire

<code-tabs name="predict" />

Sans autre réglage, `predict()` télécharge et associe le détecteur par défaut.
`face_detector` le remplace par une fonction, un modèle de détection LibreYOLO
ou une instance de `FaceDetector`. Il peut être défini sur le constructeur ou
pour chaque appel. `face_boxes` évite la détection en fournissant des boîtes que
vous possédez déjà. Dans la CLI, `face_detector=` accepte le chemin d'un
détecteur de visages `.onnx` ou le nom d'un détecteur LibreYOLO.

`model.verify(image_a, image_b)` est le raccourci pour deux images. Il encode le
visage le plus fiable de chacune et renvoie
`{"similarity", "same_person", "threshold"}`. `model.embed(sources)` renvoie
toutes les lignes de visages d'une ou plusieurs images, empilées dans un tenseur
unique `(N_total, D)`. Consultez la page [prédiction](/docs/predict) pour les
sources, le streaming et la gestion des résultats.

## Format du dataset

L'enregistrement lit un dossier par identité. Le nom du dossier devient
l'identité, et chaque image qu'il contient ajoute des références sous ce nom :

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

`libreyolo enroll` parcourt cette arborescence et écrit une galerie `.npz`. Un
fichier de galerie existant est étendu sur place plutôt que remplacé, ce qui
permet d'ajouter des identités au fil du temps. Les galeries sont liées aux
poids qui les ont produites par la dimension des embeddings et une empreinte du
fichier. Une mise en correspondance avec un autre modèle déclenche une erreur
au lieu de comparer des espaces vectoriels incompatibles.

Par défaut, chaque image source contribue une ligne de référence, celle du
visage le plus fiable. Un portrait contenant des personnes à l'arrière-plan
n'enregistre donc que son sujet. Transmettez `select="all"` à `Gallery.enroll`
pour conserver chaque ligne renvoyée.

## Entraîner

Aucune famille de cette tâche ne s'entraîne dans LibreYOLO.
`LibreFaceEmbedder.train()` déclenche une erreur. Entraînez une tête de
reconnaissance dans le projet amont, exportez-la vers ONNX conformément à la
convention ArcFace, puis chargez le fichier par son chemin.

## Valider

Cette tâche ne possède aucun validateur de dataset et `val()` déclenche une
erreur au lieu de prétendre le contraire. La précision de vérification se mesure
sur des paires d'images annotées avec `model.verify()`, en faisant varier
`threshold` pour choisir le point de fonctionnement souhaité. La précision
d'identification se mesure en enregistrant une galerie puis en lisant
`result.identities.name` et `result.identities.score` sur des images réservées.
Un nom `None` est compté comme un rejet.

## Exporter

La tête de reconnaissance est déjà un graphe ONNX. Il n'y a donc rien à
convertir et `LibreFaceEmbedder.export()` déclenche une erreur. Déployez
directement le fichier `.onnx`, ou transmettez-le à LibreYOLO et laissez la
famille gérer la détection, l'alignement et la normalisation.
