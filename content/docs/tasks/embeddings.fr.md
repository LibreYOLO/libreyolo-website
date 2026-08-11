---
title: Embeddings
seo_title: Embeddings d'images et de régions dans LibreYOLO
description: >-
  La tâche embed renvoie des vecteurs float32 normalisés L2 pour une image
  entière, pour chaque région détectée ou pour du texte. Enregistrez une
  galerie, établissez les correspondances par similarité cosinus et effectuez
  des recherches depuis Python ou la CLI.
lead: >-
  Une seule tâche couvre tous les vecteurs produits par LibreYOLO. embed renvoie
  des lignes float32 de longueur unitaire dont le produit scalaire est un score
  de similarité, que la ligne décrive une image entière, un seul visage détecté
  ou une ligne de texte, et la même Gallery les compare toutes.
keywords:
  - embeddings image python
  - embedding normalisé l2
  - recherche similarité cosinus
  - tâche embed libreyolo
  - recherche par image
  - enregistrer galerie embeddings
  - embeddings clip
  - embeddings dinov2
  - embeddings reid
last_verified: 1.5.0
verification: >-
  Clé de tâche et alias lus dans libreyolo/tasks.py. Données de résultat issues
  des classes Embeddings et Identities dans libreyolo/utils/results.py. API
  Gallery issue de libreyolo/utils/gallery.py. embed et _postprocess_embeddings
  issus de libreyolo/models/base/model.py. Familles prises en charge localisées
  en recherchant embed dans SUPPORTED_TASKS sous libreyolo/models/**/model.py.
  Interface CLI issue de libreyolo/cli/__init__.py,
  libreyolo/cli/commands/special.py et libreyolo/cli/commands/predict.py.
  Intention de conception issue de docs/adr/0015-embed-generalization.md.
meta:
  - label: Clé de tâche
    value: embed
    mono: true
  - label: Alias
    value: 'face-recognition, reid, face'
    mono: true
  - label: Données de résultat
    value: 'Embeddings, Identities'
    mono: true
  - label: Type des lignes
    value: 'float32, longueur unitaire'
snippets:
  predict:
    - label: Image entière
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # CLIP utilise classify par défaut, demandez donc explicitement le
        vecteur.

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        result = model(SAMPLE_IMAGE)


        print(result.embeddings.data.shape)  # (1, 512), une ligne par image

        print(result.boxes)                  # None : aucune localisation
    - label: Par région
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        # La ligne i décrit la région de la boîte i.
        print(result.boxes.xyxy.shape)       # (N, 4)
        print(result.embeddings.data.shape)  # (N, 512)
    - label: Plusieurs images à la fois
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Toutes les lignes de chaque résultat, concaténées en un tenseur.
        vectors = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(vectors.shape)  # (3, 384)
    - label: Texte
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        # Le texte est une méthode, jamais une source de prédiction. Une chaîne
        # passée à model(...) reste un chemin ou une URL.
        text = model.embed_text(["a photo of a cat", "a photo of a dog"])
        print(text.shape)  # (2, 512)
  similarity:
    - label: Comparer deux ensembles de lignes
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")


        query = model.embed("query.jpg")          # (1, 512)

        pool = model.embed(["a.jpg", "b.jpg"])    # (2, 512)


        # Les lignes sont unitaires, la similarité cosinus est donc un produit
        scalaire.

        scores = model("query.jpg").embeddings.similarity(pool)

        print(scores.shape)  # (1, 2)
    - label: Image comparée au texte
      language: python
      code: |
        import torch

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image = model.embed("photo.jpg")                       # (1, 512)
        text = model.embed_text(["a cat", "a dog", "a car"])   # (3, 512)

        print(torch.matmul(image, text.T))
  gallery:
    - label: Enregistrer et identifier
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("refs.npz")

        result = model("group.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # le nom vaut None sous le seuil
    - label: Recherche top-k
      language: python
      code: |
        from libreyolo import Gallery
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        gallery = Gallery.load("refs.npz", model=model)

        result = model("query.jpg")
        matches = gallery.match(result.embeddings, top_k=5, threshold=0.4)
        print(matches[0])   # [(nom, score), ...] pour la première ligne
    - label: Enregistrer un vecteur existant
      language: python
      code: |
        from libreyolo import Gallery

        gallery = Gallery()
        gallery.enroll_embedding("ada", vector)  # normalisé lors de l'ajout
        print(gallery.identities, gallery.dim, len(gallery))
  cli:
    - label: Enregistrer une arborescence de dossiers
      language: bash
      code: >
        # source/<identité>/*.jpg. Une galerie existante est étendue sur place.

        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=refs.npz
    - label: Identifier pendant la prédiction
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=group.jpg \
          gallery=refs.npz gallery_threshold=0.45
    - label: Comparer deux images
      language: bash
      code: >
        libreyolo compare model=librefacerec-l.onnx \
          source=a.jpg source2=b.jpg threshold=0.4

        # verify est la même commande sous un autre nom.

        libreyolo verify model=librefacerec-l.onnx source=a.jpg source2=b.jpg
        --json
source_hash: ffbaad5599035bc7
---

## Définition

`embed` transforme une image, une région d'image ou une chaîne en une ligne
float32 de largeur fixe dont la longueur vaut un. Comme chaque ligne est un
vecteur unitaire, la comparaison de deux lignes est un produit scalaire, et la
comparaison de deux ensembles tient en une multiplication matricielle. Rien
d'autre dans la tâche n'est propre à un modèle : la recherche par similarité,
la détection de doublons, la réidentification et la reconnaissance faciale
reposent toutes sur le même calcul appliqué à des lignes différentes.

Le vecteur est la sortie. Il n'existe aucune liste de classes, donc un nom est
associé ultérieurement par comparaison avec les références que vous fournissez,
et non par un mécanisme que le réseau aurait été entraîné à prédire.

### Trois formes

| Forme | `Results.embeddings` | `Results.boxes` | Produite par |
|---|---|---|---|
| Image entière | `(1, D)` | `None` | Passage d'une image à une famille qui traite l'image entière |
| Région | `(N, D)` | `(N, 4)`, lignes alignées | Familles qui localisent d'abord, comme la reconnaissance faciale |
| Texte | pas un `Results` | | `model.embed_text(texts)`, qui renvoie `(M, D)` |

Le résultat d'une image entière reste bidimensionnel même pour une seule image.
`(D,)` n'est pas une forme de retour autorisée, si bien qu'un consommateur n'a
jamais à traiter le cas d'une ligne unique de façon particulière. Le texte
renvoie un tenseur simple plutôt qu'un `Results`, car une chaîne n'est pas une
source d'image : en passer une à `model(...)` désigne toujours un chemin ou une
URL, et la bibliothèque ne suppose jamais qu'une chaîne est de la prose.

La clé de tâche canonique est `embed`. `embedding`, `embeddings`,
`face-recognition`, `facial-recognition`, `recognition`, `face`, `faceid` et
`reid` sont tous normalisés vers cette clé, si bien que `task="reid"` et
`task="embed"` sélectionnent exactement la même chose.

## Modèles

Quatre familles prennent en charge cette tâche et se répartissent nettement
selon qu'elles localisent ou non quelque chose au préalable.

| Famille | Forme | Dimension | Prend aussi en charge |
|---|---|---|---|
| [LibreFaceRec](/docs/models/librefacerec) | Région, une ligne par visage détecté | 512 | Rien d'autre ; `embed` est sa seule tâche |
| [CLIP](/docs/models/clip) | Image entière, avec une tour de texte associée | 512 pour `b32` et `b16`, 768 pour `l14` | `classify`, qui reste sa tâche par défaut |
| [SigLIP 2](/docs/models/siglip2) | Image entière, avec une tour de texte associée | 768 pour `b16`, 1152 pour `so400m` | `classify`, qui reste sa tâche par défaut |
| [DINOv2](/docs/models/dinov2) | Image entière, image uniquement | 384 | `semantic`, `classify` |

CLIP et SigLIP 2 conservent `classify` comme tâche par défaut, il faut donc
demander `task="embed"`. Leur checkpoint `-cls` existant est l'artefact partagé
à deux tours ; aucun checkpoint `-embed` en double n'est publié pour des poids
identiques.

`embed_text` n'existe que sur CLIP et SigLIP 2, les deux familles dotées d'une
tour de texte. DINOv2 n'en possède pas. L'embedding DINOv2 contourne les têtes
de segmentation sémantique et de classification et lit le token CLS final
normalisé à 224 pixels ; les variantes `n`, `s`, `m` et `l` utilisent toutes
l'encodeur DINOv2-S, elles renvoient donc toutes `D = 384`.

Les backbones réservés à la classification ajoutés dans cette version,
[ViT](/docs/models/vit), [Swin](/docs/models/swin) et [DeiT](/docs/models/deit),
déclarent uniquement `classify` et ne prennent pas cette tâche en charge.

<code-tabs name="predict" />

`model.embed(source, **kwargs)` est le raccourci pour les batchs : il exécute
`predict` et concatène chaque ligne de chaque résultat en un seul tenseur
float32 CPU `(N_total, D)`, avec une erreur si les lignes ont des dimensions
différentes. Une famille dont les tâches prises en charge ne contiennent pas
`embed` lève `NotImplementedError`.

## Données de résultat

`result.embeddings` est une donnée `Embeddings`. Son champ `data` est toujours
un float32 `(N, D)`, déjà normalisé L2 par le chemin d'inférence, et une entrée
qui n'est pas bidimensionnelle provoque une erreur au lieu d'être remodelée
silencieusement.

| Membre | Signification |
|---|---|
| `.data` | La matrice `(N, D)` |
| `.dim` | `D` |
| `.normalized` | Les mêmes lignes, renormalisées par sécurité |
| `.similarity(other)` | `(N, M)` face à un autre ensemble, ou `(N,)` face à un seul vecteur `(D,)` |
| `.verify(i, j, threshold=0.4)` | Indique si les lignes `i` et `j` représentent le même sujet |

`result.identities` est une donnée `Identities`, présente uniquement lorsqu'une
galerie a été fournie. C'est un conteneur simple, pas un tenseur, si bien que le
déplacement d'un `Results` entre appareils ne le modifie pas.

| Membre | Signification |
|---|---|
| `.name` | Liste des noms, avec `None` quand aucun n'a dépassé le seuil |
| `.score` | Meilleur score cosinus float32 `(N,)`, conservé même lorsque le nom vaut `None` |
| `.data` | Liste de tuples `(name, score)` |

<code-tabs name="similarity" />

Par défaut, les vecteurs sont exclus de `summary()` et `to_json()`, car une
ligne de 512 nombres float représente environ deux kilo-octets par sujet.
Chaque ligne indique plutôt `embedding_dim`, ainsi que `identity` et
`identity_score` lorsqu'une galerie a été utilisée. Passez
`summary(embeddings=True)` pour inclure les nombres.

## Galeries

Une `Gallery` est un ensemble nommé de lignes de référence. Elle stocke chaque
référence séparément au lieu d'en calculer la moyenne, si bien que le score d'un
nom correspond à sa meilleure référence unique et que l'ajout d'une mauvaise
photo ne déplace pas le centroïde d'une identité.

<code-tabs name="gallery" />

`Gallery(model)` se lie aux poids qui produiront ses vecteurs.
`enroll(name, sources, select="best")` exécute la prédiction sur chaque source
et conserve la ligne au score de confiance le plus élevé pour chaque résultat ;
`select="all"` conserve plutôt toutes les lignes, ce qui convient lorsqu'une
image de référence contient légitimement plusieurs sujets.
`enroll_embedding(name, vector)` ignore l'inférence et accepte directement un
vecteur, en le normalisant et en rejetant une ligne entièrement nulle.

`FaceGallery` est un alias permanent de la même classe, et les archives écrites
par les anciennes versions réservées aux visages se chargent toujours.

### Correspondances et seuils

La recherche de correspondances effectue une multiplication matricielle dense
avec chaque référence stockée, puis réduit le résultat à un score par nom en
prenant le maximum. Il n'existe aucun index approximatif, ce qui préserve
l'exactitude des nombres et impose une limite pratique à la taille de la
galerie.

Les deux points d'entrée diffèrent par leur comportement sous le seuil.
`match()` renvoie `[(name, score), ...]` par ligne en écartant tout ce qui est
sous le seuil, de sorte qu'une ligne sans correspondance produit une liste
vide. `identify()` renvoie une donnée `Identities` qui conserve toujours le
meilleur score et fixe le nom à `None` lorsqu'il est inférieur au seuil. Aucun
des deux ne remplace jamais le nom par le plus proche sous le seuil.

Le seuil par défaut est `0.4` partout. Il s'agit d'une valeur cosinus, pas d'une
probabilité, et le bon point de fonctionnement dépend de vos données et de votre
tolérance aux fausses correspondances. Balayez-le donc sur des paires étiquetées
au lieu d'accepter la valeur par défaut. `libreyolo enroll` et l'argument de
prédiction `gallery=` utilisent la même valeur.

### Persistance

`save(path)` écrit un fichier `.npz` compressé qui contient les vecteurs, les
noms et un bloc de métadonnées avec la version du format, la dimension de
l'embedding et une empreinte des poids ayant produit les lignes.
`Gallery.load(path, model=...)` vérifie les deux avant toute comparaison, de
sorte que le pointage d'une galerie vers un autre modèle provoque une erreur au
lieu de comparer silencieusement des vecteurs issus de deux espaces sans
rapport. L'enregistrement d'une galerie vide est refusé.

## Ligne de commande

| Commande | Fonction |
|---|---|
| `libreyolo enroll` | Parcourir une arborescence avec un dossier par identité et écrire ou étendre une galerie `.npz` |
| `libreyolo compare` | Calculer l'embedding du sujet principal dans deux images et indiquer la similarité cosinus |
| `libreyolo verify` | La même commande sous un autre nom |
| `libreyolo predict gallery=...` | Associer des identités à une exécution de prédiction ordinaire |

<code-tabs name="cli" />

Toutes les commandes LibreYOLO acceptent à la fois `key=value` et `--key value`,
si bien que `gallery=refs.npz` et `--gallery refs.npz` désignent le même
argument.

`enroll` accepte `model`, `source` et `gallery`, ainsi que les options
`face-detector`, `device`, `--json` et `--quiet`. La commande lit un dossier par
identité, dont le nom est l'identité, et chaque image qu'il contient contribue
aux références :

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

Une image qui ne produit rien est ignorée avec une ligne sur stderr au lieu
d'interrompre l'exécution, et le récapitulatif indique le nombre de références
stockées pour chaque nom. Un fichier de galerie existant est étendu sur place,
ce qui permet d'ajouter des identités au fil du temps.

`compare` et `verify` correspondent à une seule fonction enregistrée deux fois.
Elles acceptent `model`, `source`, `source2` et un `threshold` facultatif, puis
affichent la similarité cosinus, le verdict identique ou différent et le seuil
qui l'a produit. `--json` affiche les mêmes trois champs sous forme d'objet.

Avec `predict`, `gallery` pointe vers un fichier `.npz` enregistré et
`gallery_threshold` remplace la valeur par défaut `0.4`. Fournir une galerie à
un modèle dont la tâche n'est pas `embed` est une erreur plutôt qu'une opération
silencieuse, et si le fichier de galerie est absent, le message suggère la
commande `libreyolo enroll` qui le créerait.

## Visages

La reconnaissance faciale constitue la forme régionale de cette tâche et sa
seule implémentation fournie. Elle ajoute une étape de détection et d'alignement
avant la tête d'embedding, ainsi qu'une méthode `verify()`, un argument pour
fournir vos propres boîtes, des valeurs d'exactitude publiées et des conseils
d'étalonnage du seuil. Tout cela figure dans la page sur la
[reconnaissance faciale](/docs/tasks/face-recognition), qui est le guide à
suivre lorsque les sujets sont des visages. Tout le contenu de cette page s'y
applique sans modification.

## Entraîner, valider et exporter

Rien dans cette tâche ne s'entraîne dans LibreYOLO. La tête d'embedding facial
est un artefact ONNX dont les méthodes `train()`, `val()` et `export()` lèvent
toutes une erreur ; entraînez une tête en amont et chargez le fichier par son
chemin. CLIP, SigLIP 2 et DINOv2 s'entraînent et s'exportent par leurs tâches de
classification et de segmentation, pas par `embed`.

Il n'existe aucun validateur de recherche par similarité. Mesurez l'exactitude
de vérification sur des paires étiquetées en balayant `threshold`, et
l'exactitude d'identification en enregistrant une galerie puis en lisant
`identities.name` et `identities.score` sur des images de test, un nom `None`
étant compté comme un rejet.
