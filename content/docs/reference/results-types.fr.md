---
title: Types de résultats
seo_title: Référence de l'objet Results de LibreYOLO
description: "Toutes les charges utiles que peut contenir un objet Results LibreYOLO, avec un emplacement par forme de tâche\_: bounding boxes, masques, points clés, probabilités, OBB, profondeur, OCR, embeddings et dix autres."
lead: >-
  Results est l'unique type de retour par image de tous les modèles LibreYOLO.
  Il contient dix-huit emplacements de charges utiles facultatifs, un par forme
  de tâche, et ne remplit que ceux produits par le modèle.
keywords:
  - objet results libreyolo
  - Results.boxes
  - Results.masks
  - Results.probs
  - Results.depth_map
  - Results.summary
  - libreyolo results to_json
last_verified: 1.5.0
verification: >-
  Noms des emplacements, formes, propriétés et valeurs par défaut lus dans
  libreyolo/utils/results.py en v1.5.0. Sémantique citée depuis les docstrings
  des classes de charges utiles.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape, result.path)
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.names[int(result.boxes.cls[0])])
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        # Toutes les charges utiles sont déplacées ensemble.
        result = result.cpu().numpy()

        # Lignes sous forme de dictionnaires simples, puis en JSON.
        print(result.summary()[:1])
        print(result.to_json())
source_hash: 16f654364ae6448a
---

## Objet Results

Un objet `Results` décrit une image. Une source image unique en renvoie un, une
source liste ou un répertoire en renvoie une liste, et `stream=True` renvoie un
générateur qui les produit.

| Attribut | Type | Signification |
|---|---|---|
| `orig_shape` | `(int, int)` | Hauteur et largeur de l'image d'origine |
| `path` | `str` | Chemin source lorsque l'entrée provient du disque |
| `names` | `dict[int, str]` | Association des indices de classes aux noms de classes |
| `speed` | `dict[str, float]` | Millisecondes par étape |
| `track_id` | tensor | Identifiants de suivi lorsque le résultat provient de `track()` |
| `frame_idx` | `int` | Indice de l'image pour les sources vidéo et flux |
| `restore_scale` | `int` | Facteur d'agrandissement entre sortie et entrée d'un résultat de restauration\u00a0; `1` partout ailleurs |

<code-tabs name="usage" />

## Emplacements des charges utiles

Chaque emplacement vaut `None` si le modèle ne l'a pas produit. La tâche de la
famille détermine l'emplacement qu'elle remplit.

| Emplacement | Classe | Tâche |
|---|---|---|
| `boxes` | `Boxes` | detect |
| `masks` | `Masks` | segment |
| `keypoints` | `Keypoints` | pose |
| `probs` | `Probs` | classify |
| `obb` | `OBB` | obb |
| `gaze` | `Gaze` | gaze |
| `points` | `Points` | point |
| `semantic_mask` | `SemanticMask` | semantic |
| `panoptic` | `PanopticSegmentation` | panoptic |
| `depth_map` | `DepthMap` | depth |
| `normal_map` | `NormalMap` | normal |
| `edges` | `EdgeMap` | edge |
| `restored` | `RestoredImage` | restore |
| `matte` | `Matte` | matte |
| `ocr` | `OCRRegions` | ocr |
| `embeddings` | `Embeddings` | embed |
| `identities` | `Identities` | embed, avec une galerie |
| `meshes` | `Meshes` | mesh |

`result.normals` est un alias en lecture-écriture de `result.normal_map`.

Plusieurs emplacements peuvent être définis simultanément. Un modèle de
segmentation remplit `boxes` et `masks`, un modèle de regard remplit `boxes`
avec les bounding boxes des visages et `gaze` avec les angles, et un modèle de
maillage remplit `boxes` avec les bounding boxes des personnes et `meshes`
aligné ligne par ligne avec celles-ci.

## Boxes

Bounding boxes de détection pour une image.

| Membre | Valeur renvoyée |
|---|---|
| `xyxy` | Coordonnées des coins en pixels de l'image d'origine |
| `xywh` | Centre et dimensions en pixels |
| `xyxyn` | Coins normalisés dans `[0, 1]` |
| `xywhn` | Centre et dimensions normalisés dans `[0, 1]` |
| `conf` | Confiance par bounding box |
| `cls` | Indice de classe par bounding box |
| `id` | Identifiant de suivi par bounding box, ou `None` |
| `is_track` | `True` lorsque les identifiants de suivi sont présents |
| `data` | Tenseur compacté |

`with_id(id)` et `with_orig_shape(orig_shape)` renvoient un nouvel objet
`Boxes` dans lequel ce champ est remplacé.

## Masks

Masques d'instances pour une image. `data` est le tenseur de masques, `xy`
renvoie les contours par instance en pixels et `xyn` les renvoie normalisés.

## Keypoints

Points clés de pose, alignés ligne par ligne avec `boxes`. `xy` est la paire de
coordonnées de chaque point clé et `xyn` la paire normalisée. `conf` est le
troisième canal lorsque les données en possèdent un, sinon `None`.
`has_visible` est un tableau booléen, true lorsque `conf > 0`, et entièrement
true en l'absence de canal de confiance.

## Points

Localisation de points pour une image. `data` a la forme `(N, 4)` et ses lignes
sont `x, y, class, confidence`. Les coordonnées sont des pixels absolus. `xy`,
`cls` et `conf` séparent les colonnes, tandis que `xyn` normalise les
coordonnées.

## Probs

Scores de classification. `top1` est l'indice gagnant, `top5` les cinq
meilleurs indices, et `top1conf` et `top5conf` leurs scores.

## OBB

Bounding boxes orientées. `data` contient 7 ou 8 valeurs par ligne\u00a0: `xywhr`,
un identifiant de suivi facultatif, puis la confiance et la classe.

| Membre | Valeur renvoyée |
|---|---|
| `xywhr` | Centre, dimensions et rotation en radians |
| `xyxyxyxy` | Quatre coins en pixels |
| `xyxyxyxyn` | Quatre coins normalisés |
| `xyxy` | Enveloppe alignée sur les axes en pixels |
| `conf`, `cls`, `id`, `is_track` | Comme sur `Boxes` |

## Gaze

Angles du regard par visage en radians, de forme `(N, 2)`, alignés ligne par
ligne avec les bounding boxes des visages dans `boxes`. La colonne 0 est le
tangage et la colonne 1 le lacet selon la convention L2CS\u00a0: un lacet positif
tourne le regard vers la gauche du sujet et un tangage positif le tourne vers
le bas. `pitch_deg` et `yaw_deg` convertissent en degrés, et `direction_3d`
renvoie le vecteur direction unitaire.

## SemanticMask

Carte sémantique dense, de forme `(H, W)`, contenant les identifiants de classe
entiers sur le canevas de l'image d'origine. `255` est la valeur à ignorer et
n'est jamais compté comme classe (`SemanticMask.IGNORE_INDEX`). `classes`
énumère les identifiants de classe présents et `class_mask(class_id)` renvoie
le masque booléen d'une classe.

## PanopticSegmentation

Chaque pixel reçoit exactement un segment sans chevauchement, ce qui unifie
les régions stuff et les instances thing. `data` est une carte entière
d'identifiants de segments `(H, W)`. L'identifiant `0` est sans étiquette
(`PanopticSegmentation.IGNORE_INDEX`). `segments_info` est une liste de
dictionnaires, un par segment, contenant chacun au moins
`{"id": int, "category_id": int}`, où `id` correspond à une valeur de la carte
et `category_id` indexe `names`. `segment_ids` énumère les identifiants présents
et `segment_mask(segment_id)` renvoie le masque booléen d'un segment.

La distinction thing ou stuff est une propriété de la catégorie, pas du
segment. Une charge utile peut la dénormaliser sur chaque segment sous la forme
`"isthing": bool`, auquel cas la valeur doit correspondre à la table au niveau
de la catégorie.

## DepthMap

Carte dense de profondeur inverse relative, de forme `(H, W)` et composée de
flottants sur le canevas de l'image d'origine. Les valeurs élevées signifient
une plus grande proximité avec la caméra. Les valeurs sont relatives et ne
représentent pas des mètres. `min`, `max` et `mean` sont calculés sur les
valeurs finies, et `normalized()` remet la carte à l'échelle dans `[0, 1]`.

## NormalMap

Champ dense de normales de surface, float32 `(H, W, 3)`, sur le canevas de
l'image d'origine dans le repère caméra OpenCV\u00a0: `+x` vers la droite, `+y`
vers le bas et `+z` dans la scène. Les normales font face à la caméra, une
surface fronto-parallèle vaut donc `(0, 0, -1)`. Chaque pixel est un vecteur
unitaire. `assert_normalized(atol=1e-4)` vérifie cet invariant.

## EdgeMap

Carte dense de probabilités des contours, float32 `(H, W)`, sur le canevas de
l'image d'origine, où `0` n'est pas un contour et `1` en est un. La carte
continue est conservée afin que le seuil reste au choix de l'appelant\u00a0:
`binary(threshold=0.5)` en applique un et `array` renvoie la vue numpy.

## RestoredImage

Image RGB restaurée, uint8 `(H, W, 3)`. Pour la super-résolution, le canevas
est `Results.restore_scale` fois plus grand que l'entrée. `array` renvoie la
vue numpy et `save(path)` écrit l'image.

## Matte

Matte d'opacité souple, float32 `(H, W)` dans `[0, 1]` sur le canevas de
l'image d'origine. `1` représente entièrement le premier plan et `0`
entièrement l'arrière-plan. Un matte souple englobe un masque dur de suppression
de l'arrière-plan seuillé à 0.5 et conserve les bords anti-crénelés qu'un
masque binaire supprime. `array` renvoie la vue numpy.

Sur un résultat de matting, `Results.cutout(image=None)` renvoie un tableau
RGBA uint8 `(H, W, 4)` dont le quatrième canal est le matte, et
`Results.save(path, image=None)` écrit cette découpe comme PNG à arrière-plan
transparent. Les deux utilisent le RGB de `image` s'il est fourni, sinon ils
le rechargent depuis `Results.path`.

## OCRRegions

Texte localisé avec transcriptions. `data` contient des polygones float
`(N, 4, 2)` en pixels de l'image d'origine, dans l'ordre haut-gauche,
haut-droit, bas-droit, bas-gauche. Les régions suivent l'ordre de lecture, du
haut vers le bas puis de gauche à droite. `texts` est la liste des N
transcriptions. `conf` est le score de reconnaissance par région et `det_conf`
le score de détection, tous deux de forme `(N,)`.

Les quadrilatères de détection sont de véritables polygones, ils ne remplissent
donc pas `Results.boxes`. `xyxy` donne leurs enveloppes alignées sur les axes.

## Embeddings

Vecteurs normalisés L2 de la tâche `embed`, toujours de forme `(N, D)`. Un
résultat portant sur l'image entière contient une ligne et aucune bounding box.
Les embeddings de régions sont alignés ligne par ligne avec `boxes`. Chaque
ligne étant normalisée, la similarité cosinus est un produit scalaire.

| Membre | Valeur renvoyée |
|---|---|
| `dim` | `D` |
| `normalized` | Lignes renormalisées |
| `similarity(other)` | Similarité cosinus par paires avec un autre objet `Embeddings` ou tenseur |
| `verify(i, j, threshold=0.4)` | `True` lorsque les lignes `i` et `j` correspondent |

## Identities

Correspondances nommées d'une galerie, alignées ligne par ligne avec
`embeddings`. Produites lorsqu'une `Gallery` est transmise à une prédiction
`embed`. `name` est une liste dont une entrée vaut `None` sous le seuil de
correspondance. Le nom le plus proche sous ce seuil n'est jamais deviné.
`score` est le tableau des scores de correspondance et `data` les associe.

## Meshes

Maillages paramétriques de corps humains, alignés ligne par ligne avec les
bounding boxes de personnes dans `boxes`. Tout est exprimé dans le repère
caméra de l'image d'origine. `transl` est métrique, en mètres, avec `+z`
orienté à l'opposé de la caméra. `vertices` et `joints3d` sont métriques et
incluent déjà `transl`. `joints2d` est exprimé en pixels sur le canevas de
l'image d'origine, et non sur le recadrage reçu par le réseau. Aucun champ ne
porte de repère mondial ou gravitationnel.

La disposition des paramètres diffère entre les modèles corporels, rien n'est
donc codé en dur concernant les formes. `body_model` nomme le paramétrage et
les nombres sont relus dans les tenseurs\u00a0: `num_vertices`, `num_joints`,
`num_betas` et `has_vertices`. `params` renvoie le dictionnaire de paramètres
et `save_obj(path, index=0)` écrit un maillage. Les champs sont
`global_orient`, `body_pose`, `betas`, `transl`, `vertices`, `faces`,
`joints3d`, `joints2d`, `conf`, `focal_length` et `extras`.

Pour `body_model="mhr"`, les rotations sont des angles d'Euler en radians
plutôt qu'un format axe-angle, `body_pose` est un vecteur plat de paramètres
par articulation plutôt qu'un triplet par articulation, et `betas` contient
les coefficients de blendshape d'identité. L'échelle du squelette, la pose des
mains et l'expression faciale se trouvent dans `extras`.

## Conversion et sélection

Chaque charge utile possède `to(*args, **kwargs)`, `cpu()`, `cuda()` et
`numpy()`. L'appel de l'une de ces méthodes sur `Results` l'applique en une
fois à tous les emplacements remplis.

<code-tabs name="convert" />

`result[idx]` sélectionne les lignes des charges utiles alignées.
`len(result)` est le nombre de détections, ou de points en l'absence de bounding
boxes. `result.update(...)` renvoie une copie dans laquelle les emplacements
nommés sont remplacés. Il accepte chaque emplacement ainsi que `track_id` et
`restore_scale`.

## summary et to_json

`summary(normalize=False, decimals=5, embeddings=False)` renvoie une liste de
dictionnaires simples, une ligne par détection, segment, point ou région selon
les emplacements définis. `to_json(**kwargs)` transmet ses arguments à
`summary` et renvoie la chaîne JSON.

`plot()` rend un résultat dense de normales ou de contours dans sa
visualisation canonique et lève une erreur pour les autres types. Les images
annotées des autres tâches sont produites par `predict(save=True)`.
