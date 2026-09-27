---
title: Formats des datasets
seo_title: Formats de datasets LibreYOLO pour chaque tâche
description: "Contrat des fichiers de datasets par tâche canonique\_: clés YAML, arborescences de dossiers, lignes d'étiquettes, conventions des masques et cartes, et chargeur correspondant."
lead: >-
  Cette page reproduit le contrat des fichiers de datasets défini dans
  docs/dataset_schema.md de la bibliothèque. Elle couvre les clés YAML et
  l'organisation sur disque attendues par chaque tâche canonique.
keywords:
  - format dataset libreyolo
  - format étiquettes yolo
  - data.yaml
  - dataset masques segmentation
  - format coco panoptic
  - dataset profondeur
  - pose kpt_shape
last_verified: 1.5.0
verification: >-
  Reproduit docs/dataset_schema.md du dépôt libreyolo en v1.5.0, avec les noms
  des chargeurs recoupés dans libreyolo/data/.
snippets:
  usage:
    - label: Analyser une ligne d'étiquette de détection
      language: python
      code: >
        from libreyolo.data import parse_yolo_label_line


        # class_id cx cy w h, normalisés dans [0, 1]

        row = parse_yolo_label_line("0 0.5 0.5 0.25 0.5", 640, 480,
        num_classes=80)


        # (class_id, x1, y1, x2, y2, area) en pixels

        print(row)
source_hash: a8282c079624044d
---

## YAML commun

S'applique à `detect`, `segment`, `pose` et `obb`.

| Clé | Requise | Signification |
|---|---|---|
| `path` | | Racine du dataset |
| `train` | Pour l'entraînement | Images d'entraînement |
| `val` | Pour la validation | Images de validation |
| `test` | | Images de test |
| `names` | Oui | Liste de classes ou table indexée par des entiers |
| `nc` | | Nombre de classes\u00a0; doit correspondre à `names` si présent |
| `download` | | Instructions de téléchargement\u00a0; les scripts Python exigent une activation explicite |
| `annotations` | | Association des sous-ensembles à des fichiers JSON COCO natifs, pour detect, segment et obb |

`train`, `val` et `test` peuvent être des répertoires d'images, des fichiers
`.txt` qui énumèrent des images ou des listes de ces éléments. Les chemins des
étiquettes suivent une substitution\u00a0:

```text
images/.../image.jpg -> labels/.../image.txt
```

Pour un dataset JSON COCO natif, `annotations` associe chaque sous-ensemble à
son fichier JSON et le chemin du sous-ensemble indique la racine des images\u00a0:

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Lorsque `names` est présent, les noms de catégories du JSON COCO natif doivent
correspondre aux noms de classes du YAML, lesquels définissent les identifiants
d'étiquettes du modèle. Sans `names`, les identifiants de catégories COCO sont
triés et remappés de façon dense vers `0..N-1`.

Un YAML de dataset ne contient aucune clé `task`. La sélection explicite du
modèle et de la tâche l'emporte.

Règles communes à tous les fichiers d'étiquettes textuels\u00a0:

- un fichier d'étiquettes `.txt` par image\u00a0;
- un fichier d'étiquettes absent ou vide signifie qu'il n'y a aucun objet\u00a0;
- `class_id` est un entier dans `0..nc-1`\u00a0;
- les coordonnées sont des flottants finis normalisés dans `[0, 1]`\u00a0;
- les coordonnées sont relatives à la largeur et à la hauteur de l'image d'origine\u00a0;
- les lignes ne contiennent ni confiance ni identifiant de suivi.

<code-tabs name="usage" />

## detect

Exactement cinq champs par ligne\u00a0:

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h` décrit une bounding box alignée sur les axes et normalisée. `w` et
`h` doivent être positifs.

## segment

Une ligne de polygone\u00a0:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N` vaut au moins 3, le nombre de coordonnées après `class_id` doit être pair
et le polygone ne doit pas être dégénéré. Une ligne de détection à cinq champs
est également acceptée et représente un segment rectangulaire.

## pose

Le YAML ajoute `kpt_shape`, requis et égal à `[K, 2]` ou `[K, 3]`, ainsi que
`flip_idx`, facultatif, une permutation entière de `0..K-1`.

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Le nombre de champs vaut exactement `5 + K * D`, où `D` est la seconde valeur
de `kpt_shape`. Les coordonnées des points clés sont normalisées. La visibilité
`v`, lorsqu'elle est présente, vaut `0`, `1` ou `2`.

## obb

Exactement neuf champs\u00a0:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Les quatre points sont des coordonnées d'image normalisées dans `[0, 1]` et
forment un rectangle orienté non dégénéré. Aucun angle n'est stocké dans le
fichier d'étiquettes.

Le parseur canonique est strict par défaut et refuse les coordonnées hors
plage. L'ingestion des datasets et de la validation peut écrêter les
coordonnées dans `[0, 1]` pour des étiquettes de bord de recadrage par ailleurs
valides, puis refuse toujours les bounding boxes dégénérées. Le parsing tient
compte de la tâche\u00a0: neuf champs signifient `obb` uniquement en mode `obb`,
tandis qu'en mode `segment` ils peuvent former un polygone à quatre points.

En interne, les coins normalisés sont convertis en `xywhr` canonique, dont
l'angle en radians représente la rotation du côté largeur autour du centre de
la bounding box. Les résultats publics exposent les détections OBB sous forme
de lignes `xywhr, conf, cls`.

Le chargement OBB depuis un JSON COCO natif accepte les annotations dans
l'ordre de priorité suivant\u00a0: `obb` comme huit coins en pixels\u00a0; `obb` comme
`[cx, cy, w, h, angle]` avec l'angle en radians\u00a0; un polygone ou RLE COCO
`segmentation`, ajusté en rectangle de surface minimale\u00a0; puis un `bbox` COCO,
lu comme aligné sur les axes et canonisé.

Mosaic et mixup sont désactivés pour l'entraînement OBB tant qu'une
augmentation OBB consciente des coins n'existe pas.

Le parseur de lignes canonique est
`libreyolo.data.parse_yolo_obb_label_line`.

## semantic

Chaque image est associée à un masque dense mono-canal dans un format sans
perte, généralement PNG, au lieu d'un fichier `.txt`\u00a0:

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

Le masque possède un seul canal et les PNG en mode palette sont lus sous forme
d'indices de palette. Chaque pixel est un identifiant de classe dans
`0..nc-1`. La valeur `255` signifie ignorer et est exclue de la loss et des
métriques. La résolution du masque doit être égale à celle de l'image.

Deux clés YAML facultatives complètent le contrat commun. `masks_dir` est le
nom du répertoire de masques qui remplace `images` dans chaque chemin et vaut
`masks` par défaut. `label_mapping` est un remappage `{source_id: train_id}`
appliqué aux valeurs des pixels lors du chargement. Les valeurs sources non
remappées deviennent ignore et les identifiants d'entraînement doivent se
trouver dans `0..nc-1`.

Lorsque `masks_dir` est omis, les masques sont rasterisés au chargement depuis
les étiquettes polygonales `segment` résolues par la convention `images` vers
`labels`, et une classe `background` est ajoutée après les classes d'objets.
`nc` augmente donc d'une unité.

Chargeur canonique\u00a0: `libreyolo.data.SemanticDataset`.

## panoptic

LibreYOLO adopte le format COCO-panoptic tel quel (Kirillov et al., CVPR 2019).
Il n'existe aucun format panoptique propre à LibreYOLO.

Un PNG RGB par image, à la résolution de l'image, encode l'identifiant de
segment de chaque pixel dans sa couleur\u00a0:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Chaque pixel appartient à exactement un segment et les segments ne se
chevauchent jamais. L'identifiant de segment `0`, noir RGB, représente le vide\u00a0:
les pixels sans étiquette exclus de la métrique.

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1, "supercategory": "person"}]
}
```

`annotations[].file_name` nomme le PNG des identifiants de segments dans
`panoptic_dir`, et `segments_info[].id` correspond à une valeur de ce PNG.
`iscrowd` signale les régions de groupes\u00a0: elles ne sont jamais des faux
négatifs et une prédiction qui en couvre majoritairement une n'est pas un faux
positif.

La distinction thing ou stuff est une propriété de chaque catégorie.
`isthing` se trouve dans `categories`, jamais dans `segments_info`.

Les valeurs `category_id` COCO-panoptic sont les identifiants bruts du dataset
et sont généralement non contiguës. Les modèles prédisent des identifiants
contigus `0..nc-1`. Les identifiants bruts sont donc remappés par nom de
catégorie selon `names` dans le YAML, comme le fait le chargeur de détection
JSON COCO natif. L'absence dans `names` d'une catégorie du JSON est une erreur
et non une suppression silencieuse, car elle compterait autrement comme un
faux négatif permanent.

```yaml
path: coco
val: images/val2017
annotations:
  val: annotations/panoptic_val2017.json
panoptic_dir:
  val: annotations/panoptic_val2017
names: {0: person, 1: bicycle, 132: rug-merged}
```

`annotations` et `panoptic_dir` acceptent un chemin unique ou une table par
sous-ensemble.

La validation rapporte la Panoptic Quality, calculée à la résolution de la
vérité terrain et moyennée sur les catégories présentes, puis divisée entre
`PQ_things` et `PQ_stuff`. L'appariement est unique\u00a0: un segment prédit et un
segment de vérité terrain de même catégorie correspondent lorsque leur IoU
dépasse 0.5.

Chargeur canonique\u00a0: `libreyolo.data.PanopticDataset`.

## depth

Chaque image est associée à une carte de profondeur dense mono-canal\u00a0:

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

La carte est un PNG ou TIF mono-canal, ou un fichier `.npy`, à la résolution
de l'image. Ses valeurs représentent directement la profondeur dans une unité
cohérente au sein du dataset. Les valeurs nulles, négatives, NaN et infinies
signalent les pixels invalides et sont exclues de la loss et des métriques.

| Clé | Valeur par défaut | Signification |
|---|---|---|
| `depths_dir` | `depths` | Répertoire de profondeur qui remplace `images` |
| `depth_stem_suffix` | | Suffixe ajouté au nom de base de l'image\u00a0; s'il est omis, le même nom et un suffixe `_depth` sont essayés |
| `depth_mask_suffix` | `_mask` | Suffixe d'un masque de validité\u00a0; les valeurs du masque nulles ou négatives, NaN et infinies invalident le pixel de profondeur |
| `depth_scale` | `256.0` | Diviseur des cartes de profondeur à type entier, convention courante des PNG 16 bits |

Les cartes `.npy` flottantes sont utilisées telles quelles et n'appliquent pas
`depth_scale`.

Chargeur canonique\u00a0: `libreyolo.data.DepthDataset`.

## edge

Chaque image RGB est associée à une carte mono-canal sans perte de même nom de
base et à un masque de validité facultatif\u00a0:

```text
images/val/scene.jpg -> edges/val/scene.png
                     -> masks/val/scene.png
```

La carte est un PNG ou TIF mono-canal, et non une visualisation RGB, à la
résolution de l'image. Les cartes entières sont divisées par la valeur maximale
de leur dtype. Les cartes flottantes doivent déjà être finies et dans
`[0, 1]`. `0` signifie absence de contour et `1` présence d'un contour. Les
pixels d'un masque facultatif sont valides lorsqu'ils sont non nuls. Le
redimensionnement utilise l'interpolation au plus proche voisin pour les cibles
et les masques. Les pixels de remplissage sont invalides et ne participent pas
à la validation.

| Clé | Valeur par défaut | Signification |
|---|---|---|
| `edges_dir` | `edges` | Répertoire des cartes de contours qui remplace `images` |
| `edge_stem_suffix` | | Suffixe ajouté aux noms de base des images |
| `edge_extension` | `.png` | Extension sans perte des cibles |
| `edge_invert` | | Définissez true lorsque les cartes sources contiennent des contours noirs sur fond blanc |
| `masks_dir` | `masks` | Répertoire facultatif des masques de validité |

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

La validation amincit les prédictions continues avec une suppression non
maximale des gradients dans quatre directions, puis rapporte les mesures F ODS
et OIS sur un balayage de seuils configurable. Les pixels prédits et de vérité
terrain sont appariés un-à-un dans la limite de
`edge_max_dist * image_diagonal`, avec une tolérance normalisée par défaut de
`0.0075`.

Chargeur canonique\u00a0: `libreyolo.data.EdgeDataset`. Le chargeur ne s'occupe que
du format, il ne télécharge ni ne redistribue les données de benchmark.

## normal

Chaque image est associée à un PNG 16 bits à trois canaux de même nom de base,
ainsi qu'à un masque de validité facultatif de même nom\u00a0:

```text
images/val/room.jpg -> normals/val/room.png
                    -> masks/val/room.png
```

Le PNG est exactement un `uint16` à trois canaux stockés en RGB, à la
résolution de l'image. Décodez-le avec `n = png / 65535 * 2 - 1`, puis
renormalisez chaque vecteur. Les vecteurs décodés utilisent le repère caméra
OpenCV, `+x` vers la droite, `+y` vers le bas et `+z` dans la scène, et font
face à la caméra. Le masque facultatif est un PNG mono-canal où une valeur non
nulle signifie valide. Sans masque, tout vecteur décodé fini et non nul est
valide. Les pixels cibles invalides et de remplissage sont représentés en
interne par `(0, 0, 0)`. Le redimensionnement interpole les trois composantes
de façon bilinéaire puis renormalise. Les masques de validité utilisent
l'interpolation au plus proche voisin, et un retournement horizontal inverse
aussi la composante x.

| Clé | Valeur par défaut | Signification |
|---|---|---|
| `normals_dir` | `normals` | Répertoire des cartes de normales qui remplace `images` |
| `masks_dir` | `masks` | Répertoire facultatif des masques de validité |

La validation rapporte l'erreur angulaire moyenne et médiane en degrés et le
pourcentage de pixels valides à moins de 11.25, 22.5 et 30 degrés.

Chargeur canonique\u00a0: `libreyolo.data.NormalDataset`.

## restore

Chaque image d'entrée dégradée est associée à une cible RGB propre\u00a0:

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

L'entrée et la cible sont des fichiers image compatibles RGB et leurs
résolutions doivent être strictement identiques. La validation conserve la
résolution native et ajoute uniquement le remplissage nécessaire pour empiler
un batch. Les métriques sont calculées sur le canevas de l'image d'origine.
L'entraînement applique un recadrage et un retournement horizontal couplés à
la paire entrée-cible.

| Clé | Valeur par défaut | Signification |
|---|---|---|
| `input_dir` | `inputs` | Répertoire d'entrées dégradées utilisé dans les chemins des sous-ensembles |
| `target_dir` | `targets` | Répertoire de cibles propres qui remplace `input_dir` |
| `target_stem_suffix` | | Suffixe ajouté au nom de base de l'entrée avant la recherche de la cible |
| `target_stem_suffixes` | | Forme liste de `target_stem_suffix` |
| `degradation` | | Étiquette de métadonnées comme `deblur` ou `denoise` |
| `dataset` | | Étiquette de dataset ou de provenance |

Les champs YAML de type classe sont des valeurs factices du schéma\u00a0: utilisez
`nc: 1` et `names: {0: image}`. Les modèles de restauration exposent
`Results.restored`, et non des détections.

Chargeur canonique\u00a0: `libreyolo.data.RestoreDataset`.

## matte

Chaque image RGB est associée à un matte de vérité terrain mono-canal de même
nom de base, où 0 représente l'arrière-plan et 255 le premier plan\u00a0:

```text
images/subject.jpg -> mattes/subject.png
```

Deux organisations sont acceptées. Une racine de répertoire contenant
`images/` et un répertoire de mattes, détecté automatiquement parmi `mattes/`,
`matte/`, `gt/`, `masks/`, `mask/` et `alpha/`, et transmise dans `data=`. Ou
un YAML avec `path`, `val_images` et `val_mattes` par sous-ensemble, et
facultativement `train_images` et `train_mattes`, tous relatifs à `path` ou
absolus.

Le matte est lu en nuances de gris comme opacité dans `[0, 1]` et redimensionné
vers le canevas de prédiction par interpolation bilinéaire si les formes
diffèrent. Les métriques sont la MAE et la S-measure (Fan et al., ICCV 2017)
sur le canevas de l'image d'origine. La S-measure sert de fitness pour le
meilleur checkpoint.

Les champs YAML de type classe sont des valeurs factices du schéma\u00a0: utilisez
`nc: 1` et `names: {0: matte}`. Les modèles de matting exposent
`Results.matte`.

La validation est réservée à l'inférence dans cette version. Résolveur de
paires canonique\u00a0: `libreyolo.data.matte_dataset.resolve_matte_pairs`.

## ocr

Les étiquettes sont un fichier JSONL par sous-ensemble, avec un objet JSON par
image\u00a0:

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` est un quadrilatère à quatre points en coordonnées absolues de
pixels, ordonnés haut-gauche, haut-droit, bas-droit, bas-gauche. Les régions au
texte illisible utilisent `"text": "###"`, convention ICDAR à ignorer. Elles
sont exclues du score de reconnaissance et les prédictions qui les chevauchent
sont ignorées plutôt que pénalisées lors de l'appariement de détection.

Les métriques sont la hmean de détection avec appariement un-à-un des polygones
au-dessus d'une IoU de 0.5, le F1 de bout en bout qui exige à la fois une IoU
supérieure à 0.5 et une transcription exacte après normalisation NFKC et
suppression des espaces, en tenant compte de la casse, ainsi que 1-NED sur les
paires appariées. Le F1 de bout en bout sert de fitness pour le meilleur
checkpoint.

Deux organisations sont acceptées\u00a0: une racine contenant
`images/<split>/` et `labels/<split>.jsonl`, transmise dans `data=`, ou un YAML
avec `path` et les noms facultatifs des répertoires `images` et `labels`.

Les champs YAML de type classe sont des valeurs factices du schéma\u00a0: utilisez
`nc: 1` et `names: {0: text}`. Les modèles OCR exposent `Results.ocr`.

La validation est réservée à l'inférence dans cette version. Résolveur
d'échantillons canonique\u00a0: `libreyolo.data.ocr_dataset.resolve_ocr_samples`.

## classify

Une arborescence de répertoires de type ImageFolder, et non des fichiers
d'étiquettes\u00a0:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

`train/` est requis pour l'entraînement et définit l'association des classes
aux indices selon le tri des noms de dossiers. `val/` est requis pour la
validation. `test/` peut être présent, mais les commandes train et val par
défaut ne l'utilisent pas. Les sous-ensembles autres que l'entraînement doivent
contenir les mêmes noms de dossiers de classes que l'ensemble attendu par
l'entraînement ou le checkpoint. Les extensions d'images prises en charge sont
définies dans `libreyolo.data.classify_dataset.IMAGE_EXTENSIONS`.

## gaze et point

Aucun contrat de fichiers de dataset d'entraînement ou de validation n'est
implémenté pour `gaze`.

`point` est une tâche de sortie du modèle plutôt qu'un schéma d'étiquettes de
dataset. Les familles de points peuvent adapter des étiquettes existantes en
interne, par exemple en déduisant le centre des objets depuis des lignes de
bounding boxes, mais aucun format d'étiquettes textuel réservé aux points
n'est défini.
