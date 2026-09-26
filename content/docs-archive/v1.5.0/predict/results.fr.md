---
title: Travailler avec les résultats
seo_title: L'objet Results de LibreYOLO
description: "Un objet Results par image, avec un emplacement par type de charge utile\_: bounding boxes, masques, points clés, probabilités, profondeur, segmentation panoptique, OCR et plus encore. Visualisation, enregistrement et JSON."
lead: >-
  Chaque prédiction renvoie un objet Results par image. Il possède un
  emplacement nommé par type de charge utile, tous vides à l'exception de ceux
  produits par le modèle, et un artefact exporté utilise les mêmes emplacements.
keywords:
  - objet results yolo python
  - results.boxes xyxy
  - results vers json
  - enregistrer image annotée
  - masques segmentation python
  - résultats points clés
  - résultats carte profondeur
  - results summary
  - mêmes résultats onnx
last_verified: 1.5.0
verification: >-
  Classes de charges utiles, emplacements, sémantique des déplacements,
  summary(), to_json(), plot(), save() et cutout() lus dans
  libreyolo/utils/results.py. Comportement d'annotation et d'écriture sur disque
  lu dans InferenceRunner._save_annotated_image dans
  libreyolo/models/base/inference.py et resolve_save_path dans
  libreyolo/utils/general.py. Routage par suffixe lu dans LibreYOLO() dans
  libreyolo/models/__init__.py.
snippets:
  basic:
    - label: Bounding boxes
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        print(result.orig_shape)   # (hauteur, largeur) de l'image source

        print(result.path)         # chemin source, None pour une entrée en
        mémoire


        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Coordonnées normalisées
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy[:1])    # pixels, x1 y1 x2 y2

        print(result.boxes.xywh[:1])    # pixels, centre x, centre y, l, h

        print(result.boxes.xyxyn[:1])   # même bounding box divisée par la
        largeur et la hauteur

        print(result.boxes.xywhn[:1])
    - label: NumPy et appareils
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        # Chacune de ces opérations renvoie un nouvel objet Results ; l'original
        ne change pas.

        as_numpy = result.numpy()

        on_cpu = result.cpu()


        print(type(as_numpy.boxes.xyxy).__name__)

        print(type(on_cpu.boxes.xyxy).__name__)
  json:
    - label: summary et to_json
      language: python
      code: |
        import json

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        rows = result.summary()
        print(json.dumps(rows[:2], indent=2))

        # Même contenu sous forme de chaîne, avec les mêmes arguments nommés.
        print(result.to_json(normalize=True, decimals=3)[:200])
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt --json \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  saving:
    - label: Images annotées
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")


        # save=True dessine la charge utile et l'écrit sous
        runs/detect/predict*.

        result = model(SAMPLE_IMAGE, save=True)

        print(result.saved_path)
  exported:
    - label: Installer l'extra d'export
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Les mêmes Results depuis un artefact exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")   # renvoie le chemin écrit

        # LibreYOLO() effectue le routage selon le suffixe du fichier.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(type(result).__name__, len(result.boxes))
source_hash: 548dbc9c7f5552ec
---

## Un objet, un emplacement par charge utile

Une prédiction sur une image renvoie un objet `Results`. Il contient dix-huit
emplacements de charges utiles et un modèle remplit uniquement ceux que sa
tâche produit. Tous les autres valent `None`, lire `result.masks` sur un
détecteur renvoie donc `None` au lieu de lever une erreur.

| Emplacement | Classe | Forme | Produit par |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)` avec scores et classes | Détection et toute tâche qui commence par une localisation |
| `masks` | `Masks` | `(N, H, W)` | Segmentation d'instances |
| `keypoints` | `Keypoints` | `(N, K, 2)` ou `(N, K, 3)` | Pose |
| `probs` | `Probs` | `(C,)` | Classification |
| `obb` | `OBB` | `(N, 7)` ou `(N, 8)` | Bounding boxes orientées |
| `gaze` | `Gaze` | `(N, 2)` tangage et lacet en radians | Estimation du regard |
| `points` | `Points` | `(N, 4)` sous la forme x, y, classe, confiance | Localisation de points |
| `semantic_mask` | `SemanticMask` | `(H, W)` identifiants de classe | Segmentation sémantique |
| `panoptic` | `PanopticSegmentation` | `(H, W)` identifiants de segments avec `segments_info` | Segmentation panoptique |
| `depth_map` | `DepthMap` | `(H, W)` flottants | Estimation de profondeur |
| `normal_map` | `NormalMap` | `(H, W, 3)` vecteurs unitaires | Normales de surface |
| `edges` | `EdgeMap` | `(H, W)` flottants dans `[0, 1]` | Détection de contours |
| `restored` | `RestoredImage` | `(H, W, 3)` uint8 RGB | Restauration et super-résolution |
| `matte` | `Matte` | `(H, W)` flottants dans `[0, 1]` | Matting alpha et suppression de l'arrière-plan |
| `ocr` | `OCRRegions` | `(N, 4, 2)` polygones avec transcriptions | Détection et reconnaissance de texte |
| `embeddings` | `Embeddings` | `(N, D)` lignes normalisées L2 | Tâche `embed` |
| `identities` | `Identities` | N noms et scores | Tâche `embed` avec une galerie |
| `meshes` | `Meshes` | Paramètres corporels et sommets facultatifs | Reconstruction de maillage corporel |

À leurs côtés figurent les champs présents dans chaque résultat\u00a0:
`orig_shape` sous la forme `(height, width)`, `path` (le chemin source, ou
`None` pour une entrée en mémoire), `names` qui associe les identifiants de
classe aux noms de classes, `frame_idx` pour les images vidéo et en direct,
`track_id` lors du suivi, ainsi que `restore_scale`, le facteur entier
d'agrandissement d'un résultat de restauration.

`result.normals` est un alias de `result.normal_map`.

`result.speed` existe dans chaque résultat, mais seuls les
[ensembles](/docs/predict/ensembling) le renseignent. Ses clés sont `member_0`,
`member_1` et `fusion`, en millisecondes. Pour un modèle unique, il reste un
dictionnaire vide.

## Bounding boxes

<code-tabs name="basic" />

`Boxes` conserve les coordonnées et les scores dans des tableaux distincts au
lieu d'un tenseur compacté unique.

| Attribut | Contenu |
|---|---|
| `xyxy` | `(N, 4)` pixels absolus, x1 y1 x2 y2 |
| `xywh` | `(N, 4)` pixels absolus, centre x, centre y, largeur, hauteur |
| `xyxyn`, `xywhn` | Les mêmes valeurs divisées par la largeur et la hauteur de l'image |
| `conf` | `(N,)` confiance |
| `cls` | `(N,)` identifiant de classe sous forme de flottant |
| `id` | `(N,)` identifiant de suivi, ou `None` |
| `is_track` | Indique si `id` est défini |
| `data` | Tous les éléments concaténés\u00a0: bounding boxes, identifiant facultatif, confiance, classe |

`cls` est un tableau de flottants, utilisez-le donc sous la forme
`result.names[int(cls)]`.

`xyxyn` et `xywhn` ont besoin de `orig_shape`, que `Results` renseigne pour
vous.

## Charges utiles denses

Les charges utiles qui couvrent l'image entière se comportent différemment de
celles associées à chaque instance, ce qui compte lors d'un découpage.

`SemanticMask` contient les identifiants de classe `(H, W)` sur le canevas
d'origine, avec `255` réservé comme valeur à ignorer qui n'est jamais comptée
comme une classe. `classes` énumère les identifiants présents en l'excluant,
et `class_mask(id)` renvoie un tableau booléen `(H, W)`.

`PanopticSegmentation` contient les identifiants de segments `(H, W)`, avec
`0` comme identifiant vide, ainsi qu'une liste de dictionnaires `segments_info`
qui contiennent au moins `id` et `category_id`. `segment_ids` énumère les
identifiants présents et `segment_mask(id)` en sélectionne un.

`DepthMap` contient la profondeur inverse relative `(H, W)`\u00a0: une valeur plus
élevée signifie une plus grande proximité, sans représenter des mètres. Il
expose `min`, `max` et `mean` sur les valeurs finies, et `normalized()` remet
les valeurs à l'échelle dans `[0, 1]`.

`NormalMap` contient des vecteurs unitaires `(H, W, 3)` dans le repère caméra
OpenCV, où `+x` va vers la droite, `+y` vers le bas et `+z` dans la scène. Une
surface faisant face à la caméra vaut donc `(0, 0, -1)`.
`assert_normalized()` vérifie que chaque pixel est fini et de longueur unitaire.

`EdgeMap` contient des valeurs float32 `(H, W)` dans `[0, 1]`. La carte continue
est conservée plutôt que seuillée, vous choisissez donc le seuil avec
`binary(threshold=0.5)`.

`Matte` contient des valeurs float32 `(H, W)` dans `[0, 1]`, où `1` désigne
entièrement le premier plan. `array` les renvoie écrêtées au format float32.

`RestoredImage` contient une image RGB uint8 `(H, W, 3)`, avec `array` pour le
ndarray brut et `save(path)` pour l'écrire.

`Probs` contient un vecteur de probabilités pour l'image. `top1` et `top5` sont
les indices de classes, `top1conf` et `top5conf` les scores correspondants.

`Embeddings` contient des lignes `(N, D)` déjà normalisées L2, la similarité
cosinus est donc un produit scalaire. `similarity(other)` renvoie `(N, M)` par
rapport à une galerie ou `(N,)` par rapport à un seul vecteur, et
`verify(i, j, threshold=0.4)` compare deux lignes.

`OCRRegions` contient des polygones `(N, 4, 2)` dans l'ordre de lecture, dont
les sommets suivent l'ordre haut-gauche, haut-droit, bas-droit, bas-gauche. Les
transcriptions figurent dans `texts`, les scores de reconnaissance dans `conf`
et les scores de détection dans `det_conf`. Comme il s'agit de véritables
polygones tournés, ils ne remplissent pas `boxes`\u00a0; `ocr.xyxy` donne leurs
enveloppes alignées sur les axes lorsque vous avez besoin de rectangles.

## Découper et déplacer

`result[i]` renvoie un nouvel objet `Results` qui contient une instance. Les
charges utiles par instance sont découpées, tandis que celles qui couvrent
l'image entière sont conservées sans modification. Le découpage d'un résultat
de classification ne peut donc pas tronquer son vecteur de probabilités à une
seule classe, et celui d'un résultat de profondeur ne peut pas altérer la
disposition `(H, W)`.

`len(result)` compte les instances\u00a0: bounding boxes, points, embeddings,
régions OCR ou maillages. Toute charge utile dense couvrant l'image entière
compte pour `1`. Un résultat vide vaut `0`.

`to()`, `cpu()`, `cuda()` et `numpy()` renvoient chacun un nouvel objet
`Results` dans lequel chaque emplacement rempli a été converti. Ils ne
modifient pas l'original.

`update()` est la seule méthode qui modifie l'objet sur place. Elle remplace
les emplacements nommés et renvoie le même objet.

## JSON

<code-tabs name="json" />

`summary()` renvoie une liste de dictionnaires simples, et `to_json()` passe
cette liste à `json.dumps`. Les deux acceptent les trois mêmes arguments\u00a0:
`normalize=False` fait passer les coordonnées dans `[0, 1]`, `decimals=5`
définit l'arrondi et `embeddings=False` détermine si les vecteurs d'embeddings
sont inclus.

La forme des lignes suit la charge utile. Les lignes de détection contiennent
`name`, `class`, `confidence` et un dictionnaire `box`, auxquels s'ajoutent
`segments` en présence de masques, `obb` et `corners` pour les bounding boxes
orientées, les angles `gaze` en radians et degrés, `track_id` lors du suivi et
les paramètres `mesh` en présence de maillages.

En l'absence de bounding boxes, une charge utile détermine les lignes\u00a0: l'OCR
émet une ligne par région avec son `text`, les points une ligne par point, la
segmentation panoptique une ligne par segment avec `pixel_count` et
`pixel_fraction`, la segmentation sémantique une ligne par classe présente,
et la classification les cinq premières classes. La profondeur, les normales,
les contours, la restauration et le matting émettent chacun une seule ligne de
résumé décrivant la carte plutôt que ses pixels.

Deux charges utiles sont délibérément abrégées. Un vecteur d'embedding est
rapporté uniquement sous la forme `embedding_dim`, car une ligne de 512
flottants occupe environ 2\u00a0Ko par visage\u00a0; transmettez `embeddings=True` pour
inclure les valeurs. Les sommets des maillages ne sont jamais inclus, car ils
représentent des dizaines de milliers de coordonnées par personne. Lisez
`result.meshes.vertices` ou appelez `result.meshes.save_obj(path)` pour accéder
à la géométrie.

## Dessiner et enregistrer

<code-tabs name="saving" />

`predict(save=True)` annote et écrit le résultat. Il choisit la routine de
dessin d'après l'emplacement rempli. Un résultat sémantique est ainsi écrit
sous forme de masque coloré, un résultat de profondeur sous forme de
visualisation de la profondeur, un résultat panoptique avec ses segments, un
matte sous forme de PNG RGBA à arrière-plan transparent et une détection sous
forme de bounding boxes avec les masques en dessous. Le chemin écrit est joint
au résultat sous la forme `result.saved_path`.

`Results.plot()` est plus limité que son nom ne le suggère. Il est défini
uniquement pour les cartes de normales et de contours, et lève
`NotImplementedError` pour tout le reste. Utilisez `save=True` pour les autres
tâches.

`Results.save(path)` est tout aussi limité\u00a0: il écrit un résultat de matting
sous forme de découpe PNG RGBA à arrière-plan transparent et lève
`NotImplementedError` dans les autres cas. `Results.cutout()` renvoie ce même
tableau RGBA sans l'écrire. Les deux ont besoin de l'image source, lue dans
`result.path` ou transmise avec `image=`.

Deux charges utiles disposent de leur propre méthode d'écriture\u00a0:
`result.restored.save(path)` pour une image restaurée et
`result.meshes.save_obj(path, index=0)` pour un maillage.

Pour connaître l'emplacement des fichiers et le comportement de `output_path`
et `output_file_format`, consultez les
[sources de prédiction](/docs/predict/sources).

## Les artefacts exportés renvoient le même objet

<code-tabs name="exported" />

`LibreYOLO()` effectue le routage selon le suffixe du fichier. Un artefact
exporté se charge donc avec le même appel qu'un checkpoint `.pt` et renvoie le
même objet `Results`. Les fichiers `.onnx`, `.engine`, `.pte` et `.mnn` sont
reconnus par leur suffixe, tout comme les répertoires OpenVINO, Paddle et ncnn,
ainsi qu'une URL de modèle Triton. Le code qui lit `result.boxes.xyxy` ne change
pas lorsqu'un modèle est remplacé par sa version exportée. Consultez la page
[Export](/docs/export) pour l'ensemble des formats.

Si vous utilisez directement l'API du runtime, vous prenez vous-même en charge
le prétraitement, le post-traitement et les noms de classes.
