---
title: Segmentation panoptique
seo_title: Segmentation panoptique dans LibreYOLO
description: >-
  Attribuer un segment à chaque pixel dans LibreYOLO : familles qui couvrent la
  tâche, format de dataset COCO-panoptic et appels de prédiction et de
  validation.
lead: >-
  La segmentation panoptique attribue chaque pixel à un seul segment sans
  chevauchement, réunissant les instances d'objets dénombrables et les régions
  amorphes de l'arrière-plan. La clé de tâche est panoptic.
keywords:
  - segmentation panoptique python
  - qualité panoptique
  - segmentation objets régions
  - format COCO panoptic
  - carte identifiants segments
  - métrique PQ
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Le suffixe -panoptic du nom de fichier sélectionne la tâche.
        # Aucun argument task n'est donc nécessaire.
        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # identifiants de segments (H, W)
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-panoptic.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Un segment à la fois
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreEoMTl-panoptic.pt")(SAMPLE_IMAGE)
        pan = result.panoptic

        for segment in pan.segments_info:
            pixels = pan.segment_mask(segment["id"])   # booléen (H, W)
            print(result.names[segment["category_id"]], int(pixels.sum()))
    - label: Un checkpoint plus petit
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTs-panoptic.pt")
        result = model(SAMPLE_IMAGE)

        print(len(result.panoptic.segment_ids))
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        # val() renvoie un dictionnaire ordinaire et non un objet.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
        print(metrics["metrics/PQ_things"], metrics["metrics/PQ_stuff"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-panoptic.pt data=my-dataset.yaml
source_hash: b8adc9ccde7a4e6c
---

## Définition

La segmentation panoptique est l'union des deux autres tâches de segmentation.
Chaque pixel reçoit exactement un segment, les segments ne se chevauchent
jamais, et chacun constitue soit une chose, c'est-à-dire une instance d'objet
dénombrable, soit une matière, région amorphe telle que le ciel ou la route.
Elle est donc plus stricte que la
[segmentation d'instances](/docs/tasks/instance-segmentation), qui laisse les
pixels d'arrière-plan sans attribution et autorise le chevauchement des masques,
ainsi que la
[segmentation sémantique](/docs/tasks/semantic-segmentation), qui étiquette
chaque pixel mais fusionne les instances contiguës d'une même classe.

`panoptic` est la clé de tâche canonique, et le suffixe `-panoptic` du nom d'un
checkpoint la sélectionne. L'argument `task=` est donc inutile lors du
chargement des poids publiés.

`predict()` remplit `result.panoptic`. `.data` est une carte entière
d'identifiants de segments de forme `(H, W)` sur le canevas de l'image
d'origine. `.segments_info` est une liste de dictionnaires, un par segment,
contenant chacun au minimum `{"id", "category_id"}`. `id` correspond à une
valeur de la carte et `category_id` indexe `result.names`. `.segment_ids`
énumère les identifiants présents dans l'ordre croissant, et
`.segment_mask(id)` renvoie la sélection booléenne `(H, W)` d'un segment.
L'identifiant de segment `0` est la valeur vide. Ses pixels sans étiquette sont
exclus de la métrique et absents de `.segment_ids`.

La distinction entre chose et matière est une propriété de la catégorie, et non
du segment individuel. Elle est portée par les métadonnées de catégorie de
l'ensemble d'étiquettes. Une charge utile de prédiction peut la copier dans
chaque segment sous `"isthing"` pour plus de commodité, mais les métadonnées de
catégorie restent la source faisant autorité.

## Modèles

[EoMT](/docs/models/eomt) est la famille qui couvre cette tâche par
`LibreYOLO()`. Elle s'exécute avec le paquet de base et publie des checkpoints
panoptiques entraînés sur COCO dans trois tailles, s, b et l.

[SenseNova-Vision](/docs/models/sensenova-vision) émet également des cartes
panoptiques. Il s'agit d'un modèle génératif guidé doté de sa propre fabrique,
`LibreVLM`, et de son propre extra. Si aucun vocabulaire n'est défini, il revient
aux catégories panoptiques COCO sur lesquelles il a été ajusté. Ses poids sont
réservés à un usage non commercial. La latence par image est bien supérieure à
celle d'un segmenteur spécialisé, car chaque prédiction passe par un décodage
de diffusion.

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

`conf` filtre la sélection des requêtes. Consultez la page
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Format du dataset

LibreYOLO adopte le format COCO-panoptic de Kirillov et al., CVPR 2019, sans
aucune modification. Il n'existe aucun format panoptique propre à LibreYOLO.

```text
dataset/
  data.yaml
  images/
    val/000000000139.jpg
  annotations/
    panoptic_val.json
    panoptic_val/000000000139.png
```

Chaque image est associée à un PNG RVB de même résolution, dont la couleur de
chaque pixel encode l'identifiant du segment auquel il appartient :

```text
segment_id = R + 256 * G + 256 * 256 * B
```

L'identifiant de segment `0`, le noir RVB, est vide. Ses pixels sans étiquette
ne récompensent ni ne pénalisent une prédiction. Tous les autres pixels
appartiennent exactement à un segment.

Le JSON répertorie, pour chaque image, le PNG des identifiants de segments et
les segments qu'il contient :

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1}]
}
```

`annotations[].file_name` nomme le PNG dans le répertoire panoptique, et
`segments_info[].id` correspond à une valeur de ce PNG. `iscrowd` signale les
régions de groupes. Elles ne sont jamais comptées comme faux négatifs, et une
prédiction qui en couvre la majeure partie n'est pas un faux positif.
`isthing` appartient à `categories` et jamais à un segment individuel.

Le fichier YAML renvoie vers les deux :

```yaml
path: dataset
val: images/val
annotations:
  val: annotations/panoptic_val.json
panoptic_dir:
  val: annotations/panoptic_val
names:
  0: person
  1: bicycle
```

`annotations` et `panoptic_dir` acceptent chacun un chemin unique ou une
association par partition. Les identifiants bruts des catégories COCO ne sont
généralement pas contigus, tandis que les modèles prédisent une plage contiguë
`0..nc-1`. Les identifiants sont donc remappés selon le nom de catégorie par
`names`. Une catégorie JSON absente de `names` déclenche une erreur au lieu
d'être ignorée silencieusement, car sa suppression serait évaluée comme un faux
négatif permanent.

Le chargeur canonique est `libreyolo.data.PanopticDataset`.

## Entraîner

Aucune famille n'entraîne actuellement la segmentation panoptique dans
LibreYOLO. `train()` d'EoMT déclenche une `NotImplementedError`. Les checkpoints
panoptiques sont donc utilisés tels qu'ils sont publiés.

## Valider

`val()` renvoie un dictionnaire ordinaire de clés `metrics/`, calculées à la
résolution de la vérité terrain sur la partition `val` du fichier YAML du
dataset. Un segment prédit et un segment réel de même catégorie correspondent
lorsque leur IoU dépasse 0,5. Cette correspondance est unique.

<code-tabs name="val" />

`metrics/PQ` est la qualité panoptique, le nombre principal. Pour chaque
catégorie, elle est le produit de deux facteurs. La qualité de segmentation est
l'IoU moyenne sur les segments associés et indique l'alignement des formes. La
qualité de reconnaissance est `TP / (TP + 0.5 FP + 0.5 FN)`, soit le score F1 de
l'association elle-même, et indique le nombre de segments effectivement
trouvés. Les trois valeurs sont ensuite moyennées sur les catégories présentes
et rapportées sous `metrics/PQ`, `metrics/SQ` et `metrics/RQ`. La PQ rapportée
est donc la moyenne des produits par catégorie plutôt que le produit des deux
moyennes rapportées.

`metrics/PQ_things` et `metrics/PQ_stuff` calculent séparément la moyenne de
cette même PQ par catégorie sur les catégories de choses et de matières.
`metrics/categories` compte les catégories présentes et donc incluses dans la
moyenne. Le dictionnaire contient aussi `fitness`, une copie de la valeur PQ.

## Exporter

Les checkpoints panoptiques ne s'exportent pas. `export()` déclenche une
`NotImplementedError` pour cette tâche, car la sortie requêtes-masques ne
possède pas encore de contrat d'exportation d'exécution. La tâche sémantique
d'EoMT s'exporte. Consultez les pages
[segmentation sémantique](/docs/tasks/semantic-segmentation) et
[exporter et déployer](/docs/export).
