---
title: Segmentation d'instances
seo_title: Segmentation d'instances dans LibreYOLO
description: >-
  Segmenter des objets individuels dans LibreYOLO : familles qui couvrent la
  tâche, format des annotations polygonales et appels de prédiction,
  d'entraînement, de validation et d'exportation.
lead: >-
  La segmentation d'instances localise chaque instance d'objet et renvoie un
  masque par pixel pour chacune, en plus de la boîte, de la classe et du score
  produits par un détecteur. La clé de tâche est segment.
keywords:
  - segmentation instances python
  - prédiction masques objets
  - entraînement modèle segmentation
  - annotations polygones
  - bibliothèque segmentation MIT
  - mAP masques
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Le suffixe -seg du nom de fichier sélectionne la tête de masque.
        # Aucun argument task n'est donc nécessaire.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)   # (N, H, W), un masque par détection
        print(result.boxes.xyxy.shape)   # (N, 4), les mêmes N lignes
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn-seg.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Contours des masques
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDFINEn-seg.pt")

        result = model(SAMPLE_IMAGE)


        # .xy est une liste de contours (P, 2) en pixels, .xyn leur version
        normalisée.

        for name, contour in zip(result.boxes.cls, result.masks.xy):
            print(result.names[int(name)], contour.shape)
    - label: 'Une autre famille, le même appel'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Continue depuis les poids de segmentation publiés, tête de masque
        comprise.

        # data doit désigner un dataset dont les annotations contiennent des
        polygones.

        model = LibreYOLO("LibreDFINEn-seg.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Depuis des poids de détection
      language: bash
      code: |
        # Les poids de détection ne contiennent aucune tête de masque. Il s'agit
        # donc d'un transfert explicite : la tête démarre sans entraînement.
        # Demander task=segment est ce qui l'autorise.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])       # masques
        print(metrics["metrics/mAP50-95(M)"])    # masques, explicite
        print(metrics["metrics/mAP50-95(B)"])    # boîtes
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn-seg.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn-seg.pt format=onnx imgsz=640
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La fabrique s'oriente grâce au suffixe du fichier. Un artefact exporté
        # se charge comme un checkpoint et renvoie le même objet Results.
        model = LibreYOLO("LibreDFINEn-seg.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 33e331eac0f9b0af
---

## Définition

La segmentation d'instances est une détection complétée par une forme. Chaque
instance d'objet reçoit toujours une boîte, une classe et un score, ainsi qu'un
masque binaire qui couvre les pixels lui appartenant. Les masques peuvent se
chevaucher et les pixels qui n'appartiennent à aucun objet restent sans
attribution. C'est ce qui distingue cette tâche de la
[segmentation sémantique](/docs/tasks/semantic-segmentation) et de la
[segmentation panoptique](/docs/tasks/panoptic-segmentation).

`segment` est la clé de tâche canonique, et le suffixe `-seg` du nom d'un
checkpoint la sélectionne. L'argument `task=` est donc inutile lors du
chargement des poids publiés.

`predict()` remplit `result.masks` en parallèle de `result.boxes`. `.data` est
un empilement `(N, H, W)` sur le canevas de l'image d'origine, aligné par ligne
avec les boîtes. Le masque `i` appartient donc à la boîte `i`. `.xy` convertit
chaque masque en son plus grand contour extérieur sous forme de tableau de
pixels `(P, 2)`, et `.xyn` fournit le même contour normalisé.

## Modèles

Quatre familles peuvent être entraînées et prédire des masques :
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter),
[D-FINE](/docs/models/d-fine) et [RTMDet](/docs/models/rtmdet). RF-DETR
nécessite son propre extra, `pip install "libreyolo[rfdetr]"`. Les trois autres
s'exécutent avec le paquet de base.

[Mask R-CNN](/docs/models/mask-rcnn) prédit, valide et exporte des masques, mais
sa méthode `train()` déclenche une `NotImplementedError`.

[EoMT](/docs/models/eomt) prédit et valide des masques sans pouvoir non plus
s'entraîner. Son exportation est plus limitée encore : `export()` accepte
uniquement la tâche sémantique et déclenche une `NotImplementedError` pour
`segment` et `panoptic`, car le contrat d'exécution requêtes-masques nécessaire
à ces deux tâches n'a pas été défini. Utilisez EoMT pour les masques d'instances
en Python, et non au moyen d'un graphe exporté.

Un groupe distinct effectue la segmentation depuis un prompt plutôt que depuis
une liste de classes. Un clic, une boîte ou une phrase sélectionne l'objet, puis
le modèle renvoie son masque. [SAM](/docs/models/sam),
[SAM 2](/docs/models/sam-2), [SAM 3](/docs/models/sam-3),
[MobileSAM](/docs/models/mobilesam), [EdgeTAM](/docs/models/edgetam) et
[PicoSAM3](/docs/models/picosam3) fonctionnent ainsi, tout comme
[SenseNova-Vision](/docs/models/sensenova-vision), dont la segmentation est
référentielle et reçoit une phrase qui nomme un objet. Ces modèles se chargent
par leur propre fabrique et leurs propres extras. Chaque page de modèle présente
l'appel exact.

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

`conf` et `max_det` façonnent la sortie comme pour la détection, et les masques
sont filtrés avec les boîtes auxquelles ils appartiennent. Consultez la page
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Format du dataset

La structure est celle de la détection : un fichier d'étiquettes `.txt` par
image, trouvé en remplaçant `images` par `labels` dans le chemin de l'image et
en changeant l'extension.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

La ligne change. Un segment est un indice de classe suivi d'un polygone plat :

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

Le polygone doit comporter au moins trois points. Le nombre de coordonnées après
l'indice de classe est donc pair et au moins égal à six, et le polygone ne doit
pas être dégénéré. Les coordonnées sont des nombres flottants dans `[0, 1]`,
relatifs à la largeur et à la hauteur de l'image d'origine. Une ligne de
détection à cinq champs est également acceptée dans un dataset de segmentation
et interprétée comme un segment rectangulaire. Un dataset limité aux boîtes peut
ainsi être chargé sans conversion préalable.

Le fichier YAML est celui de la détection :

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

Le JSON COCO natif fonctionne également. Ajoutez une association `annotations`
entre le nom de partition et le fichier JSON. Le chemin de la partition indique
la racine des images.

## Entraîner

<code-tabs name="train" />

Par défaut, l'entraînement continue depuis un checkpoint `-seg` publié. Il est
possible de démarrer avec des poids de détection, mais il s'agit d'un transfert
volontaire. Ces poids ne contiennent aucune tête de masque, qui démarre donc
sans entraînement, et la transmission de `task=segment` autorise ce
remplacement. Consultez la page [entraînement](/docs/train) pour les datasets,
les augmentations, le multi-GPU et les systèmes de journalisation.

## Valider

`val()` renvoie un dictionnaire ordinaire de clés `metrics/`. Les boîtes et les
masques sont évalués séparément, tous deux avec l'évaluation COCO. Les nombres
des masques sont les principaux.

<code-tabs name="val" />

Les clés sans suffixe contiennent les résultats des masques :
`metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, puis
`metrics/mAP_small`, `metrics/mAP_medium` et `metrics/mAP_large` selon l'aire
de l'objet, ainsi que `metrics/AR1`, `metrics/AR10`, `metrics/AR100`,
`metrics/AR_small`, `metrics/AR_medium` et `metrics/AR_large` pour le rappel
moyen. `metrics/AR_max_det` et `metrics/max_det` enregistrent la limite de
détections utilisée pendant l'exécution.

Quatre valeurs sont aussi publiées avec un suffixe explicite, `(M)` pour masque
et `(B)` pour boîte. Une comparaison ne dépend ainsi jamais du nombre que la
famille a choisi comme principal : `metrics/mAP50-95(M)` et
`metrics/mAP50-95(B)`, `metrics/mAP50(M)` et `metrics/mAP50(B)`,
`metrics/precision(M)` et `metrics/precision(B)`, `metrics/recall(M)` et
`metrics/recall(B)`. Cette tâche ne possède aucune clé
`metrics/precision` ou `metrics/recall` sans suffixe.

Lisez attentivement les clés de précision et de rappel. Elles sont conservées
pour la rétrocompatibilité et constituent des alias, pas un point de
fonctionnement. `metrics/precision(M)` contient la même valeur que
`metrics/mAP50-95(M)` et `metrics/recall(M)` la même valeur que l'AR des
masques à 100 détections. `(B)` se comporte de la même façon pour les boîtes.
Tracer une paire de ces valeurs rapporte donc deux fois le même nombre.

## Exporter

<code-tabs name="export" />

Un artefact exporté se recharge par `LibreYOLO()` grâce au suffixe de son
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint
et renvoie le même `Results`. La couverture de la segmentation est plus étroite
que celle de la détection pour une même famille. La matrice de chaque page de
modèle est générée depuis l'ensemble validé et indique pourquoi une cible est
indisponible. Consultez la page [exporter et déployer](/docs/export) pour les
formats, leurs extras et leurs contraintes.
