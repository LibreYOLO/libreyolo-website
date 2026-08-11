---
title: Estimation de la pose
seo_title: Estimation de la pose dans LibreYOLO
description: >-
  Prédire des points clés par instance dans LibreYOLO : familles qui couvrent la
  tâche, format des annotations et appels de prédiction, d'entraînement, de
  validation et d'exportation.
lead: >-
  L'estimation de la pose localise chaque instance et renvoie pour elle un
  ensemble ordonné de points clés nommés. La sortie décrit ainsi la structure
  interne de l'objet plutôt que sa seule étendue. La clé de tâche est pose.
keywords:
  - estimation pose python
  - détection points clés
  - modèle pose humaine
  - points clés COCO
  - mAP OKS
  - entraîner modèle pose
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Le suffixe -pose du nom de fichier sélectionne la tête de points clés.
        # Aucun argument task n'est donc nécessaire.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy.shape)   # coordonnées en pixels (N, K, 2)
        print(result.boxes.xyxy.shape)     # (N, 4), les mêmes N instances
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs-pose.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Points clés visibles uniquement
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreECs-pose.pt")(SAMPLE_IMAGE)
        kpts = result.keypoints

        # .has_visible est dérivé de la troisième colonne des points clés.
        # Il est entièrement vrai si le checkpoint ne prédit que (x, y).
        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: Approche descendante
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # HRNet est descendant : il recadre d'abord chaque personne. Sans source

        # de personnes, il s'associe à un détecteur LibreYOLO9t et journalise ce
        choix.

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # coco8-pose.yaml contient un script de téléchargement intégré. Il
        nécessite

        # une autorisation explicite si les données ne sont pas déjà locales.

        model = LibreYOLO("LibreECs-pose.pt")

        model.train(
            data="coco8-pose.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            allow_download_scripts=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs-pose.pt data=coco8-pose.yaml \
          epochs=50 imgsz=640 batch=4 allow_download_scripts=True
    - label: Votre propre dataset
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml doit déclarer kpt_shape, et les lignes d'étiquettes doivent
        # contenir exactement 5 + K * D champs.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreECs-pose.pt")


        # val() renvoie un dictionnaire ordinaire et non un objet.

        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True)


        print(metrics["metrics/keypoints_mAP50-95"])

        print(metrics["metrics/keypoints_mAP50"],
        metrics["metrics/keypoints_mAP75"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs-pose.pt data=coco8-pose.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La fabrique s'oriente grâce au suffixe du fichier. Un artefact exporté
        # se charge comme un checkpoint et renvoie le même objet Results.
        model = LibreYOLO("LibreECs-pose.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
source_hash: 9de01d1f615bdf33
---

## Définition

L'estimation de la pose renvoie une structure, pas seulement une étendue. Chaque
instance reçoit toujours une boîte, une classe et un score, ainsi que `K` points
clés dans un ordre fixe. L'indice 5 désigne donc la même partie du corps pour
chaque instance et chaque image. L'ensemble d'étiquettes définit cet ordre. Rien
dans la sortie ne nomme un point clé.

`pose` est la clé de tâche canonique, et le suffixe `-pose` du nom d'un
checkpoint la sélectionne. L'argument `task=` est donc inutile lors du
chargement des poids publiés.

`predict()` remplit `result.keypoints` en parallèle de `result.boxes`. `.data`
a la forme `(N, K, 2)` ou `(N, K, 3)` et ses lignes sont alignées avec les
boîtes. L'instance `i` d'un côté correspond donc à l'instance `i` de l'autre.
`.xy` extrait les coordonnées en pixels et `.xyn` les normalise selon la taille
de l'image d'origine. `.conf` est la troisième colonne lorsque le checkpoint en
prédit une, et `None` dans le cas contraire. `.has_visible` est le masque
booléen qui en dérive. Il est entièrement vrai en l'absence de troisième
colonne.

Deux architectures produisent cette sortie. Un modèle en une étape prédit les
boîtes et les points clés en une seule passe. Un modèle descendant exécute
d'abord un détecteur, recadre chaque instance et régresse les points clés dans
le recadrage. Sa précision dépend donc du détecteur placé devant lui.

## Modèles

Trois familles peuvent être entraînées et effectuer des prédictions :
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter) et
[YOLO-NAS](/docs/models/yolo-nas), toutes en une étape. RF-DETR nécessite son
propre extra, `pip install "libreyolo[rfdetr]"`. RF-DETR et EdgeCrafter
publient des checkpoints de pose et s'affinent tous deux sur des datasets à une
classe réservée aux personnes. La tête de points clés d'EdgeCrafter est fixée à
la construction et refuse un dataset qui déclare un autre nombre, tandis que
RF-DETR réinitialise sa tête. YOLO-NAS récupère ses poids depuis le propre CDN
de Deci.AI sous une licence non commerciale, et LibreYOLO n'en publie aucun. Sa
tête de pose se reconstruit également pour un nouveau nombre de points clés.
C'est la seule des trois familles dont le nombre de classes n'est pas fixé à
un. Elle convient donc à un squelette multiclasse ou non humain, comme la pose
animale.

[HRNet](/docs/models/hrnet) constitue l'option descendante. Il prédit, valide et
exporte, tandis que sa méthode `train()` déclenche une
`NotImplementedError`. Sans source de personnes, il s'associe automatiquement
à un détecteur LibreYOLO9t. `cropped=True` traite l'image entière comme une
instance, `person_boxes=` reçoit des boîtes que vous possédez déjà et
`person_detector=` nomme un autre détecteur.

[SenseNova-Vision](/docs/models/sensenova-vision) émet également des points
clés. Il s'agit d'un modèle génératif guidé doté de sa propre fabrique,
`LibreVLM`, et de son propre extra. Si aucun vocabulaire n'est défini,
`set_task("pose")` revient à la catégorie personne. Ses poids sont réservés à
un usage non commercial, et sa latence par image est bien supérieure à celle
d'une tête de pose spécialisée, car chaque prédiction passe par un décodage de
diffusion.

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

Le nombre et l'ordre des points clés sont des propriétés du checkpoint, pas de
la bibliothèque. Un modèle entraîné sur un autre squelette renvoie donc une
autre valeur `K` et un autre sens pour chaque indice. Le contenu de la troisième
colonne des points clés dépend également du checkpoint. EdgeCrafter y écrit une
constante plutôt qu'un score par point. Il ne possède aucune tête de boîte, de
sorte que chacune de ses boîtes de pose est l'étendue englobante des propres
points clés de l'instance. Consultez la page [prédiction](/docs/predict) pour
les sources, le streaming et la gestion des résultats.

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

Une ligne est une ligne de détection à laquelle les points clés sont ajoutés :

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Le nombre de champs est exactement `5 + K * D`, où `D` est la deuxième valeur
de `kpt_shape`. Les coordonnées de la boîte et des points clés sont des nombres
flottants normalisés relativement à la largeur et à la hauteur de l'image
d'origine. La visibilité `v`, présente uniquement lorsque `D` vaut 3, est
`0`, `1` ou `2`.

Le fichier YAML ajoute deux clés au contrat partagé :

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

`kpt_shape` est obligatoire et vaut `[K, 2]` ou `[K, 3]`. `flip_idx` est une
permutation facultative de `0..K-1` qui indique, pour chaque point clé, son
indice après un retournement horizontal. Un poignet gauche reste ainsi un
poignet gauche. Si cette clé est omise, l'augmentation par retournement
horizontal est désactivée pour les points clés plutôt qu'appliquée dans le
mauvais ordre d'indices.

## Entraîner

<code-tabs name="train" />

L'entraînement continue depuis un checkpoint `-pose` publié, qui contient déjà
la tête de points clés. La tâche est lue depuis le checkpoint chargé, et non
depuis une option transmise à l'entraînement. Demander la pose ne transforme
donc pas un checkpoint de détection en exécution de pose. Le `kpt_shape` de
votre fichier YAML doit correspondre exactement à la tête d'EdgeCrafter, car
celle-ci est fixée à la construction. RF-DETR et YOLO-NAS redimensionnent en
revanche leur tête pour un autre nombre de points. Consultez la page
[entraînement](/docs/train) pour les datasets, les augmentations, le multi-GPU
et les systèmes de journalisation.

## Valider

`val()` renvoie un dictionnaire ordinaire de clés `metrics/`. L'évaluation
emploie les points clés COCO selon l'Object Keypoint Similarity (OKS), qui
pondère l'erreur de distance de chaque point par l'échelle de l'instance et par
une tolérance propre au point. Elle joue ainsi le rôle de l'IoU pour les boîtes.
Cette évaluation nécessite `pycocotools`, inclus dans l'installation de base.

<code-tabs name="val" />

`metrics/keypoints_mAP50-95` est le nombre principal, la précision moyenne
moyennée sur les seuils OKS de 0,50 à 0,95. L'entraînement l'emploie pour choisir
la meilleure époque. `metrics/keypoints_mAP50` et
`metrics/keypoints_mAP75` sont les versions à seuil unique, tandis que
`metrics/keypoints_mAP_M` et `metrics/keypoints_mAP_L` répartissent la moyenne
selon l'aire des instances moyennes et grandes. L'évaluation COCO des points
clés ne définit aucune catégorie petite. Les valeurs correspondantes de rappel
moyen sont `metrics/keypoints_AR50-95`, `metrics/keypoints_AR50`,
`metrics/keypoints_AR75`, `metrics/keypoints_AR_M` et
`metrics/keypoints_AR_L`. Chaque clé de cette tâche est préfixée par
`keypoints_`. Les clés `mAP` de boîtes renvoyées par un détecteur n'apparaissent
donc pas.

## Exporter

<code-tabs name="export" />

Un artefact exporté se recharge par `LibreYOLO()` grâce au suffixe de son
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint
et renvoie le même `Results`. La couverture des formats varie selon la famille.
La matrice de chaque page de modèle est générée depuis l'ensemble validé plutôt
que saisie manuellement. Consultez la page
[exporter et déployer](/docs/export) pour les formats, leurs extras et leurs
contraintes.
