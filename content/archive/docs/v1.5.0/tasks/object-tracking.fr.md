---
title: Suivi d'objets
seo_title: Suivi d'objets dans LibreYOLO
description: >-
  Suivre des objets entre les images d'une vidéo dans LibreYOLO avec ByteTrack,
  BoT-SORT, OC-SORT ou Deep OC-SORT, sur tout modèle de détection, de
  segmentation ou de pose.
lead: >-
  Le suivi attribue une identité stable à chaque détection d'une image vidéo à
  l'autre. LibreYOLO ne le modélise pas comme une tâche dotée de ses propres
  poids : il s'agit d'un mode de prédiction, model.track(), qui applique le
  tracker choisi aux sorties par image d'un modèle de détection, de segmentation
  ou de pose.
keywords:
  - suivi objets python
  - suivi multi objets
  - bytetrack
  - botsort
  - ocsort
  - deep ocsort
  - identifiant suivi
  - suivi réidentification
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # track() est un générateur : un Results par image traitée.
        for result in model.track("video.mp4"):
            print(result.track_id)        # tenseur entier (N,), aligné avec les boîtes
            print(result.boxes.xyxy)
    - label: Choisir un tracker
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # "bytetrack" (par défaut), "botsort", "ocsort" ou "deepocsort".
        for result in model.track("video.mp4", tracker="botsort"):
            print(result.track_id)
    - label: Enregistrer une vidéo annotée
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Sans output_path, le fichier est placé dans
        runs/track/<video_stem>.mp4.

        for result in model.track("video.mp4", save=True, vid_stride=2):
            pass
    - label: Régler un tracker
      language: python
      code: >
        from libreyolo import BoTSortConfig, LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Le type de configuration sélectionne le tracker. tracker= est donc
        superflu ici.

        config = BoTSortConfig(track_buffer=60, frame_rate=25, enable_cmc=False)

        for result in model.track("video.mp4", tracker_config=config):
            print(result.track_id)

        # Vous pouvez aussi transmettre les mêmes champs comme arguments nommés

        # et laisser track() construire la configuration.

        for result in model.track("video.mp4", tracker="botsort",
        track_buffer=60):
            print(result.track_id)
source_hash: f1fa7dcf60597d6b
---

## Définition

Le suivi ne fait pas partie des clés de tâche de LibreYOLO et il n'existe aucun
checkpoint de suivi à télécharger. Il s'agit d'une méthode du modèle,
`model.track(source)`, qui exécute la détection sur chaque image et associe les
résultats au fil du temps. Cette méthode est un générateur : elle produit un
`Results` par image traitée, avec `result.track_id` défini comme un tenseur
entier `(N,)` aligné sur `result.boxes`. Les mêmes identifiants sont également
disponibles dans `result.boxes.id`.

Seuls les objets confirmés et actuellement suivis sont produits. Une trajectoire
perdue par l'association reste active pendant un nombre d'images configurable
avant d'être supprimée : `track_buffer` pour ByteTrack et BoT-SORT, et
`max_age` pour les deux variantes d'OC-SORT. Un objet retrouvé pendant cette
fenêtre conserve donc son identifiant d'origine.

Comme l'association intervient après la détection, les autres charges utiles de
l'image sont préservées. Le `Results` suivi est le `Results` de détection
limité aux lignes mises en correspondance. Les masques et les points clés
accompagnent donc les boîtes.

## Modèles

Une exécution de suivi combine deux choix indépendants : le modèle qui produit
les boîtes de chaque image et le tracker qui les relie.

Tout modèle LibreYOLO natif dont la tâche est la détection, la segmentation ou
la pose expose `track()`. Le choix du détecteur suit donc les règles habituelles.
Consultez l'[index des modèles](/docs/models) pour la liste complète, ou
commencez par [YOLO9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr),
[D-FINE](/docs/models/d-fine) ou [RTMDet](/docs/models/rtmdet). Les tâches dont
les résultats ne possèdent aucune boîte à associer refusent l'appel au lieu de
renvoyer des identifiants dépourvus de sens. La classification, les boîtes
orientées, les points, la profondeur, les normales de surface, les contours, la
segmentation sémantique et panoptique, la restauration, l'OCR et le maillage
corporel déclenchent tous une erreur dans `track()`.

Deux niveaux de modèles LibreYOLO le refusent également. Les modèles chargés par
`LibreSAM` sont des segmenteurs d'images, tandis que ceux chargés par
`LibreOpenVocab` sont des détecteurs par image. Tous déclenchent une erreur dans
`track()` et s'emploient avec `predict()` sur chaque image.

Le suivi s'exécute sur les modèles PyTorch natifs. Un artefact exporté chargé
par `LibreYOLO("model.onnx")` renvoie un objet de backend d'exécution qui
possède `predict()`, mais pas `track()`.

La bibliothèque contient quatre trackers, sélectionnés par l'argument
`tracker` :

`"bytetrack"` est la valeur par défaut. Il repose uniquement sur le mouvement,
avec un filtre de Kalman et une association en trois étapes : d'abord les
détections à haute confiance, puis une seconde passe qui permet aux détections
à faible confiance de retrouver une trajectoire existante avant d'être
écartées, et enfin les trajectoires non confirmées. Il se configure avec
`TrackConfig`.

`"botsort"` conserve le cycle de vie en trois étapes de ByteTrack, mais emploie
un état de Kalman centre-largeur-hauteur et compense le mouvement de la caméra
sur les trajectoires prédites avant leur mise en correspondance. Il s'agit de
la variante de BoT-SORT fondée uniquement sur le mouvement, sans modèle
d'apparence. Elle se configure avec `BoTSortConfig`, qui ajoute `enable_cmc`,
`cmc_method` et `cmc_downscale`.

`"ocsort"` repose également uniquement sur le mouvement. Il ajoute à son coût
d'association un terme de direction de la vitesse, une seconde passe
d'association avec la dernière observation réelle de chaque trajectoire et un
lissage de l'état de Kalman le long d'une trajectoire virtuelle lorsqu'une
trajectoire est retrouvée. Il se configure avec `OCSortConfig`.

`"deepocsort"` étend OC-SORT avec l'apparence. Chaque trajectoire conserve une
moyenne mobile des embeddings de réidentification pondérée par la confiance, et
un terme de similarité cosinus rejoint le coût d'association. Les identités
survivent ainsi aux longues occultations et aux cibles qui se croisent. Cette
méthode coûte une propagation d'un petit réseau d'embeddings par image, et ses
poids OSNet se téléchargent à la première utilisation. Elle se configure avec
`DeepOCSortConfig`.

## Prédire

<code-tabs name="predict" />

`track_conf` définit le seuil de la première étape d'association :
`track_high_thresh` pour ByteTrack et BoT-SORT, `det_thresh` pour OC-SORT et
Deep OC-SORT. Il ne s'agit pas de la valeur `conf` de `predict()`. Pour
ByteTrack, BoT-SORT et OC-SORT, le détecteur s'exécute en interne avec un seuil
plus faible afin que les détections incertaines restent disponibles pour la
passe de récupération. Deep OC-SORT exécute le détecteur directement au seuil
`det_thresh`. Pour ByteTrack et BoT-SORT, `track_conf` doit être supérieur ou
égal à `track_low_thresh`, qui vaut 0,1 par défaut.

Les réglages du tracker arrivent de deux façons. Transmettez une instance de
configuration à `tracker_config=`. Son type sélectionne alors le tracker et rend
`tracker=` superflu. Vous pouvez aussi transmettre les champs comme arguments
nommés et laisser `track()` construire la configuration du tracker indiqué.
Les clés inconnues produisent un avertissement au lieu d'être appliquées
silencieusement. Dans les deux cas, `track_conf` est ignoré dès que la clé de
mise en correspondance est définie explicitement.

Les autres arguments reproduisent ceux de la prédiction : `iou`, `imgsz`,
`classes`, `max_det`, `vid_stride`, `show` et `save` avec `output_path`. La
source est le chemin d'un fichier vidéo. Consultez la page
[prédiction](/docs/predict) pour la gestion des résultats.

## Entraîner

Les trackers ne s'entraînent pas. Trois des quatre sont de purs modèles de
mouvement dépourvus de tout paramètre appris. Le réseau d'apparence de
Deep OC-SORT est un checkpoint de réidentification publié qui se télécharge à
la première utilisation. Pour améliorer la qualité du suivi, améliorez le
détecteur ou réglez les seuils d'association ci-dessus.
