---
title: Estimation du regard
seo_title: Estimation du regard dans LibreYOLO
description: >-
  Estimer l'inclinaison et le lacet du regard pour chaque visage dans LibreYOLO.
  Prédire depuis Python ou la CLI, lire les angles en radians et exporter la
  tête de regard vers ONNX.
lead: >-
  L'estimation du regard renvoie une direction d'observation pour chaque visage
  d'une image. LibreYOLO la modélise comme une tâche en deux étapes : un
  détecteur de visages s'exécute d'abord, puis une tête de regard lit
  l'inclinaison et le lacet dans chaque recadrage de visage obtenu.
keywords:
  - estimation regard python
  - suivi des yeux
  - inclinaison lacet regard
  - L2CS-Net
  - direction du regard
  - pose de la tête
  - tâche gaze libreyolo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sans face_detector, la prédiction revient au détecteur inclus dans
        # OpenCV. Rien d'autre que le checkpoint n'est donc téléchargé.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        gaze = result.gaze
        print(gaze.pitch, gaze.yaw)              # radians, une ligne par visage
        print(gaze.pitch_deg, gaze.yaw_deg)      # les mêmes angles en degrés
        print(gaze.direction_3d)                 # vecteurs unitaires (N, 3)
    - label: CLI
      language: bash
      code: >
        # Contrairement au parcours Python, la CLI n'a aucun repli automatique :

        # les modèles de regard exigent un détecteur de visages explicite, qui

        # doit être un détecteur LibreYOLO dont les boîtes représentent des
        visages.

        libreyolo predict model=LibreL2CSr50.pt source=photo.jpg
        face_detector=face-detector.pt save=True
    - label: Choisir la source des visages
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreL2CSr50.pt")


        # Transmettez à la tête de regard les boîtes d'un détecteur déjà
        exécuté.

        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])


        # Ou indiquez l'un des détecteurs inclus.

        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
source_hash: 22aa3c3d87b0c730
---

## Définition

L'estimation du regard renvoie deux angles par visage. `result.gaze` est une
charge utile `Gaze` de forme `(N, 2)`, avec l'inclinaison dans la colonne 0 et
le lacet dans la colonne 1, en radians. Ses lignes sont alignées avec
`result.boxes`, les boîtes des visages détectés. La convention est celle de
L2CS-Net : un lacet positif oriente le regard vers la gauche du sujet et une
inclinaison positive l'oriente vers le bas.

La même charge utile expose `pitch_deg` et `yaw_deg` pour les degrés, ainsi que
`direction_3d`, un vecteur unitaire `(N, 3)` dans le repère de la caméra, dont
les colonnes sont `(x, y, z)`.

Comme la tâche comporte deux étapes, une prédiction dépend de deux modèles. Les
visages manqués par le détecteur n'ont aucune ligne de regard, et les boîtes mal
placées produisent des angles à partir d'un visage mal recadré. La clé de tâche
canonique est `gaze`. `gaze-estimation` est normalisé vers celle-ci.

## Modèles

[L2CS-Net](/docs/models/l2cs) est l'unique famille qui couvre cette tâche. Elle
associe un tronc ResNet à deux têtes parallèles de classification par intervalles
angulaires, l'une pour l'inclinaison et l'autre pour le lacet, sur des
recadrages de visage de 448 x 448. L'architecture prend en charge cinq
profondeurs de backbone, et l'une d'elles, ResNet-50, possède un checkpoint
publié.

Les poids sont soumis à une restriction de licence. Ils ont été entraînés sur
Gaze360, dont la licence autorise uniquement la recherche et l'usage non
commercial et interdit la redistribution. LibreYOLO ne reproduit donc aucun
fichier de cette famille. Le seul checkpoint que la bibliothèque peut récupérer
automatiquement provient directement de la distribution Google Drive des
auteurs, par `gdown`, après affichage des conditions de licence. Lisez la page
[L2CS-Net](/docs/models/l2cs) avant tout déploiement.

Ce parcours de téléchargement nécessite l'extra `gaze` :

```bash
pip install "libreyolo[gaze]"
```

Sans cet extra, la bibliothèque affiche des instructions de téléchargement
manuel au lieu de tenter le transfert. La prédiction ou l'exportation d'un
checkpoint que vous possédez déjà ne nécessite aucun extra.

## Prédire

<code-tabs name="predict" />

La source des visages se choisit de trois manières. `face_boxes` transmet des
boîtes déjà calculées et ignore la détection. `face_detector` accepte `"auto"`,
`"haar"`, `"yunet"`, un modèle de détection LibreYOLO ou une simple fonction,
et peut être défini sur le constructeur ou à chaque appel. S'il est omis en
Python, la prédiction revient au détecteur inclus dans OpenCV. Un appel simple
fonctionne donc sans raccordement. Avec OpenCV 4, il s'agit de la cascade Haar
incluse dans la wheel, qui ne demande aucun téléchargement. Avec OpenCV 5, où
l'API Haar a été supprimée, il s'agit de YuNet, qui récupère une fois un petit
fichier de modèle depuis OpenCV Zoo.

La CLI ne partage pas ce mécanisme de repli. `libreyolo predict` refuse un
modèle de regard dépourvu de `face_detector=`. La valeur attendue est le nom
d'un détecteur LibreYOLO ou le chemin d'un checkpoint. Consultez la page
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Entraîner

Aucune famille de cette tâche ne s'entraîne dans LibreYOLO.
`LibreL2CS.train()` déclenche une erreur. Entraînez le modèle dans le projet
L2CS-Net amont, puis chargez ici le dictionnaire d'état obtenu.

## Valider

La validation sur des datasets de vérité terrain du regard n'entre pas dans le
périmètre et `val()` déclenche une erreur au lieu de renvoyer des métriques
qu'elle n'a pas calculées. Cette tâche ne possède aucun dictionnaire
`metrics/`. Effectuez l'évaluation dans le projet amont, sur le dataset utilisé
pour entraîner le checkpoint.

## Exporter

<code-tabs name="export" />

Le contrat d'exportation du regard couvre ONNX, TorchScript, ExecuTorch,
TensorRT et OpenVINO. Seuls le tronc ResNet et les deux têtes d'intervalles
angulaires quittent la bibliothèque. Le graphe reçoit un recadrage de visage
448 x 448 prétraité et renvoie les logits bruts du lacet et de l'inclinaison.
La détection des visages, le recadrage, le softmax, l'espérance des intervalles
et la conversion en angles restent tous en Python, dans
`libreyolo.models.l2cs.utils`. Consultez la page [exportation](/docs/export)
pour les formats et leurs arguments.
