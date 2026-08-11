---
title: YOLOv9
families:
  - yolo9
seo_title: "YOLOv9\_: prédire, entraîner et exporter sous licence MIT"
description: >-
  Exécutez YOLOv9 dans LibreYOLO, y compris sa tête de bout en bout sans NMS et
  sa tête de stride 4 pour les petits objets. Installez, prédisez, entraînez,
  validez et exportez.
lead: "YOLOv9 est un détecteur convolutionnel mono-étape\_: une passe attribue un score à une grille dense de bounding boxes et la NMS supprime les doublons. LibreYOLO en propose trois variantes, dont une sans étape de NMS."
keywords:
  - YOLOv9
  - YOLO9
  - détection d'objets
  - détection sans NMS
  - détection de bout en bout
  - détection petits objets
  - programmable gradient information
  - GELAN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Sans NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Même appel, autre checkpoint. La tête de bout en bout renvoie ses
        propres

        # prédictions aux scores les plus élevés, aucune NMS ne s'exécute donc
        et iou est ignoré.

        model = LibreYOLO("LibreYOLO9E2Es.pt")

        result = model(SAMPLE_IMAGE, conf=0.25, max_det=300)


        print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Petits objets
      language: python
      code: >
        from libreyolo import LibreYOLO9P2


        # La variante de stride 4 n'a pas de checkpoint COCO propre, utilisez
        donc le nom

        # d'un checkpoint de détection de base : son backbone et son neck se
        chargent sans

        # modification et la tour de tête de stride 4 part d'une initialisation
        aléatoire.

        model = LibreYOLO9P2(None, size="s")

        model.train(data="my-dataset.yaml", epochs=100,
        pretrained="LibreYOLO9s.pt")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=my-dataset.yaml
    - label: Sur COCO
      language: bash
      code: >
        # Le YAML COCO fourni contient un script de téléchargement intégré, il
        nécessite donc

        # une autorisation explicite sauf si le dataset est déjà présent
        localement.

        libreyolo val model=LibreYOLO9c.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: Avec la NMS dans le graphe
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx nms=True \
          conf=0.25 iou=0.45 max_det=300
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La fabrique détermine le chemin selon le suffixe du fichier ; un
        artefact exporté se charge

        # donc comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreYOLO9s.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: eaa6023a4a0b9e71
---

## Installer

YOLOv9 ne nécessite aucun extra au-delà du package de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et
mis en cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui de toutes les familles, le remplacement du
détecteur ne demande donc de modifier qu'une ligne. Sur les modèles de base et
de stride 4, `conf` définit le seuil de confiance et `iou` le seuil de NMS. Le
modèle de bout en bout n'exécute aucune NMS et ignore `iou`\u00a0; `conf` et
`max_det` déterminent donc sa sortie. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Variantes

Trois variantes partagent un backbone. Toutes trois effectuent uniquement de
la détection et acceptent les mêmes arguments.

Le modèle de base prédit sur trois échelles de caractéristiques et élimine les
bounding boxes dupliquées avec la NMS.

Le modèle de bout en bout conserve cette tête et lui ajoute une branche
d'appariement un-à-un. L'inférence lit uniquement cette branche et conserve
ses prédictions aux scores les plus élevés, aucune NMS n'est donc exécutée.
Choisissez-le lorsque le runtime de déploiement ne possède pas d'opérateur NMS.

Le modèle de stride 4 expose un niveau plus haut dans le backbone, prolonge le
neck jusqu'à celui-ci et prédit sur quatre échelles au lieu de trois. L'échelle
supplémentaire cible les objets qui couvrent peu de pixels\u00a0; l'unique checkpoint
publié pour ce modèle est entraîné sur des images aériennes. Les checkpoints
de détection de base peuvent y être transférés\u00a0: le backbone et le neck sont
chargés sans modification, les trois tours de tête pré-entraînées sont
décalées d'un emplacement et la tour de stride 4 part d'une initialisation
aléatoire.

<benchmark-table task="detect" />

<va-embed />

## Entraîner

<code-tabs name="train" />

`pretrained` détermine le point de départ de l'exécution. Transmettez `True`
pour charger le checkpoint publié du même modèle et de la même taille, ou un
nom ou chemin pour tout autre fichier. Les tenseurs dont la forme ne
correspond pas sont ignorés plutôt que refusés, et l'exécution consigne le
nombre de tenseurs chargés. Un checkpoint entraîné avec un nombre de classes
différent reste donc un point de départ utilisable.

Le modèle de stride 4 n'a aucun checkpoint COCO publié qui lui soit propre.
Pour celui-ci, `True` se résout donc en un fichier inexistant et le
téléchargement échoue. Indiquez plutôt le nom d'un checkpoint de détection de
base.

Consultez l'[entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurés sur tout dataset au format utilisé
pour l'entraînement.

<code-tabs name="val" />

## Exporter

<export-matrix />

Une coche s'applique aux trois variantes\u00a0; lorsqu'elles diffèrent, la matrice
indique la prise en charge la plus faible des trois.

Un artefact exporté se recharge dans `LibreYOLO()` grâce au suffixe de son
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint
et renvoie les mêmes `Results`. Vous pouvez également exécuter le graphe dans
un runtime nu, sans installer LibreYOLO, mais vous devez alors écrire vous-même
le prétraitement et le post-traitement.

Pour le modèle de détection de base, la moitié post-traitement peut être
intégrée au graphe. `nms=True` lors d'un export ONNX place la suppression dans
le modèle et la première sortie devient un tenseur fixe `(1, max_det, 6)` dont
les lignes sont `x1, y1, x2, y2, score, class`, complétées par des zéros après
le nombre de détections. Ce graphe utilise un batch de 1 et ne comporte aucun
axe dynamique. Les modèles de bout en bout et de stride 4 n'acceptent pas ce
paramètre.

Chaque format installe un extra différent et accepte quelques arguments qui
lui sont propres. Ces deux aspects sont décrits sur la page du format concerné.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box>

Un checkpoint présenté ici n'est pas sous licence MIT. Le modèle de stride 4
entraîné sur VisDrone2019-DET hérite des conditions CC BY-NC-SA 3.0 de ce
dataset\u00a0: usage non commercial uniquement, partage dans les mêmes conditions
de tout élément qui en dérive, hors de la licence permissive utilisée par le
reste de cette famille. Il prédit les classes aériennes VisDrone plutôt que
celles de COCO. La bibliothèque affiche toutes ces informations avant de
télécharger le fichier.

</provenance-box>

## Citation

<citation-block />

