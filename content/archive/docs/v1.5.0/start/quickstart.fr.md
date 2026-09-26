---
title: Prise en main
seo_title: Prise en main de LibreYOLO
description: >-
  Exécuter un détecteur sur une image, l'affiner sur un petit dataset et
  l'exporter vers TorchScript ou ONNX, entièrement sur CPU, en une dizaine de
  lignes de Python.
lead: >-
  Le parcours le plus court dans LibreYOLO : prédire sur une image, entraîner
  sur un petit dataset, puis exporter le résultat. Toutes les commandes
  présentées ici s'exécutent sur CPU.
keywords:
  - prise en main libreyolo
  - tutoriel libreyolo
  - prédiction libreyolo
  - entraînement libreyolo
  - export libreyolo
  - exemple yolo python
last_verified: 1.5.0
meta:
  - label: Installation
    value: pip install libreyolo
    mono: true
  - label: Checkpoint
    value: LibreYOLO9t.pt
    mono: true
  - label: Matériel
    value: Le CPU suffit pour tout le contenu de cette page
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Télécharge le checkpoint à la première utilisation, puis le met en
        cache dans weights/.

        model = LibreYOLO("LibreYOLO9t.pt")


        # Une image unique renvoie un objet Results.

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=yolo9-t save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Vidéos et flux
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # stream=True produit un Results par image au lieu de construire une
        liste.

        # Remplacez le chemin par l'indice d'une webcam, une URL RTSP ou un
        dossier.

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # coco8 est un dataset de 8 images inclus dans la bibliothèque. Il se

        # télécharge depuis une URL à la première utilisation, sans script à
        exécuter.

        results = model.train(
            data="coco8.yaml",
            epochs=1,
            imgsz=640,
            batch=4,
            device="cpu",
        )


        print(results["save_dir"])

        print(results["best_checkpoint"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=yolo9-t data=coco8.yaml \
          epochs=1 imgsz=640 batch=4 device=cpu
    - label: Valider
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() renvoie un dictionnaire ordinaire et non un objet.
        metrics = model.val(data="coco8.yaml", device="cpu")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
  export:
    - label: TorchScript
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # export() renvoie le chemin qu'elle a écrit.

        path = model.export(format="torchscript")

        print(path)


        # La fabrique s'oriente grâce au suffixe du fichier. L'artefact se
        recharge

        # donc comme un checkpoint et renvoie le même objet Results.

        exported = LibreYOLO(path)

        result = exported(SAMPLE_IMAGE)

        print(len(result.boxes))
    - label: ONNX
      language: bash
      code: |
        pip install "libreyolo[onnx]"
        libreyolo export model=yolo9-t format=onnx imgsz=640
source_hash: c11b6bdbf0b6fdf1
---

## Installer

```bash
pip install libreyolo
```

C'est tout ce dont les sections de prédiction et d'entraînement ci-dessous ont
besoin. L'exportation vers ONNX ajoute un extra. Consultez la page
[installer](/docs/install) pour obtenir la liste complète.

## Prédire

<code-tabs name="predict" />

`LibreYOLO()` est une fabrique. Elle lit le fichier, détermine à quelle famille
appartiennent les poids et renvoie le modèle de cette famille. Changer de
détecteur ne demande donc qu'une ligne. Lorsque `LibreYOLO9t.pt` est fourni sans
répertoire, la fabrique recherche `weights/LibreYOLO9t.pt` relativement au
répertoire de travail et le télécharge à cet emplacement s'il manque. Consultez
[checkpoints et poids](/docs/weights) pour connaître les règles de
téléchargement et le fonctionnement hors ligne.

`save=True` écrit une copie annotée sous `runs/detect/`, dans un répertoire
`predict` incrémenté à chaque exécution. L'objet `Results` renvoyé contient
`boxes`, tandis que `names` associe l'indice d'une classe à son étiquette. Le
chemin d'une seule image renvoie un `Results`. Un répertoire, une liste d'images
ou `stream=True` renvoie une liste ou un générateur de résultats.

## Entraîner

<code-tabs name="train" />

`data` est un fichier YAML de dataset. `coco8.yaml` est inclus dans la
bibliothèque, ce qui permet d'exécuter l'extrait tel quel. Un nom qui n'est pas
inclus est interprété comme un chemin. Les datasets sont recherchés sous
`~/datasets` ou sous `LIBREYOLO_DATASETS_DIR` lorsque cette variable est
définie.

Une exécution écrit ses fichiers dans `project/name`, par défaut dans un
répertoire sous `runs/train`, avec `weights/best.pt` et `weights/last.pt`.
`train()` renvoie un dictionnaire qui contient `save_dir`, `best_checkpoint`,
`last_checkpoint`, les pertes de chaque époque et les métriques de validation
de chaque époque. Le checkpoint entraîné se charge par `LibreYOLO()` exactement
comme le checkpoint pré-entraîné.

Toutes les familles ne peuvent pas être entraînées. Lorsqu'une famille est
limitée à l'inférence, `train()` déclenche une `NotImplementedError` et le
signale. La page [concepts fondamentaux](/docs/concepts) explique la
signification de chaque niveau de prise en charge.

## Exporter

<code-tabs name="export" />

TorchScript ne demande rien de plus que l'installation de base. Les autres
cibles possèdent chacune leur propre extra, et la couverture dépend de la
famille et de la tâche. Consultez [exporter et déployer](/docs/export).

Les arguments acceptés par tous les formats comprennent `imgsz` (un entier ou
une paire hauteur-largeur), `batch` (1 par défaut), `half`, `int8` avec un
fichier YAML `data` pour l'étalonnage, `dynamic` (true par défaut), `simplify`
(true par défaut), `opset`, `device` et `output_path`. Lorsque `output_path`
est omis, le fichier est écrit sous `weights/` avec un nom dérivé du
checkpoint.

## Pour aller plus loin

- [Concepts fondamentaux](/docs/concepts) pour les tâches, les familles, les tailles et les noms des checkpoints.
- [Checkpoints et poids](/docs/weights) pour le téléchargement automatique, l'utilisation hors ligne et la sécurité du chargement.
- [Importer des poids existants](/docs/migrate) si vous possédez déjà un checkpoint issu d'un projet amont.
- [Tous les modèles](/docs/models) pour choisir la famille adaptée à votre problème.
- [Entraîner](/docs/train), [Prédire](/docs/predict) et [Exporter](/docs/export) pour les workflows complets.
