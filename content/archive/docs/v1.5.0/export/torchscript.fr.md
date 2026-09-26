---
title: TorchScript
seo_title: Exporter vers TorchScript depuis LibreYOLO
description: "Exportez un modèle LibreYOLO vers TorchScript\_: une archive .torchscript tracée qui embarque les métadonnées LibreYOLO, chargeable depuis Python ou libtorch."
lead: >-
  TorchScript est le format de graphe sérialisé propre à PyTorch. LibreYOLO
  trace le modèle avec torch.jit.trace et enregistre le résultat avec un fichier
  supplémentaire libreyolo_metadata.json, de sorte que l'archive transporte la
  famille, la tâche, les noms de classes et la taille d'entrée.
keywords:
  - exporter yolo torchscript
  - torch.jit.trace
  - torch.jit.load
  - déploiement libtorch
  - métadonnées torchscript
  - extra_files
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="torchscript")
    mono: true
  - label: Écrit
    value: >-
      Une archive .torchscript avec un fichier supplémentaire
      libreyolo_metadata.json
  - label: Extra
    value: Aucun. TorchScript est fourni avec PyTorch.
  - label: Se recharge avec
    value: LibreYOLO("weights/LibreYOLO9t.torchscript")
    mono: true
  - label: Formes
    value: Fixes. Le graphe est tracé pour une seule forme d'entrée.
  - label: Précision
    value: 'FP32, FP16 (half=True). Pas d''INT8.'
verification: >-
  Lu depuis libreyolo/export/torchscript.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py et libreyolo/backends/torchscript.py sur la
  branche dev.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        pip install libreyolo
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Écrit weights/LibreYOLO9t.torchscript
        path = model.export(format="torchscript")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format torchscript
    - label: Arguments
      language: python
      code: >
        model.export(
            format="torchscript",
            imgsz=640,        # int, ou (hauteur, largeur)
            batch=1,
            half=False,       # poids et activations en FP16
            device=None,      # None trace sur CPU pour ce format
            output_path=None, # None écrit weights/<stem>.torchscript
        )


        # dynamic est accepté mais l'archive est toujours tracée à forme fixe,

        # et les métadonnées embarquées indiquent dynamic=False dans tous les
        cas.
  run:
    - label: Via LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.torchscript")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: PyTorch brut
      language: python
      code: |
        import json

        import torch

        extra_files = {"libreyolo_metadata.json": ""}
        module = torch.jit.load(
            "weights/LibreYOLO9t.torchscript",
            map_location="cpu",
            _extra_files=extra_files,
        )
        module.eval()

        metadata = json.loads(extra_files["libreyolo_metadata.json"])
        print(metadata["model_family"], metadata["task"], metadata["imgsz"])

        # Le prétraitement et le post-traitement sont à votre charge ici.
        with torch.no_grad():
            out = module(torch.zeros(1, 3, 640, 640))
        print(out.shape if torch.is_tensor(out) else [t.shape for t in out])
  support:
    - label: Vérifier une famille et une tâche avant d'exporter
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 286a082969ccd604
---

## Installation

<code-tabs name="install" />

TorchScript n'exige rien de plus que l'installation de base, car `torch.jit` est
fourni avec PyTorch. C'est la seule cible d'export sans dépendance optionnelle et
sans convertisseur externe, ce qui en fait une première vérification utile quand
une chaîne d'outils plus longue échoue.

## Export

<code-tabs name="export" />

Le tracing s'exécute sur CPU sauf si un device est indiqué, et l'archive est
écrite dans `weights/` sous le nom de base du checkpoint quand `output_path` est
omis.

La vérification par retracing que `torch.jit.trace` effectue normalement est
désactivée. Plusieurs wrappers d'export mettent en cache des ancres dépendantes
de la forme pendant leur premier forward, si bien qu'un second tracing observe un
chemin Python différent alors même que le graphe à forme fixe enregistré est
correct. Les tests de parité valident directement le module sauvegardé.

Les métadonnées ne vivent pas dans un fichier annexe. `torch.jit.save` stocke
`libreyolo_metadata.json` à l'intérieur de l'archive, et `torch.jit.load` le
restitue via `_extra_files`.

## Exécuter l'artefact

<code-tabs name="run" />

`LibreYOLO()` s'aiguille sur le suffixe `.torchscript` et renvoie le même objet
`Results` que le checkpoint dont il provient. Avec `device="auto"`, le module est
mappé sur CUDA si disponible, puis MPS, puis CPU.

Le second exemple est le chemin pour un lecteur qui n'a pas LibreYOLO installé,
et pour un déploiement C++ via libtorch, où la même archive se charge avec
`torch::jit::load`. Le prétraitement, le décodage, le NMS et le redimensionnement
des coordonnées sont alors à votre charge. Le fichier supplémentaire de
métadonnées reste lisible, et c'est le seul endroit où les noms de classes
existent.

## Contraintes

Le graphe est un tracé pour une seule forme d'entrée. `dynamic=True` est accepté
par symétrie d'interface mais ne change rien, et les métadonnées embarquées
indiquent `dynamic=False` pour qu'un backend ne suppose jamais un axe qu'il ne
peut pas utiliser. Exportez une seconde archive pour une seconde résolution.

`half=True` convertit le modèle et l'entrée de tracing en FP16. Il n'y a pas de
chemin INT8 : `int8=True` lève `NotImplementedError` pendant la validation.

Un `imgsz` rectangulaire fonctionne pour les familles YOLO9, HRNet, NAFNet et
Real-ESRGAN, et il est refusé pour les familles dont le contrat impose un carré
fixe.

Cinq combinaisons sont refusées avant le tracing. La segmentation YOLO9, parce
que YOLO9 ne fait que de la détection dans LibreYOLO. La segmentation
RTMDet-Ins, dont le décodage de masques à noyaux dynamiques n'a pas de contrat
pour les runtimes exportés. La détection SSD, Faster R-CNN et RetinaNet, dont les
graphes à longueur variable ou à ancres dynamiques n'ont de preuves de parité
qu'à travers le contrat ONNX Runtime.

Pour la grille complète des familles et des tâches, voir
[la matrice d'export](/docs/reference/export-matrix). Pour une seule
combinaison :

<code-tabs name="support" />
