---
title: ExecuTorch
seo_title: "Exporter vers ExecuTorch depuis LibreYOLO"
description: "Exportez un modèle LibreYOLO vers un programme .pte ExecuTorch avec délégation XNNPACK : forme fixe, batch 1, FP32, et le sidecar de métadonnées dont il a besoin."
lead: "ExecuTorch exécute des programmes PyTorch sur des cibles edge. LibreYOLO capture le modèle avec torch.export en mode strict, fait le lowering vers XNNPACK, et écrit le programme .pte et un sidecar de métadonnées JSON comme un seul ensemble."
keywords:
  - exporter yolo executorch
  - programme .pte
  - xnnpack partitioner
  - torch.export strict
  - executorch runtime
  - inférence pytorch sur edge
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="executorch")'
    mono: true
  - label: Écrit
    value: "Un programme .pte plus un sidecar de métadonnées .pte.json"
  - label: Extra
    value: 'pip install "libreyolo[executorch]"'
    mono: true
  - label: Se recharge avec
    value: 'LibreYOLO("weights/LibreYOLO9t.pte")'
    mono: true
  - label: Formes
    value: "Fixes. dynamic=True et batch != 1 sont refusés."
  - label: Précision
    value: "FP32 uniquement. half=True et int8=True sont refusés."
  - label: Délégué
    value: "XNNPACK, CPU. delegate='xnnpack' est la seule valeur acceptée."
verification: "Lu depuis libreyolo/export/executorch.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/executorch.py et pyproject.toml sur la branche dev."
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        # Volontairement hors de libreyolo[all], car ExecuTorch contraint la
        # version de Torch avec laquelle il peut être associé.
        pip install "libreyolo[executorch]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Écrit weights/LibreYOLO9t.pte et weights/LibreYOLO9t.pte.json
        path = model.export(format="executorch", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format executorch --imgsz 640
    - label: Arguments
      language: python
      code: |
        model.export(
            format="executorch",
            imgsz=640,             # int, ou (hauteur, largeur)
            batch=1,               # toute autre valeur lève ValueError
            dynamic=False,         # True lève ValueError
            delegate="xnnpack",    # la seule valeur acceptée
            device="cpu",          # tout autre périphérique lève ValueError
            output_path=None,      # None écrit weights/<stem>.pte
        )
  run:
    - label: Via LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.pte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Runtime ExecuTorch direct
      language: python
      code: |
        import json
        from pathlib import Path

        import torch
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        print(runtime.backend_registry.is_available("XnnpackBackend"))

        program = runtime.load_program(Path("weights/LibreYOLO9t.pte").read_bytes())
        method = program.load_method("forward")

        # Sur ce chemin, le prétraitement et le post-traitement sont à vous.
        outputs = method.execute((torch.zeros(1, 3, 640, 640),))
        print([tensor.shape for tensor in outputs])

        meta = json.load(open("weights/LibreYOLO9t.pte.json"))
        print(meta["model_family"], meta["task"], meta["executorch_delegate"])
  support:
    - label: Vérifier une famille et une tâche avant d'exporter
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Installation

<code-tabs name="install" />

Cet extra est délibérément en dehors de `libreyolo[all]`, parce qu'ExecuTorch fige
la version de Torch avec laquelle il fonctionne et que l'installer entraînerait
tout l'environnement sur ce couple. Installez-le dans un environnement que vous
acceptez de contraindre.

Sous Windows, l'étape de lowering appelle l'exécutable `flatc` livré avec
ExecuTorch. S'il n'est pas dans le `PATH`, l'export lève un `RuntimeError` qui le
signale, et la solution est de lancer depuis une Developer PowerShell de Visual
Studio 2022.

## Export

<code-tabs name="export" />

La capture est `torch.export.export(..., strict=True)`, c'est-à-dire une vraie
capture de graphe avec des guards plutôt qu'un trace enregistré. Les lectures de
scalaires sur l'hôte et le contrôle de flux dépendant des données sont refusés au
lieu d'être figés en silence, si bien que plusieurs familles échouent ici alors
qu'elles se tracent sans problème ailleurs ; les raisons sont consignées par
combinaison dans la matrice de support.

Le lowering exécute `to_edge_transform_and_lower` avec le partitioner XNNPACK. Si
le résultat ne contient aucune partition déléguée, l'export lève une erreur plutôt
que d'étiqueter comme XNNPACK un programme qui n'utilise que des kernels portables.

Le programme et le sidecar sont écrits ensemble. Les deux sont préparés, les deux
sont mis en place, et un échec revient à ce qui était là avant, si bien qu'une
paire incomplète n'atteint jamais le disque.

## Exécuter l'artefact

<code-tabs name="run" />

`LibreYOLO()` s'oriente sur le suffixe `.pte` et renvoie le même objet `Results`
que le checkpoint. Le sidecar est obligatoire au chargement : sans
`<program>.pte.json`, le backend lève `FileNotFoundError`, parce que le programme
ne porte de lui-même ni noms de classes, ni tâche, ni taille d'entrée. Le backend
vérifie aussi que le runtime installé fournit `XnnpackBackend` avant de charger, et
lit le programme depuis des octets plutôt que de mapper le fichier, ce qui évite de
garder un verrou de fichier Windows pendant toute la durée de vie du backend.

Le second snippet est le chemin du runtime direct. Le prétraitement, le décodage,
le NMS et le redimensionnement des coordonnées y deviennent votre affaire.

## Contraintes

Batch 1, forme fixe, FP32, CPU. `batch != 1` et `dynamic=True` lèvent tous deux
`ValueError` avant que l'export ne modifie quoi que ce soit, `half=True` et
`int8=True` sont refusés pendant la validation, et un périphérique autre que le CPU
est rejeté.

`delegate` accepte `"xnnpack"` et rien d'autre dans cette version.

Les exports de classification portent deux clés de métadonnées supplémentaires,
`crop_pct` et `interpolation`, pour que le runtime puisse reproduire la politique de
redimensionnement et de recadrage central de la famille.

Les entrées bloquées nomment l'échec concret plutôt qu'une catégorie. La détection
et la segmentation D-FINE atteignent une lecture de `ContextVar` non prise en charge
dans l'attention déformable sous capture strict, et forcer le chemin manuel du
grid-sample sérialise mais échoue ensuite à l'exécution sur un ordre de dimensions
invalide pour un tenseur délégué. DEIM et DEIMv2 se capturent, passent le lowering
et se sérialisent, puis échouent pendant l'exécution. La segmentation sémantique
EoMT échoue sur une expression symbolique dépendante des données dans le chemin des
masques. Le matting BiRefNet se capture en 1024 par 1024 mais n'a pas de variante
out pour `torchvision::deform_conv2d`. La restauration SwinIR se recharge puis
échoue dans `aten::alias_copy.out` sur des ordres de dimensions qui ne correspondent
pas.

Pour la grille complète des familles et des tâches, voir
[la matrice d'export](/docs/reference/export-matrix). Pour une seule combinaison :

<code-tabs name="support" />
