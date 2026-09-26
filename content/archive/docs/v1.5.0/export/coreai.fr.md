---
title: Core AI
seo_title: Exporter vers Apple Core AI depuis LibreYOLO
description: "Exportez un modèle LibreYOLO vers un asset .aimodel Apple Core AI\_: macOS uniquement, canevas fixe, FP32, et le contrat d'ordre des sorties nommées que les consommateurs doivent respecter."
lead: >-
  Core AI est la pile d'inférence sur appareil d'Apple. LibreYOLO capture le
  modèle avec torch.export, le fait descendre à travers le convertisseur Core AI
  et écrit un asset .aimodel portant les métadonnées du modèle et les noms des
  sorties exportées.
keywords:
  - exporter libreyolo core ai
  - aimodel
  - coreai-torch
  - torch.export apple
  - inférence sur appareil apple
  - coreai_output_names
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="coreai")
    mono: true
  - label: Écrit
    value: Un asset .aimodel avec les métadonnées attachées
  - label: Extra
    value: 'pip install "libreyolo[coreai]"'
    mono: true
  - label: Rechargement
    value: >-
      Pas via LibreYOLO. Les consommateurs utilisent directement le runtime Core
      AI.
  - label: Formes
    value: Canevas fixe. dynamic=True lève NotImplementedError.
  - label: Précision
    value: FP32 uniquement. half=True et int8=True sont refusés.
  - label: Nécessite
    value: >-
      macOS. La toolchain ne convertit ni ne s'exécute ailleurs, et coreai-torch
      fige torch en 2.11.x.
verification: >-
  Lu depuis libreyolo/export/coreai.py, libreyolo/export/coreai_compat.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py et pyproject.toml
  sur la branche dev.
snippets:
  install:
    - label: 'Installation, sur macOS'
      language: bash
      code: |
        # Volontairement hors de tous les extras agrégés, car coreai-torch fige
        # torch en 2.11.x et forcerait tout l'environnement sur cette version.
        pip install "libreyolo[coreai]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Écrit weights/LibreYOLO9t.aimodel
        path = model.export(format="coreai", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreai --imgsz 640
    - label: Arguments
      language: python
      code: |
        model.export(
            format="coreai",
            imgsz=640,        # int, ou (hauteur, largeur), le canevas d'exécution
            batch=1,
            output_path=None, # None écrit weights/<stem>.aimodel
        )

        # dynamic=True lève NotImplementedError.
        # half=True et int8=True sont refusés pendant la validation.
  outputs:
    - label: Lire l'ordre des sorties avant de brancher un consommateur
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")

        model.export(format="coreai", imgsz=640)


        # Les métadonnées de l'asset enregistrent les noms des sorties
        exportées,

        # dans l'ordre du graphe, sous "coreai_output_names". Faites
        correspondre

        # par nom le dictionnaire renvoyé par Core AI avec cette liste, jamais

        # par position avec le tuple du mode eager.
  support:
    - label: Vérifier une famille et une tâche avant d'exporter
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: a35bfeafac6d6966
---

## Installation

Ce format est réservé à macOS. La dépendance `coreai-torch` porte un marqueur
`sys_platform == 'darwin'`, et la toolchain ne convertit ni ne s'exécute nulle
part ailleurs.

<code-tabs name="install" />

L'extra reste en dehors de tous les extras agrégés, y compris `libreyolo[all]`,
parce que `coreai-torch` fige torch sur la série 2.11. Installez-le dans un
environnement que vous acceptez de contraindre à ce couple.

## Export

<code-tabs name="export" />

La capture passe par `torch.export`, une véritable capture de graphe avec des
guards, et non une trace unique enregistrée. C'est plus strict que le chemin
Core ML : les lectures de scalaires côté hôte et le contrôle de flux dépendant
des données sont rejetés au lieu d'être figés en silence, ce qui explique que
quelques familles soient bloquées ici avec un échec de capture enregistré.

Trois étapes de préparation s'exécutent dans une portée qui restaure le modèle
vivant de l'appelant, que l'export réussisse ou échoue. Les familles dérivées de
Darknet voient leur batch normalization d'inférence repliée exactement dans les
convolutions précédentes, parce que Core AI 0.4.1 ne préserve pas la formule de
Darknet avec l'epsilon après la racine carrée. Les familles à grille et à ancres
voient leurs ancres figées pour le canevas fixe. RF-DETR voit son position
embedding recalculé pour le canevas demandé, en réexécutant le chemin de
pré-calcul du modèle lui-même, parce que le convertisseur n'a pas de lowering
pour `aten._upsample_bicubic2d_aa`.

Le lowering intègre à la table de décompositions la décomposition de référence de
PyTorch pour `aten.grid_sampler_2d`, puisque le convertisseur Core AI n'a pas de
lowering pour le sampler de deformable attention qu'utilisent les familles DETR.

Les assets déclarent un OS minimum de v27, la seule valeur que propose la
toolchain. Cela conditionne le déploiement, pas la conversion : la conversion et
l'exécution côté Python fonctionnent sur des macOS antérieurs grâce au runtime
embarqué dans le wheel, mais les résultats numériques diffèrent d'une version
d'OS à l'autre, si bien que la parité enregistrée est mesurée sur macOS 27.

## Exécuter l'artefact

Il n'y a pas d'entrée Core AI dans `libreyolo/backends`, donc `LibreYOLO()` ne
charge pas un `.aimodel`. Les consommateurs utilisent directement le runtime
Core AI, et le prétraitement, le décodage, la NMS et le redimensionnement des
coordonnées leur reviennent. Une ligne validée dans la matrice de support affirme
que le graphe exporté calcule les mêmes nombres que la référence, pas que
`predict` saura l'exécuter.

La seule chose qu'un consommateur ne peut pas retrouver par lui-même, c'est
l'ordre des sorties :

<code-tabs name="outputs" />

Core AI renvoie un dictionnaire nommé dont l'ordre des clés ne correspond ni à
l'ordre du tuple du forward en mode eager, ni à quoi que ce soit de devinable.
Les noms exportés sont écrits dans les métadonnées de l'asset sous
`coreai_output_names` précisément pour cette raison. Faites la correspondance par
nom.

## Contraintes

Canevas fixe, FP32, batch tel qu'exporté. `dynamic=True` lève
`NotImplementedError`, et `half=True` et `int8=True` sont refusés pendant la
validation.

La couverture est large côté conversion. Les combinaisons validées incluent les
familles YOLO9, YOLOX, YOLO7, les quatre détecteurs de l'ère Darknet, YOLO-NAS,
PicoDet, RTMDet, RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM, DEIMv2, EC et la
détection RF-DETR ; les quatre familles de classification CNN ainsi que CLIP et
SigLIP2 à classes figées ; Depth Anything V2 et ZipDepth ; la restauration
NAFNet et Real-ESRGAN ; la segmentation sémantique PIDNet et LingBotVision ; et
la détection de points FOMO. Chacune porte son propre contexte enregistré,
qu'affiche `libreyolo formats`.

Bloquées, avec le motif enregistré pour chaque combinaison :

| Combinaison | Motif |
|---|---|
| Segmentation sémantique EoMT | La capture stricte échoue avec `GuardOnDataDependentSymNode` : quelque chose dans le chemin des masques lit une valeur dans un tenseur et branche dessus |
| Segmentation sémantique SegFormer | Le chemin de capture n'a pas été évalué, et ses poids publiés sont non commerciaux quel que soit le format |
| Regard L2CS | Le modèle lui-même ne prend en charge qu'ONNX, TorchScript, ExecuTorch, TensorRT et OpenVINO, ce qui est une décision côté modèle |
| Profondeur Depth Anything 3 | La famille refuse l'export pour tous les formats |

RF-DETR s'accompagne d'une réserve qu'il vaut mieux lire avant de comparer des
artefacts. Sa parité est enregistrée contre le graphe que prépare l'exportateur
Core AI lui-même, et non contre ONNX, et sur un canevas de 640 l'artefact ONNX de
RF-DETR diverge de ce graphe préparé. Le pré-calcul refait par Core AI préserve
le redimensionnement avec antialiasing qu'effectue le modèle en mode eager, alors
que le chemin ONNX désactive l'antialiasing. ONNX n'est donc pas une référence
valide pour cette famille sur un canevas non natif.

Pour le format Apple précédent, voir [Core ML](/docs/export/coreml). Pour la
grille complète des familles et des tâches, voir [la matrice
d'export](/docs/reference/export-matrix). Pour une seule combinaison :

<code-tabs name="support" />
