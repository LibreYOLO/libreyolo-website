---
title: Matrice d'export complète
seo_title: Matrice de prise en charge de l'export LibreYOLO et ses règles
description: "Méthode utilisée par LibreYOLO pour décider si une combinaison de famille, tâche et format peut être exportée\_: douze formats, trois niveaux, règles de repli et seuils de parité."
lead: >-
  La prise en charge de l'export est une recherche sur le triplet (famille,
  tâche, format). Cette page décrit la structure de cette matrice, les règles
  qui remplissent les cellules sans entrée explicite et la méthode pour
  interroger une combinaison donnée.
keywords:
  - support export libreyolo
  - matrice export
  - onnx tensorrt openvino tflite
  - commande libreyolo formats
  - seuil parité export
  - NotImplementedError export
last_verified: 1.5.0
verification: "Formats, niveaux, ordre de repli, blocages de tâches et familles et blocages NCNN lus dans libreyolo/export/support.py\_; alias et arguments partagés lus dans libreyolo/export/exporter.py\_; définitions des niveaux lues dans docs/adr/0011-export-support-tiers.md\_; seuils de parité lus dans docs/export_support.md, le tout en v1.5.0. Les cellules par combinaison ne sont pas retranscrites ici\_; interrogez-les avec l'extrait ci-dessous."
snippets:
  usage:
    - label: Interroger la matrice sans modèle
      language: python
      code: |
        from libreyolo.export.support import (
            EXPORT_FORMATS,
            get_support,
            validated_alternatives,
        )

        print(EXPORT_FORMATS)

        entry = get_support("yolo9", "detect", "onnx")
        print(entry.tier, entry.since)
        print(entry.constraint)

        print(validated_alternatives("yolo9", "detect"))
    - label: CLI
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
        libreyolo formats --family yolo9 --task detect --json
  export:
    - label: Exporter et lire un refus
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.export.support import get_support


        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.export(format="onnx"))


        # Vérifiez avant l'appel : une combinaison bloquée lève une erreur lors
        des vérifications préalables

        # et le message contient cette raison.

        blocked = get_support("domedetr", "detect", "onnx")

        print(blocked.tier)

        print(blocked.reason)
source_hash: 83de3289634888c6
---

## Structure de la matrice

La matrice utilise les clés `(family, task, format)`. Les clés de familles sont
les noms canoniques du registre de modèles, celles des tâches proviennent de
`libreyolo.tasks.TASKS`, et douze formats existent\u00a0:

`onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`,
`rknn`, `ncnn`, `tflite`, `coreml`, `coreai`.

`model.export(format=...)` accepte en outre deux alias\u00a0: `engine` pour
`tensorrt` et `litert` pour `tflite`, qui est le nom actuel de TensorFlow Lite.
Le format et le suffixe `.tflite` restent inchangés.

<code-tabs name="usage" />

Comme une cellule dépend de trois clés, la grille complète est volumineuse et
change à chaque version. Elle est générée plutôt qu'écrite à la main et se
trouve dans `docs/export_support.md` dans le dépôt de la bibliothèque.
Interrogez la matrice depuis Python ou le CLI au lieu d'en lire une copie.

## Trois niveaux

| Niveau | Signification |
|---|---|
| `validated` | La parité numérique est couverte par la CI ou une exécution nocturne documentée |
| `available` | La conversion est implémentée, mais aucune preuve de parité numérique dans le runtime n'a été consignée |
| `blocked` | Les vérifications préalables lèvent `NotImplementedError` avec une raison avant le traçage |

Les combinaisons validated et available poursuivent toutes deux sans demande
de confirmation ni avertissement général. Leurs preuves et contraintes
consignées restent visibles dans la documentation générée. Une combinaison
blocked échoue avant la vérification des dépendances, le chargement de la
calibration, le traçage ou la création de l'artefact.

L'ajout d'une entrée validated nécessite un test de parité et un champ `since`.

Un objet `SupportEntry` contient quatre champs\u00a0: `tier`, une chaîne `reason`,
la version `since` et une chaîne `constraint`. Cette contrainte est essentielle
lors de l'intégration\u00a0: une coche ne s'applique que dans les conditions qu'elle
nomme, généralement un canevas d'entrée fixe, un batch de 1, le FP32 et une
version précise du runtime.

## Méthode de décision d'une cellule

`get_support(family, task, fmt)` suit l'ordre ci-dessous. La première règle
correspondante gagne.

1. Une tâche inconnue ou un format absent des douze renvoie `blocked`.
2. Une entrée explicite `(family, task, format)` est renvoyée telle qu'elle est consignée.
3. Un blocage portant sur toute la famille renvoie `blocked` avec la raison de cette famille.
4. Un blocage portant sur toute la tâche renvoie `blocked` avec la raison de cette tâche.
5. Pour `ncnn`, une famille de la liste de blocage NCNN renvoie `blocked`.
6. `mnn` renvoie `blocked`\u00a0: aucun contrat de runtime n'existe pour cette famille et cette tâche.
7. `rknn` renvoie `blocked`. Dans cette version, RKNN se limite aux variantes de détection exactes testées dans le simulateur\u00a0: YOLO9-t, YOLO9-E2E-t, YOLO-NAS-s et PicoDet-s sur RK3588.
8. `tensorrt` et `openvino` renvoient `available`\u00a0: le chemin de conversion existe, mais la parité du runtime n'a pas été consignée pour cette famille et cette tâche.
9. `tflite`, `paddle`, `coreai` et `coreml` renvoient `blocked`, chacun avec sa propre raison.
10. Tout le reste renvoie `available`\u00a0: la conversion est implémentée, mais la parité numérique du runtime n'est pas consignée.

L'asymétrie des étapes 8 à 10 est délibérée. TensorRT et OpenVINO convertissent
de manière générique depuis ONNX, une combinaison non répertoriée mérite donc
d'être tentée. TFLite, Paddle, Core AI et CoreML nécessitent chacun un chemin
propre à la famille. Une combinaison non répertoriée constitue donc un refus
plutôt qu'une invitation.

## Tâches bloquées

Ces tâches sont bloquées pour toute famille sans entrée explicite.

| Tâche | Raison |
|---|---|
| `ocr` | Deux réseaux avec recadrage dynamique par région ne correspondent pas au contrat d'export en graphe unique |
| `point` | La famille n'est pas reliée au contrat partagé de heatmap de points et de décodage des pics par le backend |
| `semantic` | La famille n'est pas reliée au contrat partagé des logits denses et de l'argmax dans le backend |
| `mesh` | Les sorties du graphe de maillage corporel, les métadonnées et le contrat de runtime ne sont pas définis |
| `normal` | La famille n'est pas reliée au contrat partagé de normales unitaires denses sur canevas fixe et de renormalisation dans le backend |
| `panoptic` | L'export panoptique ne possède aucun contrat de runtime dans le backend |
| `gaze` | La famille n'est pas reliée au contrat partagé de logits à deux têtes et de décodage des espérances dans le backend |

Une entrée explicite remplace ces règles. C'est ainsi qu'une famille sémantique
reliée au contrat peut tout de même être exportée.

## Familles bloquées

| Famille | Blocage |
|---|---|
| `depth_anything3` | Tous les formats\u00a0; son graphe de profondeur ne fait pas partie du contrat de runtime exporté |
| `domedetr` | Tous les formats. PAQI définit le nombre de requêtes par image, un graphe tracé n'est donc valide que pour l'image du traçage. Utilisez D-FINE comme DETR exportable |
| `eomt` | Export d'instances et panoptique, qui ne possède aucun parsing dans le runtime |
| `l2cs` | Tout format autre que ONNX, TorchScript, ExecuTorch, TensorRT et OpenVINO |
| `hrnet` | Tout format autre que ONNX, TorchScript, OpenVINO et TensorRT |
| `sam`, `sam2`, `sam3`, `edgetam`, `mobilesam` | Tous les formats\u00a0; l'export de modèles guidables est hors périmètre du contrat de runtime v1 |
| `grounding_dino`, `owlv2`, `omdet_turbo`, `ov_deim` | Tous les formats\u00a0; l'export du runtime à vocabulaire ouvert est hors périmètre de la v1 |
| `florence2`, `kosmos2`, `lfm2vl`, `internvl3`, `qwen3vl`, `smolvlm2`, `locateanything` | Tous les formats\u00a0; l'export de VLM génératifs est hors périmètre de la v1 |

PicoSAM3 est l'exception du niveau guidable\u00a0: il exporte son réseau ROI brut de
96 pixels vers ONNX.

## Blocages pour NCNN

Les décodeurs de style DETR nécessitent des opérations d'échantillonnage que
NCNN n'implémente pas. Les familles suivantes sont donc bloquées pour `ncnn`,
sauf si une entrée explicite indique le contraire\u00a0: Deformable DETR, DETR,
DINO-DETR, D-FINE, LW-DETR, DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4,
RF-DETR et EC. Le message de refus cite ONNX, OpenVINO, TorchScript et TensorRT
comme solutions de remplacement.

## Seuils de parité

Une cellule validated signifie que l'artefact exporté a reproduit le modèle
natif dans les limites suivantes\u00a0:

| Groupe de tâches | Seuil |
|---|---|
| Détection et OBB | IoU des bounding boxes appariées supérieure à 0.95, MAE des scores inférieure à 0.01 |
| Segmentation et panoptique | IoU des masques supérieure à 0.95 |
| Pose | L2 des points clés inférieure à 2 pixels à la résolution native |
| Classification | Cosinus des logits supérieur à 0.999 et même classe top-1 |
| Profondeur et restauration | PSNR supérieur à 40\u00a0dB par rapport à la sortie native |
| Normales de surface | Erreur angulaire moyenne inférieure à 0.1 degré |
| Point | Emplacements des pics égaux à une cellule de sortie près |

Les lignes de requêtes DETR forment un ensemble non ordonné. La parité des
familles DETR les aligne donc comme un ensemble et non selon leur position.

## Exporter

<code-tabs name="export" />

Une combinaison blocked lève `NotImplementedError` lors des vérifications
préalables et le message contient la raison consignée.
`validated_alternatives(family, task)` renvoie les formats validated pour cette
paire. C'est l'information utile à afficher à côté d'un refus.

Les arguments partagés par tous les exporteurs sont énumérés sur la page de
l'[API du modèle](/docs/reference/model-api). Les arguments propres à chaque
format figurent sur les pages individuelles des formats.

## Lire une contrainte

Une cellule validated constitue une affirmation sur une configuration mesurée,
et non sur le format en général. Une chaîne de contrainte telle que
`FP32, batch 1, fixed 520x520 input` signifie que la parité a été consignée
pour cette forme et cette précision. Un export à une autre résolution ou taille
de batch produit toujours un artefact, mais cette configuration n'est pas celle
ayant produit la mesure.
