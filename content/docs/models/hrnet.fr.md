---
title: HRNet
families:
  - hrnet
seo_title: "HRNet\_: l'estimation de pose top-down dans LibreYOLO"
description: >-
  Utilisez HRNet dans LibreYOLO pour l'estimation de pose top-down COCO-17.
  Installez, prédisez, validez et exportez les checkpoints W32 et W48, sous
  licence MIT.
lead: >-
  HRNet est un réseau convolutif qui conserve un flux de caractéristiques en
  haute résolution grâce à des fusions multi-échelles répétées, au lieu de
  récupérer la résolution après sous-échantillonnage. LibreYOLO encapsule la
  variante officielle de pose top-down pour l'inférence et la validation.
keywords:
  - HRNet
  - estimation de pose humaine python
  - pose top-down
  - points clés COCO-17
  - détection de squelette humain
  - réseau haute résolution
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sans source de personnes : HRNet s'associe automatiquement à un
        # détecteur LibreYOLO9t léger et journalise ce choix une fois.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreHRNetw32-pose.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Source des personnes
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        # Sautez entièrement la détection : l'image entière est une personne.
        result = model(SAMPLE_IMAGE, cropped=True)

        # Ou passez à HRNet les boîtes d'un détecteur déjà exécuté.
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        # Ou associez-le à un détecteur LibreYOLO précis plutôt qu'au
        # LibreYOLO9t par défaut.
        result = model(SAMPLE_IMAGE, person_detector="rfdetr")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreHRNetw32-pose.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreHRNetw32-pose.pt format=onnx
    - label: Utiliser le fichier exporté
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # Le graphe exporté est la seule tête de heatmap à canevas fixe : il
        # prend un batch de crops de personnes déjà découpés et normalisés,
        # et renvoie des heatmaps brutes. Détection de personnes, géométrie du
        # crop, décodage des heatmaps et suppression OKS n'en font pas partie ;
        # l'exécuter hors de LibreYOLO impose de réimplémenter ce décodage.
        session = ort.InferenceSession("LibreHRNetw32-pose.onnx")
        name = session.get_inputs()[0].name
        heatmaps = session.run(
            None, {name: np.zeros((1, 3, 256, 192), dtype=np.float32)}
        )[0]
source_hash: 5a5540fd54ee6f23
---

## Installation

HRNet ne demande aucun extra au-delà du paquet de base.

```bash
pip install libreyolo
```

Son détecteur de personnes par défaut, un checkpoint LibreYOLO9t léger, se
télécharge automatiquement la première fois que HRNet s'y associe.

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

HRNet est un estimateur de pose top-down : il lui faut une boîte de personne
avant que la tête de pose puisse s'exécuter, si bien que chaque appel en résout
une. Laissé à lui-même, il s'associe à un détecteur LibreYOLO9t la première fois
et journalise ce choix. `cropped=True` saute la détection et traite l'image
entière comme une seule personne ; `person_boxes` accepte les boîtes d'un
détecteur que vous avez déjà lancé ; `person_detector` accepte `"auto"`,
`"rfdetr"`, n'importe quel modèle de détection LibreYOLO, ou un simple callable.
`flip_test=True` exécute aussi le modèle sur le crop retourné horizontalement et
moyenne les deux heatmaps, l'augmentation au moment du test propre à HRNet ; le
`augment=True` générique n'est pas défini ici. Les sources multi-images sont
traitées séquentiellement : le détecteur de HRNet et le nombre variable de
personnes par image ne permettent pas la prédiction empilée. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Variantes

Deux tailles, `w32` et `w48`, qui prédisent toutes deux le jeu standard de
points clés COCO-17 à partir d'un crop de personne à résolution fixe ; `w48` est
le plus large des deux backbones.

Le model zoo upstream publie l'exactitude de pose de chaque taille avec son
propre détecteur de personnes, son propre réglage de flip-testing et le
protocole d'évaluation COCO officiel. L'association par défaut de LibreYOLO
utilise un autre détecteur, si bien qu'une validation lancée ici mesure cette
combinaison, pas celle d'upstream ; retrouver les chiffres upstream demande les
mêmes boîtes de personnes, les mêmes scores de détecteur et le même réglage de
flip que ceux de l'évaluation d'origine.

## Valider

`val()` calcule l'OKS-AP de points clés à la façon de COCO et accepte un
`data.yaml` YOLO-pose ou un JSON de points clés COCO accompagné d'un dossier
d'images. Le backend de métriques est faster-coco-eval par défaut,
`pycocotools` prenant automatiquement le relais quand faster-coco-eval n'est pas
installé ; `faster_coco_eval=False` force le chemin `pycocotools`.

<code-tabs name="val" />

La validation pilote en interne le `predict()` de HRNet, elle utilise donc le
détecteur de personnes avec lequel le modèle a été construit ou appelé.
Construisez le modèle avec un `person_detector=` explicite pour garder cette
source fixe d'une exécution à l'autre, plutôt que de laisser chaque appel
re-résoudre la valeur par défaut.

## Exporter

<export-matrix />

Le contrat d'export de HRNet ne couvre qu'ONNX, TorchScript, OpenVINO et
TensorRT ; tout autre format lève une erreur avant le début du tracé. Chaque
export est la seule tête de heatmap à canevas fixe, en batch un et FP32, qui
prend un crop de personne et renvoie des heatmaps brutes : la géométrie affine
du crop en amont et le décodage des heatmaps, la restauration du flip et la
suppression OKS en aval restent en Python, si bien qu'un pipeline complet, image
en entrée et points clés en sortie, a toujours besoin de LibreYOLO à l'autre
bout.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
