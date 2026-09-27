---
title: Quantification
seo_title: Quantifier un modèle LibreYOLO dans PyTorch
description: "L'API de quantification PyTorch de LibreYOLO\_: neuf recettes, des données de calibration tenues à l'écart des données d'entraînement, QAT et QAD, et deux artefacts de déploiement."
lead: "La quantification dans LibreYOLO s'exécute entièrement dans PyTorch\_: model.quantize() remplace les modules Conv2d et Linear d'un modèle par leurs équivalents quantifiés et les calibre. Le résultat conserve le contrat habituel predict, val, train et save, si bien qu'un modèle quantifié est évalué par les mêmes validateurs qu'un modèle float."
keywords:
  - quantification libreyolo
  - quantification int8 pytorch
  - ptq int8
  - quantization aware training qat
  - qat qad
  - nvfp4 mxfp4
  - dataset de calibration
  - export onnx qdq int8
last_verified: 1.5.0
meta:
  - label: Appel
    value: 'model.quantize(recipe="int8", calib="coco128.yaml")'
    mono: true
  - label: Commande
    value: libreyolo quantize --model M.pt --recipe int8 --calib coco128.yaml
    mono: true
  - label: Extra
    value: Aucun. La quantification s'exécute dans PyTorch.
  - label: Familles
    value: 'yolo9, rfdetr, birefnet, feynobg'
  - label: Recettes
    value: 'fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4, mxfp4, int2'
    mono: true
  - label: Artefacts de déploiement
    value: >-
      export(format="pt") pour un checkpoint compacté, export(format="onnx")
      pour un graphe QDQ INT8
    mono: true
verification: >-
  Lu depuis libreyolo/quant/api.py, libreyolo/models/base/model.py,
  libreyolo/cli/commands/quantize.py et docs/quantization.md sur la branche dev.
  Les chiffres de taille de checkpoint sont les valeurs mesurées consignées dans
  docs/quantization.md.
snippets:
  quantize:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Échange de structure puis calibration. calib est un petit jeu d'images

        # NON ÉTIQUETÉES, lu en forward seul pour déduire plages et scales.

        qmodel = model.quantize(recipe="int8", calib="coco128.yaml",
        samples=128)


        print(qmodel.quant_info())

        qmodel.val(data="coco8.yaml")          # mêmes validateurs qu'un modèle
        float

        qmodel.save("LibreYOLO9s-int8.pt")     # le checkpoint porte un
        manifeste quant
    - label: CLI
      language: bash
      code: >
        libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib
        coco128.yaml
    - label: Arguments
      language: python
      code: |
        model.quantize(
            recipe="int8",
            calib="coco128.yaml",      # chemin data.yaml ou nom intégré, None saute la calibration
            samples=128,               # nombre maximal d'images de calibration
            batch=8,                   # taille de batch de calibration
            algorithm="auto",          # auto et minmax sont identiques, percentile est l'alternative
            keep_high_precision=None,  # None applique la politique de la famille
            verbose=True,
        )
  reload:
    - label: Un checkpoint quantifié se recharge comme tel
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Le manifeste quant reconstruit la structure quantifiée et les scales
        # avant le chargement des poids.
        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        print(qmodel.quant_info())
  train:
    - label: Le QAT est un simple train() sur un modèle quantifié
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # Un fine-tuning, pas un run à partir de zéro. Utilisez des learning
        # rates de fine-tuning.
        qmodel.train(data="coco8.yaml", epochs=5, lr0=1e-4)
    - label: Le QAD ajoute les arguments de distillation existants
      language: python
      code: |
        qmodel.train(
            data="coco8.yaml",
            epochs=5,
            lr0=1e-4,
            distill_model="LibreYOLO9m.pt",
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train --model LibreYOLO9s-int8.pt --data coco8.yaml --epochs 5
        --lr0 1e-4
  export:
    - label: Checkpoint PyTorch compacté
      language: python
      code: >
        from libreyolo import LibreYOLO


        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")


        # Écrit LibreYOLO9s-int8-final.pt, poids et scales compactés basse

        # précision, maîtres fp32 supprimés, reste non quantifié converti en
        fp16.

        qmodel.export(format="pt")


        # remainder="fp32" conserve les tenseurs non quantifiés exacts.

        qmodel.export(format="pt", remainder="fp32")
    - label: QDQ INT8 ONNX
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # Paires QuantizeLinear/DequantizeLinear dans le graphe, portant les
        # scales calibrés ou entraînés en QAT du modèle lui-même.
        qmodel.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9s-int8.pt --format onnx
  dequantize:
    - label: 'Retour en float, en conservant les poids entraînés par QAT'
      language: python
      code: >
        from libreyolo import LibreYOLO


        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        qmodel.dequantize()


        # N'importe quel exporteur float s'applique alors, à toute précision
        gérée.

        qmodel.export(format="tensorrt", half=True)
source_hash: 4ffb06b87cad017e
---

## Installation

La quantification ne demande aucun extra. L'échange de modules, la passe de
calibration et l'arithmétique simulée s'exécutent tous dans PyTorch, donc
`pip install libreyolo` est toute l'exigence. Les artefacts de déploiement
demandent ce qu'exige leur propre format, soit `libreyolo[onnx]` pour la voie ONNX.

## Quantifier

<code-tabs name="quantize" />

`quantize()` transforme le modèle chargé sur place et le renvoie. Aucun gradient
n'intervient : l'échange installe des modules quantifiés et la passe de
calibration ne fait que du forward.

Le checkpoint obtenu est un checkpoint LibreYOLO ordinaire auquel est attaché un
manifeste `quant`, il se recharge donc avec sa structure et ses scales intacts :

<code-tabs name="reload" />

Les checkpoints écrits par l'entraîneur pendant un run QAT portent eux aussi le
manifeste, ce qui veut dire que le `best.pt` d'un tel run est lui-même un
checkpoint quantifié.

## Recettes

Quatre familles sont prises en charge : `yolo9`, `rfdetr`, `birefnet` et
`feynobg`.

| Recette | Ce qu'elle fait | Familles | Calibration |
|---|---|---|---|
| `fp16` | Conversion en demi-précision, avec un contrat d'entrée et de sortie en float32. Inférence seulement. | les quatre | aucune |
| `bf16` | Conversion en bfloat16, qui conserve la plage d'exposants du float32. Le remède quand fp16 déborde sur un modèle de type DETR. Inférence seulement. | les quatre | aucune |
| `fp8` | Poids et activations E4M3 sur `Conv2d` et `Linear` : scales de poids par canal, scales d'activation par tenseur calibrés. | les quatre | obligatoire |
| `int8` | W8A8 sur `Conv2d` et `Linear` : poids symétriques par canal, activations affines par tenseur. | les quatre | obligatoire, ou `calib=None` pour les poids seuls |
| `w4a16` | Poids INT4 symétriques groupés, groupe de 128 le long de `in_features`, activations float, sur `Linear`. | rfdetr, birefnet, feynobg | inutile |
| `w4a8` | Poids INT4 groupés plus activations INT8 calibrées, sur `Linear`. | rfdetr, birefnet, feynobg | obligatoire |
| `nvfp4` | NVFP4 W4A4 sur `Linear` : éléments E2M1, blocs de 16 éléments, scales de bloc FP8 E4M3, scale de tenseur FP32. Mise à l'échelle dynamique des activations. | rfdetr, birefnet, feynobg | inutile |
| `mxfp4` | MXFP4 OCP sur `Linear` : éléments E2M1, blocs de 32 éléments, scales de bloc E8M0 en puissances de deux. Mise à l'échelle dynamique des activations. | rfdetr, birefnet, feynobg | inutile |
| `int2` | Recherche seulement : poids 2 bits groupés, groupe de 64, plus activations INT8, sur `Linear`. Le post-training seul est inutilisable, donc QAT ou QAD est obligatoire. | rfdetr | obligatoire |

Les recettes en dessous de 8 bits visent `nn.Linear` et sont rejetées pour
`yolo9` à dessein : sur le matériel actuel, cette accélération ne concerne que
les GEMM, les convolutions restent donc dans une précision plus élevée. YOLO9
utilise `int8` ou `fp8`. `int2` est rejetée pour `birefnet` et `feynobg` parce
que ces familles sont réservées à l'inférence, donc le QAT réparateur dont dépend
la recette n'y est pas disponible.

Les valeurs par défaut propres à chaque famille gardent la première couche et les
têtes en float, et la convolution DFL de YOLO9 n'est jamais quantifiée : c'est un
opérateur d'espérance intégrale fixe. Surchargez avec
`keep_high_precision=("head.",)` quand vous avez une raison de le faire.

## Les données de calibration ne sont pas des données d'entraînement

`calib=` prend quelques centaines d'images, ne lit aucune étiquette et ne fait que
du forward pour estimer les plages d'activation. `data=` dans `train()` et `val()`
est le dataset étiqueté utilisé pour les gradients et les métriques. Ce sont deux
arguments différents aux finalités différentes, et la valeur par défaut de `calib`
est `coco128.yaml`.

`algorithm="minmax"` conserve les extrêmes absolus observés sur l'ensemble des
batches de calibration, et c'est ce que sélectionne `"auto"`. `"percentile"`
utilise la moyenne des percentiles 0.1 et 99.9 par batch ; les mesures montrent
qu'il fait s'effondrer l'exactitude de la famille DETR, parce que les valeurs
aberrantes des activations des transformers jouent un rôle porteur. Ce qui corrige
réellement la sensibilité INT8 des petits modèles, c'est de calibrer sur assez de
batches : avec la valeur par défaut `coco128`, YOLO9-t se situe à environ un point
de mAP de son score float. L'algorithme choisi est consigné dans le manifeste du
checkpoint.

## Récupérer de l'exactitude

<code-tabs name="train" />

Les modules quantifiés conservent des poids maîtres fp32 et appliquent une fausse
quantification avec un straight-through estimator, si bien que les gradients
atteignent les maîtres et que les entraîneurs existants fonctionnent sans
changement : EMA, AMP, reprise depuis un checkpoint et arguments de distillation
se composent tous.

Le QAT est un fine-tuning d'un modèle déjà entraîné. Utilisez des learning rates
de fine-tuning plutôt que les valeurs par défaut prévues pour un entraînement à
partir de zéro, sinon un run court détruira les poids pré-entraînés, quantification
ou pas. La disponibilité du QAD suit la prise en charge de la distillation par
famille, ce qui veut dire aujourd'hui `yolo9` et `rfdetr`.

Les modèles quantifiés en `fp16` et `bf16` sont réservés à l'inférence, et
l'entraîneur les rejette en renvoyant vers `amp=True`.

## Export

<code-tabs name="export" />

`format="pt"` cristallise le modèle. Des poids et des scales compactés en basse
précision remplacent les maîtres, et le reste non quantifié est converti en fp16
sauf si `remainder="fp32"` est passé. L'invariant du compactage est que le
décompactage reproduit la simulation bit pour bit sur l'appareil où vous avez
finalisé, donc le fichier finalisé obtient exactement le score que vous avez
validé. Mesuré : YOLO9-s int8 passe de 29.5 Mo à 9.6 Mo, RF-DETR-n nvfp4 de
122 Mo à 26 Mo. En charger un donne un modèle prêt pour l'inférence, et appeler
`train()` dessus reconstruit automatiquement les maîtres à partir des poids
compactés.

`format="onnx"` s'applique aux modèles `int8` et émet un graphe QDQ portant les
scales calibrés ou entraînés en QAT du modèle lui-même, qu'ONNX Runtime et
TensorRT exécutent avec de vrais kernels INT8. C'est une voie différente de
[`export(format="onnx", int8=True)`](/docs/export/onnx) sur un modèle float, où
ONNX Runtime dérive lui-même les scales.

Les recettes de conversion n'ont besoin d'aucun exporteur quantifié :

<code-tabs name="dequantize" />

## Contraintes

L'arithmétique quantifiée s'exécute en simulation, c'est-à-dire une fausse
quantification calculée dans des îlots float32, même sous AMP. La simulation est
fidèle numériquement, donc un score `val()` sur n'importe quel appareil est une
affirmation réelle sur l'arithmétique quantifiée. Ce n'est pas une affirmation de
vitesse.

Deux exceptions s'exécutent nativement. `fp16` et `bf16` sont de simples
conversions. Les modules `fp8` finalisés exécutent leur GEMM directement sur des
poids E4M3 compactés via `torch._scaled_mm`, sur du matériel de classe Ada, Hopper
et Blackwell, en utilisant les mêmes scales d'activation calibrés que la
simulation ; définir `LIBREYOLO_KERNELS=off` rétablit partout le chemin simulé
exact.

La couverture de déploiement est plus étroite que la liste des recettes. Seul
`int8` dispose ici d'une forme ONNX déployable ; `fp8` et les recettes linéaires
en dessous de 8 bits s'exécutent dans PyTorch et se cristallisent via
`format="pt"`. Leur demander un export ONNX lève une erreur qui donne cette
instruction, tout comme demander un format non-ONNX à un modèle `int8` :
construisez plutôt les moteurs en aval à partir du graphe QDQ.

Exporter un modèle `int8` dont les activations n'ont jamais été calibrées émet un
avertissement dans les logs et produit un graphe ne portant que la quantification
des poids.
