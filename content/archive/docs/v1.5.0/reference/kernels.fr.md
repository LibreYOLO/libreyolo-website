---
title: Kernels
seo_title: Registre de kernels LibreYOLO et kernels du Hub
description: "Méthode de sélection des implémentations accélérées par LibreYOLO\_: registre sous libreyolo/kernels, kernel MS-deform-attn facultatif de Hugging Face Hub et réglage de l'attention fusionnée."
lead: >-
  Chaque opération accélérée de LibreYOLO possède une implémentation portable
  par défaut et, parfois, une variante plus rapide enregistrée par-dessus. La
  sélection s'effectue à l'exécution au moyen d'un prédicat. Une dépendance
  facultative manquante provoque un repli plutôt qu'une erreur, et un graphe
  exporté utilise toujours le chemin portable.
keywords:
  - kernels libreyolo
  - LIBREYOLO_KERNELS
  - LIBREYOLO_HUB_KERNELS
  - extra hub-kernels
  - kernel ms_deform_attn
  - set_fused_attention
  - kernels triton libreyolo
last_verified: 1.5.0
verification: >-
  API du registre lue dans libreyolo/kernels/__init__.py en v1.5.0, API
  d'attention lue dans libreyolo/kernels/attention/__init__.py et sdpa.py,
  fournisseur du Hub lu dans libreyolo/kernels/attention/ms_deform_attn.py, y
  compris sa révision épinglée et son prédicat d'éligibilité. Arborescence des
  répertoires relevée dans libreyolo/kernels/. Définition de l'extra lue dans
  pyproject.toml. Notes de comportement et mesures de benchmark lues dans
  docs/kernels.md. Historique du contrôle en v1.4.0 lu dans le commit de câblage
  de l'emplacement RF-DETR et l'entrée 1.5.0 du CHANGELOG.
meta:
  - label: Package
    value: libreyolo.kernels
    mono: true
  - label: Extra d'activation
    value: 'libreyolo[hub-kernels]'
    mono: true
  - label: Imposer la référence
    value: LIBREYOLO_KERNELS=off
    mono: true
snippets:
  usage:
    - label: Voir la sélection active
      language: python
      code: >
        import libreyolo.kernels as kernels


        # Association de l'emplacement d'opération au nom de l'implémentation
        choisie, ou "unavailable".

        print(kernels.active())
    - label: Imposer le chemin de référence
      language: bash
      code: |
        # off et reference ont le même sens et empêchent également
        # tout import des fournisseurs accélérés.
        LIBREYOLO_KERNELS=off python train.py
    - label: Désactiver les kernels du Hub sans désinstaller
      language: bash
      code: |
        LIBREYOLO_HUB_KERNELS=0 python predict.py
    - label: Passer une famille à l'attention fusionnée
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.kernels.attention import set_fused_attention

        model = LibreYOLO("LibreSwinIRs.pt")

        # Renvoie le nombre de modules d'attention modifiés.
        print(set_fused_attention(model))
    - label: Enregistrer votre propre implémentation
      language: python
      code: |
        import libreyolo.kernels as kernels

        kernels.register(
            "fake_quant_fp8",
            my_impl,
            name="mybackend",
            predicate=my_check,
        )
source_hash: 23d504e88b7959f8
---

## Registre

`libreyolo/kernels/` est un petit registre d'exécution d'implémentations
interchangeables. Un emplacement d'opération porte un nom comme
`fake_quant_fp8` ou `ms_deform_attn`. Les appelants demandent un emplacement
au registre et reçoivent la première implémentation enregistrée dont le
prédicat réussit, la plus récente l'emportant. Si aucune ne convient,
l'implémentation de référence est utilisée.

Cette structure garantit qu'une dépendance facultative ne devient jamais une
exigence stricte. Une machine sans Triton, sans CUDA ou sans le package
`kernels` exécute le même code et produit les mêmes valeurs, plus lentement.

| Fonction | Rôle |
|---|---|
| `active()` | Associe l'emplacement d'opération au nom de l'implémentation sélectionnée, ou `"unavailable"` |
| `resolve(op)` | Callable qui serait exécuté, ou `None` |
| `register(op, impl, *, name, predicate=None)` | Ajoute une implémentation, les plus récentes en premier |
| `unregister(op, name)` | En retire une |
| `clear_cache()` | Supprime la résolution mémorisée |

<code-tabs name="usage" />

Un prédicat qui lève une erreur est intercepté et produit un avertissement,
sans jamais propager l'erreur. Une implémentation tierce défectueuse provoque
donc un repli vers le chemin portable au lieu de casser la prédiction.

### Organisation

L'arborescence est organisée d'abord par usage, puis par backend. Un
emplacement est ainsi trouvé selon son calcul plutôt que selon la bibliothèque
qui l'implémente aujourd'hui.

| Répertoire | Contenu |
|---|---|
| `kernels/quant/simulate/` | Kernels Triton de quantification simulée, avec passe backward straight-through, sur tout appareil. Utilisés par QAT et par la quantification simulée post-entraînement |
| `kernels/quant/execute/` | Chemins de précision réelle réservés aux modèles finalisés, sans backward\u00a0: GEMM FP8 sur tensor cores, prologue et épilogue Triton fusionnés, et kernels de décompactage des poids compactés |
| `kernels/attention/` | Opérations d'attention partagées entre les familles\u00a0: emplacement `ms_deform_attn` et politique SDPA fusionnée |

La frontière entre `simulate` et `execute` dépend de la finalisation du modèle,
et non de son entraînement ou de son déploiement. Les implémentations de
référence restent dans `libreyolo/quant/`, qui définit la signification des
valeurs\u00a0; `kernels/` ne fait que les accélérer. La compaction des poids ne
possède aucune variante, car elle fait partie du contrat de checkpoint.

Les emplacements GEMM et attention ne possèdent aucune implémentation de
référence. Un appelant doit vérifier que `resolve()` a renvoyé une valeur et
conserver son propre chemin portable. C'est pourquoi les graphes ONNX,
TensorRT et `torch.export` contiennent toujours les calculs portables.

### Remplacements de sélection

`LIBREYOLO_KERNELS=off` ou `=reference` impose les implémentations de référence
et court-circuite entièrement l'import des fournisseurs accélérés. Toute autre
valeur limite la sélection aux implémentations enregistrées sous ce nom.
`LIBREYOLO_QUANT_KERNELS` reste pris en compte comme ancien alias datant de
l'époque où le registre se trouvait sous `libreyolo/quant/`. Il n'est lu que
si `LIBREYOLO_KERNELS` n'est pas défini. Les deux figurent avec les autres
variables dans les [paramètres](/docs/reference/settings).

## Kernels du Hub

Les kernels CUDA compilés publiés sur Hugging Face Hub sont chargés à
l'exécution au moyen du package facultatif `kernels`. Rien n'est intégré à
LibreYOLO\u00a0: l'artefact est récupéré et mis en cache par ce package, et chaque
fournisseur épingle une révision de commit auditée. La modification d'une
révision épinglée exige donc une exécution de parité sur GPU avant intégration.

L'installation de l'extra active la fonctionnalité\u00a0:

```bash
pip install "libreyolo[hub-kernels]"
```

Sans le package, rien ne change et aucune requête réseau n'est effectuée.
`LIBREYOLO_HUB_KERNELS=0` désactive la récupération sans rien désinstaller. Un
kernel dont le chargement ou l'exécution échoue se désactive pour le reste du
processus et se rabat sur le chemin portable avec un avertissement unique.

Un emplacement est aujourd'hui fourni par le Hub\u00a0: `ms_deform_attn`, les passes
forward et backward compilées de l'attention déformable multi-échelle de
Deformable DETR, sous licence Apache 2.0. Il est relié à toute la lignée
déformable\u00a0: RF-DETR, Deformable DETR, DINO-DETR, LW-DETR, Grounding DINO,
RT-DETR, RT-DETRv2, D-FINE, RT-DETRv4, DEIM, DEIMv2, EC et OV-DEIM. La passe
backward étant également compilée, l'entraînement en profite autant que la
prédiction.

L'éligibilité est délibérément stricte. Les entrées doivent être CUDA et
float32, et l'exécution doit être eager\u00a0: le fournisseur se désiste sous
`torch.jit.is_tracing()`, `torch.compiler.is_compiling()`,
`torch.compiler.is_exporting()` et `torch.onnx.is_in_onnx_export()`. Deux
structures d'entrée se rabattent aussi sur le chemin portable\u00a0: un nombre de
points par niveau variable entre les niveaux et l'échantillonnage discret par
indices entiers. La variante de pose EC n'est pas reliée.

### Ce kernel vient de devenir accessible

Lisez cette section avant d'installer l'extra sur un projet existant.

Dans la v1.4.0, l'emplacement était consulté depuis un assistant, derrière une
condition exigeant l'absence des paires de formes spatiales. RF-DETR transmet
toujours ces paires à son décodeur. La condition n'était donc jamais satisfaite
et le kernel ne s'exécutait dans aucune passe forward eager. La consultation a
été déplacée dans la v1.5.0 et le kernel s'exécute désormais réellement.

En pratique, passer à la v1.5.0 tout en installant
`libreyolo[hub-kernels]` sur CUDA amène RF-DETR et sa lignée à utiliser pour la
première fois un binaire compilé pour leur passe forward. Les prédictions et
les métriques peuvent donc varier dans la tolérance des flottants. Une
installation standard sans l'extra n'est pas affectée. Pour comparer les
métriques avant et après la mise à niveau, conservez le même état de l'extra ou
définissez `LIBREYOLO_HUB_KERNELS=0` des deux côtés.

## Attention fusionnée

L'attention par produit scalaire mis à l'échelle fusionnée ne nécessite aucune
dépendance facultative, uniquement PyTorch standard. Elle est donc régie par
une politique et non par sa disponibilité. Deux règles s'appliquent.

Premièrement, une capture de graphe ne l'utilise jamais. Chaque site d'appel
remplacé conserve l'équation à opérations primitives derrière un contrôle
d'export. Cela couvre l'export ONNX, dont l'opset par défaut ne possède aucun
symbolique SDPA, et `torch.jit.trace`, utilisé par TorchScript, CoreML et NCNN.
Les captures Dynamo sont délibérément hors du contrôle, car `torch.compile`
abaisse mieux SDPA que les calculs manuels, tandis que Core AI et ExecuTorch
décomposent eux-mêmes SDPA vers le noyau ATen.

Deuxièmement, la condition de parité pour l'activer par défaut exige une égalité
octet par octet. Les familles qui la satisfont utilisent SDPA par défaut\u00a0:
SegFormer, Depth Anything et MoGe-2, BERT, Grounding DINO, SwinIR et PP-OCR.
Les autres conservent les calculs manuels et exposent un paramètre `fused_attn`,
que `set_fused_attention(model)` modifie\u00a0: Swin, le backbone Swin de DINO-DETR,
BiRefNet et FeyNobg, OWLv2, LW-DETR, SigLIP 2, ZipDepth et MobileSAM. ViT et
DeiT portent le même paramètre mais l'activent par défaut, comme l'upstream.
Le même appel avec `enabled=False` les désactive donc.

L'opération est utile lorsqu'elle s'applique. Sur une RTX 5070 Ti avec
autocast fp16, l'attention par fenêtres Swin passe de 1.278\u00a0ms à 0.721\u00a0ms,
soit un gain de 1.77x, et l'attention visuelle OWLv2 de 6.483\u00a0ms à 1.735\u00a0ms,
soit 3.74x.

## Matériel

| Plateforme | Comportement |
|---|---|
| CPU et MPS | Tous les prédicats CUDA et Triton échouent, tout s'exécute donc avec les références |
| NVIDIA CUDA | Les kernels Triton et les kernels du Hub et GEMM éligibles s'activent |
| AMD ROCm | Triton peut s'activer puisque les wheels ROCm fournissent son backend AMD, mais la parité n'est exercée que sur NVIDIA dans la CI |

## Ajouter une implémentation

Appelez `register()` avec un nom et un prédicat. Des kernels compilés hors de
l'arborescence peuvent être fournis dans un package `libreyolo_kernels`
distinct qui s'enregistre lors de l'import. Un backend privé reste ainsi
entièrement hors de l'arborescence LibreYOLO.

La parité conditionne toute intégration dans l'arborescence\u00a0: correspondance
forward exacte avec la référence et gradients à moins de 1e-6 de l'estimateur
straight-through, sur l'ensemble de formes couvert par la suite de tests.

La sélection des kernels interagit avec les
[graphes CUDA](/docs/reference/cuda-graphs)\u00a0: la matrice de parité de
l'inférence a été exécutée sans le package `kernels`, elle ne couvre donc pas
la sécurité de capture lorsqu'un kernel compilé est actif.
