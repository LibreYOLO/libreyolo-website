---
title: RKNN
seo_title: Exporter vers RKNN pour les NPU Rockchip
description: "Compilez un détecteur LibreYOLO en artefact Rockchip .rknn\_: le SDK fabricant que vous installez vous-même, les quatre variantes RK3588 validées et la parité en simulateur."
lead: >-
  RKNN est le format NPU compilé de Rockchip. LibreYOLO exporte un ONNX
  intermédiaire en opset 19, le compile avec le SDK RKNN Toolkit2, et sait
  comparer le graphe compilé à ONNX Runtime dans le simulateur hôte de Toolkit2,
  sans carte.
keywords:
  - exporter yolo rknn
  - npu rockchip
  - rk3588
  - rknn-toolkit2
  - parité simulateur rknn
  - inférence rockchip orange pi
last_verified: 1.5.0
meta:
  - label: Flag
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: Écrit
    value: >-
      Un fichier .rknn, un sidecar .rknn.metadata.json et un rapport
      .rknn.parity.json quand verify=True
  - label: Extra
    value: >-
      Aucun sur PyPI. rknn-toolkit2 est un SDK fabricant que vous installez
      vous-même.
  - label: Rechargement
    value: >-
      Pas via LibreYOLO. L'artefact s'exécute sur la carte avec le runtime de
      Rockchip.
  - label: Formes
    value: 'Carré fixe, batch 1, opset 19. Les trois sont imposés.'
  - label: Précision
    value: Le build flottant du fabricant. half=True et int8=True sont refusés.
  - label: Portée
    value: "Quatre variantes de détection sur RK3588\_: YOLO9-t, YOLO9-E2E-t, PicoDet-s et YOLO-NAS-s"
verification: >-
  Lu depuis libreyolo/export/rknn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py et docs/rknn.md sur la branche dev. Les chiffres
  de parité mesurés proviennent du relevé de validation daté du 2026-08-04 dans
  docs/rknn.md.
snippets:
  install:
    - label: Côté LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'SDK fabricant, installé par vos soins'
      language: bash
      code: |
        # rknn-toolkit2 est un SDK Rockchip sous licence distincte. LibreYOLO
        # ne l'embarque ni ne l'installe. Linux x86_64 uniquement. Sous
        # Windows, utilisez WSL2 ou un conteneur Linux.
        #
        # Toolkit2 2.3.2 exige setuptools<81 et échoue sur ONNX 1.19 ou plus
        # récent, dont son compilateur importe encore onnx.mapping, supprimé.
        pip install "setuptools==80.9.0" "onnx==1.18.0"

        # Installez ensuite la wheel rknn-toolkit2 correspondante depuis le
        # dépôt de wheels de Rockchip, puis confirmez qu'elle s'importe.
        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Écrit weights/LibreYOLO9t.rknn et
        weights/LibreYOLO9t.rknn.metadata.json

        path = model.export(format="rknn", name="rk3588", imgsz=640,
        verify=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format rknn --name rk3588 \
          --imgsz 640 --verify
    - label: Arguments
      language: python
      code: |
        model.export(
            format="rknn",
            name="rk3588",     # plateforme cible, target= et target_platform= acceptés
            imgsz=640,         # doit correspondre au canevas enregistré de la variante
            batch=1,           # toute autre valeur lève NotImplementedError
            dynamic=False,     # True lève ValueError
            opset=19,          # toute autre valeur lève NotImplementedError
            verify=False,      # True lance le simulateur PC et impose la parité
        )
  parity:
    - label: Parité sans carte contre un artefact ONNX existant
      language: python
      code: |
        import numpy as np
        from libreyolo.export import verify_rknn_simulator_parity

        input_tensor = np.random.default_rng(0).standard_normal(
            (1, 3, 640, 640), dtype=np.float32
        )
        metrics = verify_rknn_simulator_parity(
            "weights/LibreYOLO9t.onnx",
            input_tensor,
            target_platform="rk3588",
            rtol=1e-3,
            atol=1e-4,
            raise_on_failure=False,
        )
        print(metrics)
  support:
    - label: Vérifier une famille et une tâche avant de compiler
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c659713cc3c8cc9e
---

## Installation

La compilation exige le RKNN Toolkit2 de Rockchip, distribué comme SDK fabricant
sous la licence propre à Rockchip et qui n'est pas une dépendance de LibreYOLO.
Il n'existe pas d'extra `libreyolo[rknn]`, et rien dans ce format ne s'installe
en une seule ligne.

<code-tabs name="install" />

Aucune carte n'est nécessaire pour compiler ni pour vérifier la parité numérique.
Une carte RK3588 est nécessaire pour les mesures de latence, de consommation et
de température, dont aucune n'a été relevée.

## Export

<code-tabs name="export" />

La demande est validée contre une liste de variantes de modèle exactes avant
toute compilation, et le canevas est validé lui aussi : passer un `imgsz` autre
que celui auquel la variante a été enregistrée lève une erreur plutôt que de
compiler en silence quelque chose qui n'a pas été testé. LibreYOLO écrit un ONNX
intermédiaire en opset 19, le compile, le simule éventuellement, puis supprime
l'intermédiaire.

Les métadonnées prennent la forme d'un sidecar nommé
`<model>.rknn.metadata.json`, car le format RKNN n'a pas de champ de métadonnées
portable.

`verify=True` lance le simulateur PC de Toolkit2 dans la session même qui a
compilé l'artefact, compare chaque sortie à ONNX Runtime sur la même entrée, et
écrit `<model>.rknn.parity.json` avec les métriques d'erreur par sortie. Les
seuils sont une similarité cosinus d'au moins 0.9999 et un RMSE normalisé d'au
plus 0.02, appliqués à toute sortie qui n'est pas déjà proche élément par
élément ; le build flottant du fabricant abaisse les tenseurs internes en
demi-précision, si bien qu'un `allclose` strict ne tient pas, même lorsque les
boîtes décodées sont stables. Une exécution en échec écrit
`<model>.rknn.failed.parity.json`, jette le candidat, et laisse intact un export
antérieur réussi situé à ce chemin.

Pour comparer un artefact ONNX que vous possédez déjà, sans réexporter :

<code-tabs name="parity" />

Le simulateur de Toolkit2 exécute le graphe en mémoire produit par `load_onnx` et
`build`. Il ne peut pas recharger un fichier `.rknn` propre à une cible sans
carte, et c'est pourquoi `verify=True` fait la compilation, l'export et la
simulation en une seule session.

## Exécuter l'artefact

Il n'y a pas d'entrée RKNN dans `libreyolo/backends`, donc `LibreYOLO()` ne
charge pas de fichier `.rknn`. L'artefact compilé est déployé sur la carte et
exécuté par le runtime propre à Rockchip, et le prétraitement, le décodage, le
NMS et la remise à l'échelle des coordonnées y relèvent de l'application.

`<model>.rknn.metadata.json` porte les noms de classe, la taille d'entrée, la
tâche et la plateforme cible, c'est-à-dire ce dont une application a besoin pour
reproduire le post-traitement de LibreYOLO. Livrez-le avec le modèle compilé.

Pour une vérification côté hôte qui ne demande pas la carte, conservez un
artefact ONNX à la même forme fixe et comparez-le dans le simulateur, comme
ci-dessus.

## Contraintes

Quatre combinaisons compilent, et ce sont des variantes de modèle plutôt que des
familles :

| Variante | Tâche | Canevas | Cible |
|---|---|---:|---|
| YOLO9-t | detect | 640 | RK3588 |
| YOLO9-E2E-t | detect | 640 | RK3588 |
| PicoDet-s | detect | 320 | RK3588 |
| YOLO-NAS-s | detect | 640 | RK3588 |

Tout le reste est refusé avant compilation, avec le message que RKNN, dans cette
version, se limite aux variantes de détection exactes testées en simulateur. Des
résultats de compilation seule existent pour d'autres modèles, mais ne sont
délibérément pas présentés comme un support : lors de la même campagne de mesure,
RF-DETR a laissé deux nœuds `GridSample` du décodeur non abaissés, et D-FINE,
RT-DETR, RT-DETRv2, RT-DETRv4, DEIM, DEIMv2 et EC ont compilé et simulé avec des
sorties décodées matériellement fausses.

Batch 1, formes statiques, opset 19. `half=True` est refusé, parce que RKNN
n'expose pas le contrat `half` de LibreYOLO, et `int8=True` est refusé tant qu'il
n'existe pas de calibration représentative ni de résultats d'exactitude par
tâche.

Les autres cibles Rockchip sont refusées : `rk3588` est la seule plateforme
validée.

Pour la grille complète des familles et des tâches, voir
[la matrice d'export](/docs/reference/export-matrix). Pour une combinaison :

<code-tabs name="support" />
