---
title: Maillage du corps
seo_title: Reconstruction du maillage corporel dans LibreYOLO
description: >-
  Reconstruire un maillage corporel 3D paramétrique par personne dans LibreYOLO.
  Prédire depuis des boîtes de personnes ou un détecteur, puis lire les sommets,
  les articulations et la translation de la caméra.
lead: >-
  La reconstruction du maillage corporel transforme une image unique et un
  ensemble de boîtes de personnes en un corps 3D paramétrique par personne :
  paramètres de forme et de pose, sommets articulés, articulations 3D et
  translation de la caméra qui les place devant l'objectif.
keywords:
  - reconstruction maillage humain python
  - maillage corporel
  - pose corps 3d
  - SAM 3D Body
  - MHR
  - modèle corporel paramétrique
  - tâche mesh libreyolo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Cette famille n'est pas enregistrée dans la fabrique LibreYOLO(). Elle

        # est donc construite directement. model_path=None déclenche le

        # téléchargement Hugging Face soumis à autorisation ; une chaîne est

        # interprétée comme un checkpoint local existant et n'est jamais
        téléchargée.

        # L'inférence nécessite CUDA.

        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        meshes = result.meshes

        print(meshes.body_model)      # paramétrage employé par ces tenseurs

        print(meshes.vertices.shape)  # (N, V, 3), repère caméra, mètres

        print(meshes.joints3d.shape)  # (N, J, 3)

        print(meshes.joints2d.shape)  # (N, J, 2), pixels de l'image source
    - label: Avec un détecteur de personnes
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # person_detector accepte un détecteur LibreYOLO construit, une simple

        # fonction ou une instance de PersonDetector. Il n'existe aucun
        raccourci de nom.

        detector = LibreYOLO("LibreYOLO9s.pt")

        model = LibreSAM3DBody(None, size="d3", device="cuda")


        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 31c5b44171cbcd0e
---

## Définition

La reconstruction du maillage corporel renvoie une charge utile `Meshes` par
image, dont les lignes sont alignées avec `result.boxes` : la ligne `i` décrit
la personne dans la boîte `i`, selon le même contrat que celui employé par la
tâche de pose pour les points clés.

Toutes les valeurs sont exprimées dans le repère de la caméra de l'image
d'origine. `transl` est une mesure en mètres, l'axe +z pointant à l'opposé de la
caméra. `vertices` et `joints3d` sont métriques et incluent déjà `transl`. Ils
ne nécessitent donc aucune composition supplémentaire. `joints2d` est exprimé
en pixels sur le canevas de l'image d'origine, pas sur le recadrage vu par le
réseau. `faces` contient une seule fois la topologie du maillage pour toute
l'image, et non par ligne, car toutes les personnes la partagent. Cette version
ne définit aucun repère du monde ou de gravité, et aucun champ ne s'y substitue
silencieusement.

La disposition des paramètres diffère entre les modèles corporels. Rien ne fixe
donc leurs formes : `body_model` nomme le paramétrage et les nombres sont relus
depuis les tenseurs. Pour `"mhr"`, Momentum Human Rig, les rotations sont des
angles d'Euler en radians plutôt que des représentations axe-angle,
`body_pose` est un vecteur plat de paramètres par articulation plutôt qu'un
triplet par articulation, et `betas` représente les coefficients des blendshapes
d'identité. L'échelle du squelette, la pose des mains et l'expression du visage
se trouvent dans `extras`.

La clé de tâche canonique est `mesh`. `body-mesh`, `hmr` et
`human-mesh-recovery` sont normalisés vers celle-ci.

## Modèles

[SAM 3D Body](/docs/models/sam-3d-body) est l'unique famille qui couvre cette
tâche. Il s'agit d'un wrapper plutôt que d'un portage : le paquet
`sam-3d-body` de Meta est publié sous la licence SAM, dont le propre code de
LibreYOLO ne peut pas dériver. Aucun de ses éléments n'est donc intégré. Deux
backbones partagent le même modèle corporel MHR, `d3` sur un encodeur
DINOv3 ViT-H/16+ et `h` sur le ViT-H d'origine.

Trois conditions s'appliquent avant une première prédiction, et aucune n'est
facultative.

Vous devez installer vous-même le paquet amont, et non par l'intermédiaire de
LibreYOLO :

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

Indiquez le clone à la bibliothèque au moyen de `sam_3d_body_path=` ou de la
variable d'environnement `SAM_3D_BODY_PATH`. Un utilisateur qui ne construit
jamais cette famille ne déclenche jamais l'importation.

L'accès au miroir du checkpoint est soumis à autorisation. Acceptez la licence
sur la page du modèle Hugging Face et authentifiez-vous avec `hf auth login`,
sinon le premier téléchargement échoue. Le modèle corporel MHR est une version
distincte sous licence Apache-2.0, récupérée depuis son propre emplacement
public et mise en cache localement.

L'inférence nécessite un périphérique CUDA. L'estimateur amont déplace son lot
vers le GPU sans effectuer de contrôle. Il n'existe donc aucun parcours CPU de
repli et `device="cpu"` déclenche une erreur.

## Prédire

<code-tabs name="predict" />

Les personnes atteignent le modèle de deux façons. `person_boxes` transmet des
boîtes que vous possédez déjà, uniquement pour une image. Un ensemble fixe de
boîtes ne peut pas suivre les personnes d'une vidéo. Le fournir avec une source
vidéo déclenche donc une erreur au lieu de réutiliser silencieusement les
boîtes de la première image. `person_detector` accepte un détecteur LibreYOLO
construit, une fonction ou un `PersonDetector`, et constitue le parcours à
utiliser pour la vidéo. `focal_length` fournit une intrinsèque de caméra connue.
Si elle est omise, le modèle emploie sa propre estimation, que
`meshes.focal_length` expose.

Cette famille n'est raccordée ni à la fabrique `LibreYOLO()` ni à la commande
CLI `libreyolo predict`. `LibreSAM3DBody` est son seul point d'entrée.
Consultez la page [prédiction](/docs/predict) pour les sources, le streaming et
la gestion des résultats.

## Entraîner

Aucune famille de cette tâche ne s'entraîne dans LibreYOLO.
`LibreSAM3DBody.train()` déclenche une erreur. Entraînez le modèle dans le
projet amont, puis chargez ici le checkpoint obtenu.

## Valider

Il n'existe aucun validateur de maillage et `val()` déclenche une erreur. Les
benchmarks habituels sont réservés à la recherche. Aucun n'est donc inclus ni
récupérable pour vous.

Les métriques sont toutefois disponibles sous
`libreyolo.validation.mesh_metrics` pour évaluer un dataset que vous possédez
déjà. La fonction reçoit les articulations prédites et cibles, éventuellement
les sommets prédits et cibles, puis renvoie un dictionnaire dont les clés
correspondent exactement à celles d'un validateur :

`metrics/mpjpe` est l'erreur moyenne de position par articulation après
alignement de l'articulation racine. Elle évalue donc la pose en ignorant la
position de la personne dans la scène. `metrics/pa_mpjpe` est la même grandeur
après un alignement de Procrustes complet, avec rotation, échelle uniforme et
translation. Il supprime l'orientation globale et l'erreur de taille du corps
pour ne conserver que la pose articulée. `metrics/pve` est l'erreur moyenne par
sommet sur la surface du maillage après alignement sur le centroïde des sommets.
Contrairement aux métriques des articulations, elle est sensible à la forme du
corps et n'apparaît que lorsque les deux tableaux de sommets sont fournis. Pour
les trois métriques, les valeurs faibles sont préférables. Les entrées sont
supposées métriques, en mètres, et `scale_to_mm` convertit les résultats dans
l'unité utilisée dans les publications, le millimètre.

## Exporter

L'exportation de maillage n'est pas implémentée. LibreYOLO n'a pas défini de
contrat de métadonnées pour le graphe exporté de cette tâche, notamment sur la
manière de transporter le paramétrage MHR hors de PyTorch. `export()` déclenche
donc une erreur au lieu de produire un graphe dont la sortie ne pourrait pas
être interprétée.
