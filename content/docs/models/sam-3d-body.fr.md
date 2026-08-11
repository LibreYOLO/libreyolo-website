---
title: SAM 3D Body
families:
  - sam3dbody
seo_title: "SAM 3D Body\_: reconstruction d'un maillage corporel complet dans LibreYOLO"
description: "Utilisez SAM 3D Body dans LibreYOLO pour reconstruire le maillage 3D d'un corps humain complet. Installez-le et lancez des prédictions\_; les checkpoints sont soumis à la SAM License de Meta et CUDA est requis."
lead: >-
  SAM 3D Body est le modèle guidable de Meta qui reconstruit, à partir d'une
  seule image et de bounding boxes de personnes, le maillage 3D d'un corps
  complet, mains et pieds compris. LibreYOLO encapsule le package upstream au
  lieu de le porter.
keywords:
  - SAM 3D Body
  - reconstruction maillage humain
  - maillage corporel 3D
  - MHR
  - Momentum Human Rig
  - pose 3D
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Cette famille n'est pas enregistrée dans la fabrique LibreYOLO(), elle

        # est donc construite directement. model_path=None déclenche le

        # téléchargement protégé depuis Hugging Face ; une chaîne est au
        contraire

        # traitée comme le chemin d'un checkpoint local, jamais récupéré
        automatiquement.

        # L'inférence nécessite un appareil CUDA ; aucun chemin CPU n'existe.

        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        meshes = result.meshes

        print(meshes.vertices.shape)    # (N, V, 3), repère caméra, mètres

        print(meshes.joints3d.shape)    # (N, J, 3)
    - label: Avec un détecteur de personnes
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Aucun raccourci par chaîne nommée ici : transmettez un détecteur

        # LibreYOLO construit, un callable simple ou une instance de
        PersonDetector.

        detector = LibreYOLO("LibreRFDETRn.pt")

        model = LibreSAM3DBody(None, size="d3", device="cuda")


        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 8edc8d7872f3f875
---

## Installer

```bash
pip install libreyolo
```

Cette commande installe uniquement l'adaptateur de LibreYOLO. SAM 3D Body
lui-même n'est pas inclus, car sa licence n'autorise pas le code propre à
LibreYOLO à en être dérivé\u00a0: clonez le dépôt upstream et installez vous-même
ses dépendances, puis indiquez à LibreYOLO le chemin du clone.

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

```python
from libreyolo.models.sam3dbody import LibreSAM3DBody

model = LibreSAM3DBody(
    None,
    size="d3",
    sam_3d_body_path="/path/to/sam-3d-body",
    device="cuda",
)
```

Vous pouvez aussi définir la variable d'environnement `SAM_3D_BODY_PATH` au
lieu de transmettre `sam_3d_body_path` à chaque appel. Un utilisateur qui ne
construit jamais cette famille ne déclenche jamais l'import et ne rencontre
jamais la SAM License. Cette famille n'est intégrée ni à la fabrique
`LibreYOLO()` ni à la commande CLI `libreyolo predict`\u00a0; `LibreSAM3DBody` est
le seul point d'entrée.

## Prédire

<code-tabs name="predict" />

Le téléchargement du checkpoint est protégé\u00a0: vous devez accepter la licence
de Meta sur la page du modèle Hugging Face et vous authentifier avec
`hf auth login` pour que le premier téléchargement aboutisse. L'inférence
elle-même exige toujours un appareil CUDA\u00a0: l'estimateur upstream déplace son
batch vers le GPU sans effectuer de vérification, si bien qu'une machine sans
GPU lève une erreur au lieu de se rabattre sur le CPU. `result.meshes` est une
charge utile `Meshes`, alignée ligne par ligne avec `result.boxes` (une ligne
par personne détectée)\u00a0: `vertices` et `joints3d` sont exprimés dans le système
métrique et incluent déjà la translation estimée de la caméra, `joints2d` est
exprimé en pixels sur l'image d'origine, et les rotations suivent la convention
de MHR, avec des angles d'Euler plutôt qu'un format axe-angle. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Variantes

Deux backbones utilisent le même modèle corporel MHR\u00a0: `d3` emploie un encodeur
DINOv3 ViT-H/16+, tandis que `h` emploie l'encodeur ViT-H d'origine.

## Export

<export-matrix />

L'export du maillage corporel n'est pas implémenté\u00a0: LibreYOLO n'a pas encore
défini de contrat de graphe exporté pour la tâche de maillage, notamment pour
représenter la disposition des paramètres MHR hors de PyTorch.

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box>

Le modèle corporel piloté par les checkpoints, MHR (Momentum Human Rig), est
une version Meta distincte sous licence Apache-2.0. À l'exécution, LibreYOLO
récupère son asset TorchScript depuis la version publique de MHR et le met en
cache localement\u00a0; ce fichier n'est pas hébergé par LibreYOLO et relève de ses
propres conditions Apache-2.0, pas de la SAM License.

</provenance-box>

## Citation

<citation-block />

