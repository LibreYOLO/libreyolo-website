---
title: OWLv2
families:
  - owlv2
seo_title: 'OWLv2 dans LibreYOLO : détection d''objets zero-shot'
description: >-
  Utilisez OWLv2 dans LibreYOLO pour détecter tout objet décrit par du texte.
  Installez l'extra openvocab et prédisez avec un vocabulaire en texte libre.
lead: >-
  OWLv2 est un détecteur d'objets à vocabulaire ouvert, développé par Google
  Research, qui évalue les régions d'une image par rapport aux embeddings
  textuels d'un encodeur de style CLIP. LibreYOLO l'intègre comme famille
  réservée à la prédiction dans sa catégorie de détecteurs à vocabulaire ouvert.
keywords:
  - OWLv2
  - OWL-ViT
  - détection d'objets à vocabulaire ouvert
  - détection zero-shot
  - détecteur conditionné par texte
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Vocabulaire par défaut
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        # Omettre set_classes() conserve le vocabulaire COCO-80 de la catégorie.
        model = LibreOpenVocab("owlv2-l14")
        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        print(result.names)
source_hash: 2d0ce68af0daabb7
---

## Installer

OWLv2 se charge par la catégorie de détecteurs à vocabulaire ouvert de
LibreYOLO, qui nécessite l'extra `openvocab` :

```bash
pip install "libreyolo[openvocab]"
```

Cet extra installe `transformers` et `timm`, les bibliothèques Hugging Face
appelées par cette catégorie.

## Prédire

OWLv2 n'est pas un checkpoint chargé par LibreYOLO au moyen de `LibreYOLO()`.
Il se charge par la factory sœur `LibreOpenVocab`, qui télécharge un snapshot
Hugging Face lors de la première utilisation et le met en cache sous `weights/`.

<code-tabs name="predict" />

`set_classes()` définit un vocabulaire textuel persistant : rappelez la méthode
pour remplacer la liste, ou omettez-la pour conserver les étiquettes COCO-80 par
défaut. Chaque étiquette est placée dans un template de requête fixe avant
d'atteindre la tour textuelle, conformément à la méthode d'entraînement de
`Owlv2ForObjectDetection` dans `transformers`.

OWLv2 ne possède aucun seuil de token textuel : seul `conf` filtre les
détections, et fournir `text_threshold` lève une erreur. `iou` est accepté pour
la compatibilité de l'API, mais produit un avertissement et ne fait rien,
puisqu'aucune suppression non maximale ne s'exécute ici. `imgsz` et
`augment=True` sont refusés sans condition : le processeur `transformers`
contrôle le redimensionnement et l'augmentation au moment du test dépasse le
périmètre de cette catégorie. `predict()` sur une seule image renvoie un objet
`Results`, pas une liste ; fournissez un dossier, une liste d'images ou
`stream=True` pour une source vidéo afin d'en obtenir plusieurs. Cette famille
n'est pas accessible par la CLI : `libreyolo predict` charge uniquement les
checkpoints `.pt` avec `LibreYOLO()`, les familles `LibreOpenVocab` s'exécutent
donc depuis Python. Consultez la [prédiction](/docs/predict) pour les types de
sources et le streaming.

## Variantes

Deux checkpoints sont proposés, `b16` (base, taille de patch 16) et `l14`
(large, taille de patch 14). `b16` est la taille par défaut de cette catégorie
lorsqu'aucune n'est indiquée. Tous deux reproduisent la version officielle de
Google Research au moyen de `Owlv2ForObjectDetection` de `transformers`. Ils
sont téléchargés une fois dans un snapshot Hugging Face hébergé par LibreYOLO
qui conserve les fichiers amont. Aucune mesure d'exactitude ou de latence n'est
encore publiée pour cette famille.

L'entraînement, la validation de datasets et l'export dépassent tous le
périmètre de cette catégorie : `train()`, `val()` et `export()` lèvent toujours
`NotImplementedError`. Il s'agit d'un wrapper de prédiction autour d'un
checkpoint publié.

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
