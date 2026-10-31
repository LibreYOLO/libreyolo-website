---
title: OMDet-Turbo
families:
  - omdet_turbo
seo_title: 'OMDet-Turbo dans LibreYOLO : détection zero-shot en temps réel'
description: >-
  Utilisez OMDet-Turbo dans LibreYOLO pour la détection à vocabulaire ouvert en
  temps réel. Installez l'extra openvocab et prédisez avec un vocabulaire en
  texte libre.
lead: >-
  OMDet-Turbo est un détecteur d'objets à vocabulaire ouvert en temps réel,
  développé par Om AI Lab, qui dissocie les embeddings de classes d'une requête
  de tâche en langage naturel. LibreYOLO l'intègre comme famille réservée à la
  prédiction dans sa catégorie de détecteurs à vocabulaire ouvert.
keywords:
  - OMDet-Turbo
  - OmDet
  - détection d'objets à vocabulaire ouvert
  - détection temps réel
  - détection zero-shot
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.3)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Seuil NMS personnalisé
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("omdet-turbo")

        model.set_classes(["traffic light", "bicycle"])


        # OMDet-Turbo est la seule famille de cette catégorie qui respecte iou=
        :

        # son propre post-traitement reçoit le seuil de suppression en argument,

        # avec 0.5 par défaut lorsque iou= n'est pas défini.

        result = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.7)

        print(result.names, len(result))
source_hash: c2a375d234341b7e
---

## Installer

OMDet-Turbo se charge par la catégorie de détecteurs à vocabulaire ouvert de
LibreYOLO, qui nécessite l'extra `openvocab` :

```bash
pip install "libreyolo[openvocab]"
```

Cet extra installe `transformers` et `timm`, les bibliothèques Hugging Face
appelées par cette catégorie ; le backbone Swin d'OMDet-Turbo est chargé par le
wrapper `TimmBackbone` de `transformers`.

## Prédire

OMDet-Turbo n'est pas un checkpoint chargé par LibreYOLO au moyen de
`LibreYOLO()`. Il se charge par la factory sœur `LibreOpenVocab`, qui télécharge
un snapshot Hugging Face lors de la première utilisation et le met en cache sous
`weights/`.

<code-tabs name="predict" />

`set_classes()` définit un vocabulaire textuel persistant : rappelez la méthode
pour remplacer toute la liste, ou omettez-la pour conserver les étiquettes
COCO-80 par défaut ; un résultat vide constitue une sortie valide et non une
erreur. Contrairement à Grounding DINO, OMDet-Turbo dissocie ses embeddings de
classes de la requête de tâche en langage naturel. Le post-traitement de
`transformers` renvoie donc des étiquettes qui correspondent directement à la
liste de classes demandée, sans étape de désambiguïsation des expressions.

OMDet-Turbo ne possède aucun seuil de token textuel : seul `conf` filtre les
détections, et fournir `text_threshold` lève une erreur. C'est la seule famille
de cette catégorie qui exécute sa propre suppression non maximale dans
`post_process_grounded_object_detection`. `iou` est donc respecté ici au lieu
de produire un avertissement. `imgsz` et `augment=True` sont refusés sans
condition : le processeur `transformers` contrôle le redimensionnement et
l'augmentation au moment du test dépasse le périmètre de cette catégorie.
`predict()` sur une seule image renvoie un objet `Results`, pas une liste ;
fournissez un dossier, une liste d'images ou `stream=True` pour une source vidéo
afin d'en obtenir plusieurs. Cette famille n'est pas accessible par la CLI :
`libreyolo predict` charge uniquement les checkpoints `.pt` avec `LibreYOLO()`,
les familles `LibreOpenVocab` s'exécutent donc depuis Python. Consultez la
[prédiction](/docs/predict) pour les types de sources et le streaming.

## Variantes

Un seul checkpoint, `t`, constitue l'unique taille de cette catégorie. Il
reproduit `omlab/omdet-turbo-swin-tiny-hf` à une révision amont figée par
`OmDetTurboForObjectDetection` de `transformers` ; le fichier de poids mis en
miroir est identique octet par octet à ce snapshot amont. Aucune mesure
d'exactitude ou de latence n'est encore publiée pour cette famille.

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
