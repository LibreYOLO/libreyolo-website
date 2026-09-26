---
title: OV-DEIM
families:
  - ov_deim
seo_title: 'OV-DEIM dans LibreYOLO : détection à vocabulaire ouvert'
description: >-
  Utilisez OV-DEIM dans LibreYOLO pour la détection à vocabulaire ouvert en
  temps réel de style DETR. Installez l'extra openvocab et prédisez avec un
  vocabulaire en texte libre.
lead: >-
  OV-DEIM est un détecteur d'objets à vocabulaire ouvert de style DETR qui
  associe les requêtes du décodeur aux embeddings textuels d'une tour textuelle
  MobileCLIP incluse. LibreYOLO le porte nativement comme famille réservée à la
  prédiction dans sa catégorie de détecteurs à vocabulaire ouvert.
keywords:
  - OV-DEIM
  - DEIMv2
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

        model = LibreOpenVocab("ov-deim-s")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Remplacer le vocabulaire
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-l")
        model.set_classes(["traffic light", "bicycle"])
        first = model.predict(SAMPLE_IMAGE, conf=0.3)

        # Un second appel à set_classes() remplace tout le vocabulaire et le
        # réencode avec la tour textuelle ; un résultat vide constitue une
        # sortie valide et non une erreur.
        model.set_classes(["giraffe"])
        second = model.predict(SAMPLE_IMAGE, conf=0.5)
        print(second.names, len(second))
source_hash: 0c295f555a9eb303
---

## Installer

OV-DEIM se charge par la catégorie de détecteurs à vocabulaire ouvert de
LibreYOLO, qui nécessite l'extra `openvocab` :

```bash
pip install "libreyolo[openvocab]"
```

Contrairement au reste de cette catégorie, OV-DEIM est un portage LibreYOLO
natif et non un wrapper `transformers`, car il n'existe aucune classe de modèle
`transformers` pour celui-ci. Le même extra fournit cependant les paquets
`huggingface_hub`, `safetensors`, `regex` et `ftfy` nécessaires lors de la
prédiction.

## Prédire

OV-DEIM n'est pas un checkpoint chargé par LibreYOLO au moyen de `LibreYOLO()`.
Il se charge par la factory sœur `LibreOpenVocab`, qui télécharge un snapshot
Hugging Face lors de la première utilisation et le met en cache sous `weights/`.

<code-tabs name="predict" />

`set_classes()` définit un vocabulaire textuel persistant : rappelez la méthode
pour remplacer toute la liste, ou omettez-la pour conserver les étiquettes
COCO-80 par défaut ; un résultat vide constitue une sortie valide et non une
erreur. Chaque requête du décodeur reçoit un score de similarité cosinus par
rapport aux embeddings textuels d'une tour textuelle MobileCLIP-B(LT) incluse.
Ceux-ci sont calculés en ligne pour le vocabulaire défini et mis en cache jusqu'à
ce qu'il change, si bien que les requêtes arbitraires fonctionnent sans fichier
d'embeddings précalculés.

OV-DEIM ne possède aucun seuil de token textuel : seul `conf` filtre les
détections, et fournir `text_threshold` lève une erreur. La correspondance est
une sélection top-K un-à-un. Aucune suppression non maximale ne s'exécute donc
ici, et `iou` est accepté pour la compatibilité de l'API, mais produit un
avertissement et ne fait rien. `imgsz` et `augment=True` sont refusés sans
condition : le modèle contrôle une entrée fixe avec letterboxing, et
l'augmentation au moment du test dépasse le périmètre de cette catégorie.
`predict()` sur une seule image renvoie un objet `Results`, pas une liste ;
fournissez un dossier, une liste d'images ou `stream=True` pour une source vidéo
afin d'en obtenir plusieurs. Cette famille n'est pas accessible par la CLI :
`libreyolo predict` charge uniquement les checkpoints `.pt` avec `LibreYOLO()`,
les familles `LibreOpenVocab` s'exécutent donc depuis Python. Consultez la
[prédiction](/docs/predict) pour les types de sources et le streaming.

Chaque appel à `predict()` exécute également la tour textuelle MobileCLIP-B(LT)
incluse pour produire les embeddings du vocabulaire actuel ; consultez la
section Licence pour connaître les conditions que cela ajoute.

## Variantes

Trois checkpoints sont proposés, `s`, `m` et `l`. `s` est la taille par défaut
de cette catégorie lorsqu'aucune n'est indiquée. Contrairement au reste de cette
catégorie, OV-DEIM est un portage natif et non un wrapper `transformers` :
LibreYOLO fournit les modules du détecteur sous la même licence Apache-2.0 que le
code amont et réutilise l'adaptateur de backbone DINOv3 déjà construit pour la
famille DEIMv2. Le backbone du checkpoint `l` est un fine-tuning de DINOv3-S,
concédé séparément sous la DINOv3 License de Meta. Aucune mesure d'exactitude ou
de latence n'est encore publiée pour cette famille.

L'entraînement, la validation de datasets et l'export dépassent tous le
périmètre de cette catégorie : `train()`, `val()` et `export()` lèvent toujours
`NotImplementedError`. Il s'agit d'un wrapper de prédiction autour d'un
checkpoint publié.

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box>

OV-DEIM superpose trois licences amont à chaque appel de prédiction : les poids
du détecteur sous sa propre licence CC BY-NC 4.0, la tour textuelle en ligne sous
la Machine Learning Research Model License d'Apple (usage réservé à la
recherche) et, pour le checkpoint `l`, un fine-tuning du backbone DINOv3-S sous
la DINOv3 License de Meta. Les textes des trois licences sont fournis dans le
dépôt de poids LibreYOLO.

</provenance-box>

## Citation

<citation-block />
