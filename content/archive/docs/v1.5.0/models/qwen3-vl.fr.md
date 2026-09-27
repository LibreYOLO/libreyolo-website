---
title: Qwen3-VL
families:
  - qwen3vl
seo_title: 'Qwen3-VL dans LibreYOLO : détection à vocabulaire ouvert'
description: >-
  Qwen3-VL dans LibreYOLO : installez, définissez un vocabulaire ouvert, puis
  prédisez ou discutez avec le modèle vision-langage Apache-2.0 d'Alibaba.
lead: >-
  Qwen3-VL est le modèle vision-langage d'Alibaba doté d'une localisation 2D
  native. LibreYOLO l'intègre comme détecteur d'objets à vocabulaire ouvert et
  expose directement son chat libre : fournissez une liste de classes à détecter
  ou posez-lui une question.
keywords:
  - Qwen3-VL
  - modèle vision-langage
  - détection à vocabulaire ouvert
  - localisation d'objets
  - Alibaba
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")
        model.set_classes(["forklift", "pallet", "safety vest"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: >
        from libreyolo import LibreVLM, SAMPLE_IMAGE


        model = LibreVLM("qwen3-vl-4b")


        # L'accès direct sous l'interface de détection : toute question,

        # pas uniquement une requête de bounding boxes.

        answer = model.chat(SAMPLE_IMAGE, "How many people are wearing a safety
        vest?")

        print(answer)
source_hash: ee225b6221d624d9
---

## Installer

Qwen3-VL appartient à la catégorie de VLM utilisés comme détecteurs de
LibreYOLO, une interface produit distincte des familles basées sur des
checkpoints et dotée de sa propre factory. Il nécessite l'extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement. Appelé sans argument, `LibreVLM()` utilise Qwen3-VL-4B par
défaut.

<code-tabs name="predict" />

Cette famille se charge par la factory `LibreVLM()`, et non `LibreYOLO()` : les
familles VLM ne déclarent aucun chargeur de checkpoint. Le routage par suffixe
de fichier décrit sur les autres pages de modèles ne s'applique donc pas ici.
`set_classes()` définit le vocabulaire que Qwen3-VL doit rechercher ; il est
persistant et reste actif dans tous les appels `predict()`/`track()` suivants
jusqu'à ce que vous le redéfinissiez. Chaque détection porte la même confiance
de substitution. Le filtrage par `conf` fonctionne donc en tout ou rien et ne
constitue pas un classement. `iou` a en revanche un effet pour cette famille :
il élimine une boîte ultérieure de la même classe lorsque son chevauchement avec
une boîte déjà conservée dépasse le seuil, car un générateur répétitif peut sinon
émettre des boîtes presque identiques pour un objet. Contrairement à Florence-2
et Kosmos-2, Qwen3-VL répond aussi à des questions libres au moyen de `chat()`,
le même accès direct documenté sur la factory `LibreVLM`. La CLI LibreYOLO ne
couvre pas cette catégorie : elle n'offre aucune forme `libreyolo predict
model=...` pour celle-ci. Consultez la [prédiction](/docs/predict) pour les
sources, le streaming et le traitement des résultats.

## Variantes

Trois tailles sont proposées : Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct et
Qwen3-VL-8B-Instruct, chargées respectivement par
`LibreVLM("qwen3-vl-2b")`, `LibreVLM("qwen3-vl-4b")` et
`LibreVLM("qwen3-vl-8b")`. Toutes trois déclarent une entrée nominale de
1024 px, mais le redimensionnement intelligent du propre processeur Qwen décide
du canevas réellement transmis au réseau. Cette valeur n'est donc pas une
résolution de fonctionnement fixe comme pour les autres familles de ce site.
LibreYOLO n'a publié aucun benchmark comparant l'exactitude des trois tailles.

LibreYOLO n'entraîne, ne valide et n'exporte pas Qwen3-VL : `train()`, `val()`
et `export()` lèvent tous `NotImplementedError` pour chaque famille de cette
catégorie (consultez le niveau de prise en charge ci-dessus). Effectuez le
fine-tuning de Qwen3-VL en amont et chargez les poids obtenus si vous avez besoin
d'un vocabulaire personnalisé intégré. Vérifiez visuellement la sortie de
`predict()` au lieu d'effectuer une passe de validation de style COCO, puisque
chaque détection porte la même confiance de substitution.

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
