---
title: SmolVLM2
families:
  - smolvlm2
seo_title: "SmolVLM2 dans LibreYOLO\_: détection à vocabulaire ouvert"
description: "SmolVLM2 dans LibreYOLO\_: installez, définissez un vocabulaire ouvert, puis lancez une prédiction ou discutez avec le modèle vision-langage Apache-2.0 de Hugging Face."
lead: "SmolVLM2 est le petit modèle vision-langage de Hugging Face. LibreYOLO l'encapsule comme détecteur d'objets à vocabulaire ouvert et expose directement son chat libre\_: fournissez une liste de classes pour effectuer une détection, ou posez-lui une question."
keywords:
  - SmolVLM2
  - modèle vision-langage
  - détection vocabulaire ouvert
  - petit modèle multimodal
  - Hugging Face
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")
        model.set_classes(["cat", "dog"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: >
        from libreyolo import LibreVLM, SAMPLE_IMAGE


        model = LibreVLM("smolvlm2-500m")


        # La porte de sortie sous l'outil pratique de détection : toute
        question,

        # et pas seulement une requête de bounding box.

        answer = model.chat(SAMPLE_IMAGE, "What is the cat doing?")

        print(answer)
source_hash: b30823b62d6347b5
---

## Installer

SmolVLM2 appartient au niveau VLM comme détecteur de LibreYOLO, une surface
produit distincte des familles basées sur des checkpoints et dotée de sa
propre fabrique. Il nécessite l'extra `vlm`, qui installe aussi `num2words`,
une dépendance du processeur propre à SmolVLM2.

```bash
pip install "libreyolo[vlm]"
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et
mis en cache localement.

<code-tabs name="predict" />

Cette famille se charge avec la fabrique `LibreVLM()` et non `LibreYOLO()`\u00a0:
les familles VLM ne déclarent aucun chargeur de checkpoint, le routage par
suffixe de fichier décrit sur les autres pages de modèles ne s'applique donc
pas ici. `set_classes()` définit le vocabulaire que SmolVLM2 doit rechercher.
Ce réglage est persistant et reste appliqué à tous les appels `predict()` ou
`track()` ultérieurs jusqu'à ce que vous le redéfinissiez. SmolVLM2 ne
nécessite aucune substitution de parseur dans LibreYOLO\u00a0: il suit la même
sortie avec template de chat et JSON que la valeur par défaut partagée du
niveau, son prompt de détection et son format de bounding boxes ne sont donc
pas propres à la famille. Chaque détection porte la même confiance factice.
Le filtrage `conf` fonctionne ainsi en tout ou rien plutôt que comme un
classement\u00a0; `iou` a bien un effet, en supprimant une bounding box ultérieure
de même classe dès qu'elle chevauche une bounding box déjà conservée au-delà
du seuil, car un générateur répétitif peut sinon émettre plusieurs bounding
boxes presque identiques pour un même objet. SmolVLM2 répond aussi à des
questions libres par `chat()`, la même porte de sortie que celle documentée
pour la fabrique `LibreVLM`. Le CLI de LibreYOLO ne couvre pas ce niveau\u00a0: il
n'existe aucune forme `libreyolo predict model=...` pour celui-ci. Consultez
la [prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Variantes

Une taille figure dans le registre\u00a0: SmolVLM2-500M-Video-Instruct, chargée avec
`LibreVLM("smolvlm2-500m")`. SmolVLM2 est un détecteur moins performant que
les modèles de grounding spécialisés de ce niveau. Le propre wrapper de
LibreYOLO le décrit comme la démonstration qu'une nouvelle famille n'a besoin
d'aucun parsing particulier pour fonctionner ici, et non comme la meilleure
option à vocabulaire ouvert.

LibreYOLO n'entraîne, ne valide et n'exporte pas SmolVLM2\u00a0: `train()`, `val()`
et `export()` lèvent tous `NotImplementedError` pour chaque famille de ce
niveau (consultez le niveau de prise en charge ci-dessus). Effectuez le
fine-tuning de SmolVLM2 upstream et chargez les poids obtenus si vous avez
besoin d'intégrer un vocabulaire personnalisé. Vérifiez visuellement la sortie
de `predict()` plutôt qu'avec une passe de validation de type COCO, puisque
chaque détection porte la même confiance factice.

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
