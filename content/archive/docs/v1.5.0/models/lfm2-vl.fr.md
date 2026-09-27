---
title: LFM2-VL
families:
  - lfm2vl
seo_title: 'LFM2-VL : détection à vocabulaire ouvert dans LibreYOLO'
description: >-
  Utilisez LFM2-VL dans LibreYOLO pour la détection d'objets à vocabulaire
  ouvert sur l'appareil. Prédisez avec n'importe quelle étiquette textuelle ;
  l'entraînement, la validation et l'export ne sont pas pris en charge.
lead: >-
  LFM2-VL est un modèle vision-langage compact conçu pour l'appareil et publié
  par Liquid AI. LibreYOLO l'intègre comme détecteur d'objets à vocabulaire
  ouvert : toute liste d'étiquettes textuelles devient l'ensemble des classes,
  sans tête fixe ni fine-tuning requis.
keywords:
  - LFM2-VL
  - LFM2
  - Liquid AI
  - modèle vision-langage
  - détection à vocabulaire ouvert
  - VLM
  - VLM edge
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE


        model = LibreLFM2VL(size="450m")


        # Vocabulaire ouvert : tous les mots fonctionnent, sans tête de classe
        fixe.

        # S'applique aux appels predict()/track() suivants jusqu'à redéfinition.

        model.set_classes(["person", "bicycle", "dog"])

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat brut
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # L'accès direct sous l'interface de détection : questions libres,
        # comptage ou toute requête non couverte par le wrapper de boîtes.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 40237f0ecc0d2cd5
---

## Installer

LFM2-VL nécessite l'extra `vlm`, qui installe `transformers` pour le
backbone de template de chat.

```bash
pip install "libreyolo[vlm]"
```

## Prédire

`LibreLFM2VL` est une classe Python, pas un checkpoint `.pt` : elle n'est pas
chargée par la factory `LibreYOLO()` et la CLI `libreyolo` ne la résout pas.
La factory `LibreVLM(...)` (`from libreyolo import LibreVLM`) permet aussi
d'accéder à cette famille par un alias, par exemple `LibreVLM("lfm2-vl-450m")` ;
elle construit la classe utilisée ci-dessous. Les poids proviennent du propre
dépôt Hugging Face de Liquid AI, et non d'un miroir LibreYOLO ; le premier appel
les télécharge et les met en cache localement, après avoir journalisé une fois
un avis de licence.

<code-tabs name="predict" />

`result.boxes` contient les détections analysées comme pour toute autre famille.
La confiance est une valeur de substitution : LFM2-VL n'émet aucun score par
boîte, chaque détection reçoit donc la même confiance constante, et `conf=` ne
fait qu'éliminer les lignes sous cette constante sans les classer. `iou` élimine
les boîtes presque identiques d'une même classe au-dessus du chevauchement donné,
un effet secondaire du décodage glouton qui répète un objet ; il ne s'agit pas
d'une passe NMS par classe. Si vous omettez `set_classes()`, le vocabulaire prend
par défaut les 80 noms de COCO. Consultez la [prédiction](/docs/predict) pour les
sources, le streaming et le traitement des résultats.

## Variantes

Deux tailles sont proposées, 450m et 1.6b, toutes deux issues de la version
LFM2.5-VL de Liquid AI et conçues pour un déploiement sur l'appareil. Le banc
d'essai de LibreYOLO n'a pas mesuré cette famille, aucune valeur d'exactitude
publiée ne permet donc de les comparer ; choisissez la taille en fonction de
votre propre budget de calcul.

LibreYOLO expose cette famille uniquement pour la prédiction. `train()`, `val()`
et `export()` lèvent tous `NotImplementedError` : effectuez le fine-tuning en
amont et chargez plutôt le résultat, la validation du dataset est ignorée parce
qu'une confiance de substitution rendrait la mAP COCO trompeuse, et l'export
dépasse le périmètre d'un modèle génératif sans dictionnaire d'état à tracer.

## Licence

<provenance-box>

La LFM Open License v1.0 autorise l'utilisation commerciale, la reproduction et
la modification, mais uniquement sous un seuil de chiffre d'affaires annuel de
10 millions de dollars ; une entité juridique qui atteint ou dépasse ce seuil
n'est pas du tout couverte par cet accord pour un usage commercial et doit
contacter directement Liquid AI. Les organisations à but non lucratif éligibles
sont exemptées du seuil pour un usage non commercial ou de recherche. LibreYOLO
n'inclut aucun code source de LiquidAI, puisque le modèle est chargé par la
bibliothèque Apache-2.0 `transformers`, et n'héberge ni ne redistribue les poids :
`LibreLFM2VL` télécharge la taille correspondante directement depuis le propre
dépôt Hugging Face de Liquid AI à sa première exécution et journalise un avis
unique avant ce téléchargement.

</provenance-box>

## Citation

<citation-block />
