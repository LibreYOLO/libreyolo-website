---
title: libreyolo doctor
seo_title: "référence de la commande libreyolo doctor"
description: "Vérifiez un dataset de détection avant l'entraînement : les arguments avec leurs valeurs par défaut, les familles de contrôles que vous pouvez ignorer ou sélectionner, et les codes de sortie sur lesquels la CI peut bloquer."
lead: "Exécute une série de contrôles de santé sur un dataset de détection et signale ce qui nuirait à un entraînement : fichiers manquants, étiquettes cassées, images corrompues, fuite entre splits et déséquilibre des classes."
keywords: [libreyolo doctor cli, vérifier un dataset yolo, validation dataset détection, fuite de données entre splits, libreyolo doctor strict]
last_verified: "1.5.0"
meta:
  - label: Commande
    value: libreyolo doctor
    mono: true
  - label: Requis
    value: data
    mono: true
  - label: Sortie
    value: "Un rapport de constats sur stdout. Code de sortie 1 quand des erreurs sont trouvées"
snippets:
  examples:
    - label: Base
      language: bash
      code: |
        # download=true laisse le coco8.yaml fourni télécharger ses images si absentes.
        libreyolo doctor coco8.yaml download=true
    - label: Passe rapide, sans décodage des images
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true fast=true
    - label: Blocage CI sur des contrôles sélectionnés
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true strict=true json=true \
          only=labels,files,config
---

## Synopsis

```bash
libreyolo doctor <data.yaml> [key=value ...]
```

Le dataset est positionnel, et `data=<path>` est accepté comme alternative.
Donner les deux avec des valeurs différentes se termine par `config_conflict`.
Tout le reste est une paire `key=value`, et la forme POSIX fonctionne aussi,
donc `imgsz=1024` et `--imgsz 1024` sont le même argument.

## Arguments

| Argument | Défaut | Signification |
|---|---|---|
| `data` | | Positionnel. YAML de dataset au format de détection YOLO, p. ex. `coco8.yaml`. Requis |
| `imgsz` | `640` | Taille d'image d'entraînement utilisée pour les contrôles exprimés en pixels, comme les objets minuscules |
| `fast` | `false` | Ignore le décodage des images, ce qui supprime les contrôles de corruption, de doublons et de fuite |
| `skip` | | Ids de contrôles ou familles à ignorer, séparés par des virgules, p. ex. `images,labels.tiny_object` |
| `only` | | Ids de contrôles ou familles à exécuter exclusivement, séparés par des virgules |
| `strict` | `false` | Les avertissements font aussi échouer le code de sortie, pour les blocages en CI |
| `download` | `false` | Autorise le téléchargement du dataset par URL s'il est absent. Jamais de scripts |
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Supprime stderr |
| `help_json` | `false` | Affiche le schéma de la commande en JSON et quitte |

### Familles de contrôles

`skip` et `only` acceptent soit un id de contrôle complet, soit un préfixe de
famille, donc `images` sélectionne tous les contrôles `images.*`.

| Famille | Couvre |
|---|---|
| `config` | Le YAML du dataset lui-même : `names` manquant, `nc` face à `names`, splits manquants, `path` non résolu, noms de classes en double |
| `files` | Appariement des images et des étiquettes : étiquettes manquantes, images manquantes, étiquettes orphelines, extensions non prises en charge, collisions de casse |
| `labels` | Contenu des étiquettes : syntaxe, lignes de polygones, ids de classes hors plage, coordonnées hors plage, boîtes dégénérées, objets minuscules, boîtes énormes, rapports d'aspect extrêmes, boîtes en double, images encombrées, fichiers identiques |
| `images` | Données pixel : fichiers corrompus, orientation EXIF, modes colorimétriques inhabituels, dimensions minuscules ou extrêmes, images uniformes, doublons exacts et approchés |
| `splits` | Fuite entre splits, exacte et approchée |
| `balance` | Distribution des classes : classes avec zéro ou peu d'instances, déséquilibre, couverture des splits, ratio d'arrière-plan, asymétrie entre splits |

## Exemples

<code-tabs name="examples" />

## Notes

### Codes de sortie

`0` quand aucune erreur n'a été trouvée, `1` dès qu'un constat est une erreur.
Avec `strict=true`, les avertissements portent aussi le code de sortie à `1`,
ce qui est le réglage que veut un blocage en CI.

Les problèmes d'utilisation ont leurs propres codes : `2` pour un id de
contrôle ou une famille inconnus dans `skip` ou `only`, `3` quand le dataset
est introuvable, et `3` quand le dataset n'a pas la forme d'un dataset de
détection.

### La sélection est résolue avant le scan

`skip` et `only` sont résolus face au registre des contrôles avant toute
lecture sur disque, donc une faute de frappe échoue immédiatement plutôt
qu'après une longue passe sur les images. Un sélecteur qui ne correspond à rien
est une erreur, et le message liste les familles connues.

Si la combinaison de `skip`, `only` et `fast` ne laisse aucun contrôle à
exécuter, c'est aussi une erreur plutôt qu'un succès silencieux.

### Téléchargements

Le dataset n'est pas récupéré sauf si `download=true`, et seuls des
téléchargements par URL sont effectués. Un script Python de téléchargement
embarqué dans un YAML de dataset n'est jamais exécuté par cette commande, quel
que soit le flag.

### Périmètre

Les contrôles sont écrits pour des datasets de détection. Un dataset dont les
étiquettes ont la forme pose, segmentation ou boîte orientée est détecté et
refusé avec `data_invalid` plutôt que noté selon les mauvaises règles.

### Sortie

Le rapport lisible va sur stdout, et `json=true` le remplace par un objet
structuré portant les compteurs du résumé, les statistiques du dataset, chaque
constat, et la liste des contrôles qui ont été ignorés.

En lien : [`libreyolo train`](/docs/cli/train), l'exécution avant laquelle
cette commande est censée être lancée.
