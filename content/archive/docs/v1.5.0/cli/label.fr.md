---
title: libreyolo label
seo_title: Référence de la commande libreyolo label
description: "Lance l'outil local d'annotation de bounding boxes\_: les arguments et leurs valeurs par défaut, l'option d'assistance IA, et ce qu'expose le binding sur une interface réseau."
lead: >-
  Démarre un outil web local pour dessiner et modifier des bounding boxes. Il
  écrit des fichiers d'étiquettes au format natif de LibreYOLO, donc un dataset
  annoté ici s'entraîne sans aucune étape de conversion.
keywords:
  - libreyolo label cli
  - outil annotation bounding box
  - étiqueter un dataset yolo
  - auto labeling cli
  - partager libreyolo label
last_verified: 1.5.0
meta:
  - label: Commande
    value: libreyolo label
    mono: true
  - label: Sortie
    value: "Une URL de serveur sur stdout\_; les étiquettes sont écrites dans labels/*.txt à côté des images"
snippets:
  examples:
    - label: Base
      language: bash
      code: |
        # Ouvre l'accueil du projet ; choisissez ou créez un dataset ensuite.
        libreyolo label
    - label: 'Manuel uniquement, port fixe'
      language: bash
      code: |
        libreyolo label no_assist=true port=9200 no_browser=true
    - label: Laisser vos collègues se connecter
      language: bash
      code: |
        libreyolo label share=true
source_hash: bddad245877793b1
---

## Synopsis

```bash
libreyolo label [data=<dataset.yaml|folder>] [key=value ...]
```

Les arguments sont des paires `key=value`, et la forme POSIX fonctionne aussi,
donc `port=9200` et `--port 9200` sont le même argument.

## Arguments

| Argument | Valeur par défaut | Signification |
|---|---|---|
| `data` | | YAML ou dossier du dataset à ouvrir directement. Démarre sur l'accueil du projet si non renseigné |
| `host` | `127.0.0.1` | Hôte ou interface sur lequel faire le binding |
| `port` | `8000` | Port sur lequel faire le binding. Passe au suivant libre s'il est occupé |
| `device` | `auto` | Appareil pour l'auto-étiquetage IA : `0`, `cpu`, `mps`, `auto` |
| `no_assist` | `false` | Désactive l'auto-étiquetage IA, ne laissant qu'un étiqueteur manuel |
| `no_browser` | `false` | Ne pas ouvrir automatiquement le navigateur |
| `share` | `false` | Fait le binding sur `0.0.0.0` pour que vos collègues sur votre réseau puissent se connecter |
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Supprime stderr |
| `verbose` | `false` | Sortie stderr détaillée |

## Exemples

<code-tabs name="examples" />

## Notes

### Ce qu'il écrit

Les bounding boxes sont enregistrées dans des fichiers `labels/*.txt` au format
natif de LibreYOLO, celui que lit `libreyolo train`, donc il n'y a rien à
convertir ensuite. Cette version ne gère que les bounding boxes. Les
modifications sont enregistrées au fur et à mesure que vous passez d'une image à
l'autre.

### Ouvrir un dataset

Sans `data`, l'outil démarre sur l'accueil du projet et le dataset se choisit ou
se crée depuis le navigateur. Passer `data=path/to/data.yaml` ouvre ce dataset
directement, et la ligne de démarrage indique le nombre d'images, le nombre de
classes et si le dataset est accessible en écriture. Un dataset en lecture seule
s'ouvre quand même et explique pourquoi il ne peut pas être écrit.

### Le partage, et ce que fait `host`

`share=true` fait le binding sur l'adresse joker, ce qui permet aux autres
machines de votre réseau d'atteindre l'outil, tandis que les actions
administratives (changer ou supprimer des projets et lancer des calculs)
restent sur cette machine.

Définir `host` sur une interface précise fait quelque chose de différent et de
moins sûr : l'hôte devient indiscernable d'un client réseau, donc chaque client
obtient les droits administratifs. La commande affiche un avertissement sur
stderr quand vous le faites. Préférez `share=true`.

### Ports et arrêt

Un port occupé passe au suivant, jusqu'à vingt au-delà de celui demandé. Si les
vingt échouent, la commande se termine avec `io_error`. L'URL affichée sur stdout
correspond au port réellement utilisé. Avec `share=true`, le résultat contient
aussi `lan_url`, l'adresse que vos collègues doivent ouvrir.

La commande sert au premier plan jusqu'à Ctrl+C.

Voir aussi : [`libreyolo doctor`](/docs/cli/doctor) pour vérifier le dataset
étiqueté avant l'entraînement, et [`libreyolo train`](/docs/cli/train) pour
l'entraîner dessus.
