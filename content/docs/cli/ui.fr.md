---
title: libreyolo ui
seo_title: "référence de la commande libreyolo ui"
description: "Lancer l'interface web d'inférence locale : adresse d'écoute, comportement du port, choix de l'appareil et fin du processus."
lead: "Lance un serveur web local qui accepte les images déposées ou collées, leur applique le modèle choisi et affiche les résultats dans le navigateur."
keywords: [libreyolo ui cli, interface web libreyolo, inférence locale dans le navigateur, inférence glisser-déposer, libreyolo ui port]
last_verified: "1.5.0"
meta:
  - label: Commande
    value: libreyolo ui
    mono: true
  - label: Sortie
    value: "Une URL de serveur sur stdout, puis le processus reste au premier plan"
snippets:
  examples:
    - label: Simple
      language: bash
      code: |
        libreyolo ui
    - label: Port fixe, sans navigateur
      language: bash
      code: |
        libreyolo ui port=9000 no_browser=true
    - label: Sur le CPU, lisible par une machine
      language: bash
      code: |
        libreyolo ui device=cpu json=true
---

## Synopsis

```bash
libreyolo ui [key=value ...]
```

Les arguments sont des paires `key=value`, et la forme POSIX fonctionne aussi,
donc `port=9000` et `--port 9000` sont le même argument.

## Arguments

| Argument | Défaut | Signification |
|---|---|---|
| `host` | `127.0.0.1` | Hôte ou interface sur lequel écouter |
| `port` | `8000` | Port sur lequel écouter. Passe au suivant libre s'il est occupé |
| `device` | `auto` | Appareil : `0`, `cpu`, `mps`, `auto` |
| `no_browser` | `false` | Ne pas ouvrir automatiquement le navigateur |
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Supprimer stderr |
| `verbose` | `false` | Sortie stderr détaillée |

## Exemples

<code-tabs name="examples" />

## Notes

Par défaut, l'écoute se fait sur la boucle locale, donc l'interface n'est
accessible que depuis cette machine.

Si le port demandé est occupé, la commande essaie le suivant et continue
jusqu'à vingt ports au-delà de celui qui a été demandé. Si les vingt échouent,
elle se termine avec `io_error` et suggère de passer un autre port. L'URL
affichée sur stdout correspond au port réellement lié, donc lisez-la plutôt que
de supposer que c'est celui que vous avez demandé.

Sauf si `no_browser=true`, un onglet de navigateur s'ouvre sur cette URL peu
après la mise en écoute.

La commande sert ensuite au premier plan jusqu'à Ctrl+C, qui arrête proprement
le serveur. Il n'y a pas de mode détaché ; passez-la en arrière-plan avec votre
shell si vous voulez récupérer le terminal.

`json=true` affiche l'URL et l'appareil dans un seul objet avec
`schema_version` avant le démarrage du serveur, c'est ainsi qu'un script
récupère le port lié.

Voir aussi : [`libreyolo label`](/docs/cli/label) pour dessiner des boîtes
englobantes et enregistrer des étiquettes, [`libreyolo monitor`](/docs/cli/monitor)
pour suivre les entraînements. Les deux sont des serveurs web locaux avec le
même comportement de port et de navigateur.
