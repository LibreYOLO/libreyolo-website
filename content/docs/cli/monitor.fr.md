---
title: libreyolo monitor
seo_title: "référence de la commande libreyolo monitor"
description: "Sert un tableau de bord en direct pour les entraînements : les arguments et leurs valeurs par défaut, ce que le serveur lit sur le disque, et comment un seul serveur couvre plusieurs runs."
lead: "Sert un tableau de bord web pour les entraînements, en lisant les artefacts qu'un run écrit sur le disque. Il ne s'attache jamais au processus d'entraînement, si bien que les runs en cours, terminés et plantés s'affichent tous."
keywords: [libreyolo monitor cli, dashboard entraînement, suivre un entraînement en direct, libreyolo monitor port, visualiseur de métriques d'entraînement]
last_verified: "1.5.0"
meta:
  - label: Commande
    value: libreyolo monitor
    mono: true
  - label: Sortie
    value: "Une URL de serveur sur stdout, puis le processus reste au premier plan"
snippets:
  examples:
    - label: Basique
      language: bash
      code: |
        # Surveille runs/ et liste tous les runs qu'il contient.
        libreyolo monitor
    - label: Une autre racine de runs
      language: bash
      code: |
        libreyolo monitor experiments/
    - label: Un seul run, port fixe, sans navigateur
      language: bash
      code: |
        libreyolo monitor runs/train/exp port=9100 no_browser=true
---

## Synopsis

```bash
libreyolo monitor [<run-dir|runs-root>] [key=value ...]
```

Le répertoire est positionnel. Tout le reste est une paire `key=value`, et la
forme POSIX fonctionne aussi, donc `port=9100` et `--port 9100` sont le même
argument.

## Arguments

| Argument | Valeur par défaut | Signification |
|---|---|---|
| `run_dir` | `runs` | Positionnel. Une racine de runs à surveiller, ou un répertoire de run unique à ouvrir directement. Dans les deux cas, tous les runs situés sous la racine sont listés |
| `host` | `127.0.0.1` | Hôte ou interface sur laquelle écouter |
| `port` | `8420` | Port sur lequel écouter. Passe au suivant libre s'il est occupé |
| `no_browser` | `false` | Ne pas ouvrir automatiquement le navigateur |
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Supprime la sortie stderr |
| `verbose` | `false` | Sortie stderr détaillée |

## Exemples

<code-tabs name="examples" />

## Notes

### Un seul serveur, plusieurs runs

Le serveur surveille une racine de runs plutôt qu'un run unique, et adresse
chaque run par URL, si bien que plusieurs runs sur une même machine partagent un
seul port. Ouvrez l'URL racine pour l'index, ou un onglet par run ; le paramètre
`?run=` de chaque URL indique lequel.

Pointer la commande vers un répertoire de run unique enracine le serveur sur le
répertoire parent, de sorte que les runs voisins apparaissent toujours dans
l'index, et ouvre un lien direct vers celui qui a été nommé.

### Ce qu'il lit

Le tableau de bord est construit à partir des fichiers qu'écrit
`libreyolo train` : `status.json`, `metrics.jsonl`, `train.log` et les images du
run. Rien n'est lu depuis le processus d'entraînement lui-même, donc un run
terminé, ou mort, s'affiche exactement comme un run en cours.

### Prérequis et ports

Au moins un run doit déjà exister. Sans argument et sans répertoire `runs/`, la
commande se termine avec `source_not_found` ; il en va de même lorsque le
répertoire indiqué ne contient aucun run.

Un port occupé passe au suivant, jusqu'à vingt au-delà de celui demandé. Si les
vingt échouent, la commande se termine avec `io_error`. L'URL affichée sur stdout
correspond au port réellement lié.

La commande sert au premier plan jusqu'à Ctrl+C. `json=true` affiche l'URL, la
racine surveillée et le nombre de runs trouvés, sous la forme d'un seul objet
avec `schema_version`.

Voir aussi : [`libreyolo train`](/docs/cli/train), dont les arguments `project`
et `name` déterminent où atterrissent ces répertoires de run.
