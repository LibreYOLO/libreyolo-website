---
title: Résolution des problèmes
seo_title: Corriger les erreurs courantes de LibreYOLO
description: >-
  Erreurs les plus souvent levées par LibreYOLO, signification et correction.
  Inclut deux échecs qui produisent une sortie incorrecte au lieu de lever une
  erreur.
lead: "Erreurs regroupées selon le message affiché. Les deux dernières entrées couvrent le problème inverse\_: le code s'exécute, renvoie un résultat plausible, mais celui-ci est faux."
keywords:
  - erreur libreyolo
  - modulenotfounderror libreyolo
  - mémoire cuda insuffisante libreyolo
  - notimplementederror libreyolo
  - dépannage libreyolo
last_verified: 1.5.0
source_hash: e271ab29b789865a
---

Les erreurs sont regroupées selon le texte affiché. Si votre message n'est pas
présent, la [FAQ](/docs/faq) répond aux questions qui ne sont pas des erreurs,
et `libreyolo models` indique ce que votre installation peut réellement charger.

## ModuleNotFoundError nomme un package que vous n'avez jamais importé

Certaines familles nécessitent un extra facultatif. Le message nomme le package
manquant plutôt que l'extra, la correction n'est donc pas toujours évidente
dans la traceback.

Exécutez `libreyolo models`. Toute famille dont une dépendance manque est
affichée avec la commande pip exacte qui l'active. Vous n'avez donc pas à
retrouver vous-même l'extra correspondant au package.
`libreyolo models --json` affiche les mêmes informations sous forme d'objet.

La [page d'installation](/docs/install) énumère chaque extra et son rôle.

## L'inférence ONNX nécessite onnxruntime

```
ImportError: ONNX inference requires onnxruntime. Install with: pip install onnxruntime
```

Le package de base ne dépend d'aucun runtime, car le choix dépend de votre
matériel. Installez `onnxruntime` pour CPU ou `onnxruntime-gpu` pour CUDA. Les
deux fournissent le même module `onnxruntime`, installez-en un seul.

## Modèle ONNX introuvable

```
FileNotFoundError: ONNX model not found: <path>
```

Le chemin est résolu par rapport au répertoire de travail et non au script.
Ce message apparaît aussi lorsqu'un export a silencieusement écrit ailleurs\u00a0:
`export()` renvoie le chemin écrit. Récupérez cette valeur au lieu de supposer
un nom.

## NotImplementedError depuis train()

Toutes les familles ne s'entraînent pas. Certaines sont portées uniquement
pour la prédiction, la validation et l'export. Leur méthode `train()` lève une
erreur au lieu de simuler une exécution.

L'[entrée de FAQ](/docs/faq) explique ce choix. Pour vérifier une famille avant
d'écrire un script d'entraînement, consultez sa page de modèle.

## NotImplementedError depuis export()

Une famille peut prendre en charge une tâche sans pouvoir l'exporter. EoMT est
un cas fréquent\u00a0: `export()` accepte la tâche semantic et lève une erreur pour
`segment` et `panoptic`, car le contrat de runtime de masques de requêtes dont
elles ont besoin n'est pas défini.

```
NotImplementedError: LibreEoMT instance and panoptic export need query-mask runtime contracts.
```

Chaque page de famille contient une matrice d'export qui indique les
combinaisons de tâches et formats validées.

## Mémoire CUDA insuffisante

Réduisez d'abord `batch`, puis `imgsz`. Tous deux modifient la mémoire à peu
près proportionnellement à leur taille, mais vous pouvez réduire le batch sans
changer ce que voit le modèle.

Si l'échec se produit pendant la validation plutôt que pendant l'entraînement,
celle-ci utilise sa propre taille de batch. Réduisez-la également.

Sous Windows, un GPU d'affichage possède un second mode d'échec qui ressemble
à une erreur CUDA aléatoire plutôt qu'à un manque de mémoire\u00a0: le pilote
réinitialise un GPU qui ne répond pas pendant un délai trop long et interrompt
son travail. Les kernels longs sur la carte qui pilote votre écran peuvent le
déclencher.

## Les poids ne se téléchargent pas

Les poids sont récupérés depuis Hugging Face à la première utilisation et mis
en cache localement. La [FAQ](/docs/faq) indique l'emplacement du cache et la
méthode pour travailler entièrement hors ligne.

Si un téléchargement renvoie une erreur 404, vérifiez le nom de fichier
transmis. L'URL en est dérivée, suffixe de tâche compris. Un nom qui ne
correspond à aucun checkpoint publié produit donc une URL inexistante. Le
tableau des checkpoints de chaque page de modèle énumère les noms exacts.

## L'entraînement se fige ou redémarre sous Windows

Windows ne possède pas `fork`. Les workers du dataloader démarrent donc en
réimportant votre script. Sans garde `if __name__ == "__main__":`, chaque
worker relance votre appel d'entraînement, ce qui provoque un deadlock ou crée
des processus sans fin.

```python
def main():
    ...  # build the model and call train()

if __name__ == "__main__":
    main()
```

Définir `workers=0` évite aussi le problème, au prix du débit. La garde est la
meilleure correction.

## Deux échecs qui ne lèvent aucune erreur

Le reste de cette page porte sur des erreurs. Ces deux cas sont pires, car le
code s'exécute et renvoie un élément d'apparence correcte.

### Indexer un résultat unique

`predict()` renvoie un objet `Results` pour une image et une liste pour
plusieurs. Indexer le retour mono-image sélectionne une *détection*, et non une
image\u00a0:

```python
result = model.predict("image.jpg")   # a Results
result.boxes                          # every detection, correct
result[0].boxes                       # ONE detection, silently
```

Aucune erreur n'est levée, car l'indexation d'un objet `Results` est une
opération valide qui renvoie un sous-ensemble. Du code écrit pour la forme
liste rapporte silencieusement une bounding box par image. N'indexez que les
valeurs dont vous savez qu'elles sont des listes.

### Lire les métriques comme des attributs

`val()` renvoie un dictionnaire simple indexé par nom de métrique et non un
objet à accès par attribut\u00a0:

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # correct
metrics.box.map               # AttributeError
```

Les clés utilisent les espaces de noms `metrics/` et `speed/`. Affichez le
dictionnaire une fois pour voir ce que la tâche a produit, car l'ensemble
diffère selon la tâche.

## Vérifier un dataset avant l'entraînement

La plupart des échecs d'entraînement proviennent des datasets.
`libreyolo doctor data.yaml` exécute des contrôles de santé sur un dataset de
détection et rapporte les résultats par gravité. C'est plus rapide que de lire
une traceback à la première époque.

```python
from libreyolo import doctor

report = doctor.diagnose("data.yaml", imgsz=640)
if report.errors:
    ...
```

Consultez la [commande doctor](/docs/cli/doctor) pour le catalogue des
contrôles.
