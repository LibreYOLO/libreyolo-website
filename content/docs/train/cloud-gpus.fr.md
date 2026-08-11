---
title: Entraînement sur un GPU loué
seo_title: Entraîner LibreYOLO sur un GPU cloud loué
description: >-
  Exécutez un entraînement LibreYOLO sur un GPU loué ou serverless : préparez
  les données, installez, lancez, suivez en direct, récupérez les poids et
  cessez la facturation.
lead: >-
  Un GPU loué transforme un entraînement en une tâche avec un début, une fin et
  une facture. Le travail est identique à un entraînement local ; ce qui change,
  c'est l'envoi des données, le suivi depuis l'extérieur, la récupération des
  poids et l'arrêt de la machine.
keywords:
  - entraînement gpu cloud
  - louer un gpu
  - entraînement vast.ai
  - gpu serverless modal
  - gpu beam
  - entraînement à distance
  - héberger dataset hugging face
  - coût gpu par époque
last_verified: 1.5.0
snippets:
  install:
    - label: Sur la machine
      language: bash
      code: >
        pip install libreyolo


        # Ajoutez seulement les extras nécessaires : rfdetr pour entraîner
        RF-DETR,

        # lora pour le fine-tuning efficace, onnx pour exporter ensuite.

        pip install "libreyolo[rfdetr,lora]"
    - label: Vérifier le GPU avant toute chose
      language: python
      code: |
        import torch

        print(torch.__version__, torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))

        # Un wheel pour une autre architecture renvoie True, puis échoue
        # au premier vrai kernel. Exécutez-en donc un.
        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  stage:
    - label: Empaqueter et envoyer une fois depuis votre machine
      language: bash
      code: >
        tar cf my-dataset.tar my-dataset/

        huggingface-cli upload my-org/my-dataset my-dataset.tar --repo-type
        dataset
    - label: Préparer sur la machine
      language: python
      code: |
        import tarfile

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "my-org/my-dataset", "my-dataset.tar", repo_type="dataset"
        )
        with tarfile.open(path) as archive:
            archive.extractall("/root/data")
  launch:
    - label: Détaché pour survivre à une déconnexion
      language: bash
      code: |
        nohup libreyolo train \
          model=LibreYOLO9s.pt \
          data=/root/data/my-dataset/data.yaml \
          epochs=100 batch=-1 imgsz=640 \
          project=/root/runs name=run1 \
          > /root/train.log 2>&1 &
    - label: Multi-GPU depuis un fichier Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="/root/data/my-dataset/data.yaml",
                epochs=100,
                batch=64,          # batch global sur tous les GPU
                device="0,1,2,3",
                project="/root/runs",
                name="run1",
            )
  watch:
    - label: Une lecture peu coûteuse
      language: bash
      code: |
        cat /root/runs/run1/status.json
    - label: Depuis un script
      language: python
      code: |
        import json

        with open("/root/runs/run1/status.json") as handle:
            status = json.load(handle)

        print(status["state"], status["current_epoch"], status["eta_seconds"])
        print(status.get("metrics"))
    - label: Dans un navigateur par tunnel SSH
      language: bash
      code: >
        # Sur la machine (écoute sur 127.0.0.1:8420 par défaut) :

        libreyolo monitor /root/runs/run1 --no-browser


        # Depuis votre machine, ouvrez ensuite http://localhost:8420 localement
        :

        #   ssh -L 8420:localhost:8420 <user>@<host>
  push:
    - label: Envoyer les poids vers un stockage permanent
      language: bash
      code: |
        huggingface-cli upload my-org/my-run \
          /root/runs/run1/weights/best.pt best.pt
source_hash: 75d314de06aca3b6
---

## Avant toute location

Deux décisions coûtent beaucoup plus cher plus tard que maintenant.

Placez d'abord le dataset sur un CDN. L'empaqueter dans un seul fichier tar au
sein d'un dépôt de dataset Hugging Face fonctionne de la même façon chez tous
les fournisseurs, le distribue rapidement auprès de chacun et ne demande rien
d'autre qu'un `HF_TOKEN` dans l'environnement de la tâche lorsque le dépôt est
privé. Envoyer un dataset depuis une connexion domestique ou le récupérer sur
la machine depuis une origine lente revient à payer du temps GPU d'attente.

<code-tabs name="stage" />

Dimensionnez ensuite le disque. Les fournisseurs qui facturent le stockage le
font selon la capacité allouée, pas la capacité utilisée, et un disque ne peut
pas être réduit après sa création. Additionnez les données préparées, les
checkpoints et environ 30 % de marge, puis arrêtez-vous là.

## Installer sur la machine

<code-tabs name="install" />

Installez d'abord PyTorch si l'image ne contient pas déjà un build CUDA adapté
à la carte, puis LibreYOLO, afin que pip ne résolve pas sa propre version de
torch réservée au CPU. Le second extrait n'est pas une formalité facultative :
un wheel construit pour la mauvaise architecture GPU indique
`torch.cuda.is_available() == True`, puis échoue à la première vraie opération
avec `CUDA error: no kernel image is available for execution on the device`.
Une multiplication matricielle le détecte avant de perdre une heure de
configuration.

Pointez `HF_HOME` vers un stockage persistant si le fournisseur propose un
volume, afin que les téléchargements de checkpoints et de datasets survivent
entre les exécutions.

## Lancer

Exécutez la tâche en mode détaché. Une session interactive interrompue par la
perte de votre connexion réseau emporte aussi l'entraînement.

<code-tabs name="launch" />

`batch=-1` est particulièrement utile ici, car vous utilisez généralement une
carte sur laquelle vous n'avez encore jamais entraîné de modèle. Cette option
teste le modèle en mode entraînement avec une véritable passe backward et
sélectionne la plus grande puissance de deux qui tient en mémoire, ce qui est
plus rapide que de découvrir la limite par une erreur de mémoire insuffisante
après vingt minutes. Consultez les
[hyperparamètres](/docs/train/hyperparameters).

Sur une machine multi-GPU, `device="0,1,2,3"` lance automatiquement un worker
par GPU, et `batch` reste le batch global partagé entre eux. La protection
`__main__` est obligatoire, car chaque worker réimporte le script. Ce point et
le reste du comportement distribué figurent dans la page sur
l'[entraînement multi-GPU](/docs/train/multi-gpu).

## Suivre depuis l'extérieur

Chaque exécution écrit `status.json` dans son répertoire, en le réécrivant de
façon atomique à chaque époque. C'est une lecture peu coûteuse : quelques
centaines d'octets contenant l'état, l'époque actuelle, l'ETA et les dernières
mesures, sans analyser de journal.

<code-tabs name="watch" />

Le fichier `metrics.jsonl` voisin contient l'historique complet par époque et
`train.log` la sortie de la console. `libreyolo monitor` fournit un tableau de
bord dans le navigateur qui utilise les trois avec la seule bibliothèque
standard. La machine n'a donc besoin de rien d'autre que LibreYOLO. Accédez-y
par redirection de port SSH.

Aucun de ces fichiers ne touche au processus d'entraînement. Ils peuvent donc
se rattacher à une exécution active, rouvrir une exécution terminée ou inspecter
une exécution qui a planté.

## Récupérer les poids avant de cesser de payer

La machine est jetable. Envoyez les checkpoints à chaque étape importante, pas
seulement à la fin, car un plantage, une préemption ou l'épuisement du crédit
ferait sinon perdre toute l'exécution.

<code-tabs name="push" />

`weights/best.pt` et `weights/last.pt` sont écrits à chaque époque et à chaque
amélioration. `save_period=N` ajoute en plus des instantanés
`weights/epoch_<N>.pt`, ce qui rend peu coûteux un envoi en cours d'exécution.
`summary.json` et `results.csv`, lorsque la famille les écrit, sont petits et
méritent aussi d'être récupérés.

Un callback sur `on_train_epoch_end` est la façon propre d'automatiser l'envoi.
Consultez les [loggers d'expériences](/docs/train/loggers), dont les backends
hébergés vous fournissent également les mesures sans toucher à la machine.

## Cesser de payer

C'est la partie qui coûte réellement de l'argent lorsqu'elle est mal gérée, et
la règle dépend du modèle du fournisseur.

Sur une place de marché où vous louez une machine brute, la facturation suit le
temps réel jusqu'à la destruction de l'instance. Un GPU inactif coûte exactement
autant qu'un GPU occupé, si bien que l'arrêt du seul processus d'entraînement
n'économise rien. Une instance arrêtée continue de facturer son disque.

Sur une plateforme serverless où la tâche est une fonction décorée, le
conteneur revient à zéro lorsque la fonction se termine, si bien qu'une machine
oubliée est beaucoup moins probable. Une tâche bloquée sans délai maximal reste
facturée, définissez-en donc toujours un.

Arrêter au lieu de détruire est un vrai levier, mais aussi un vrai piège. Mesuré
sur une machine louée équipée de 8 RTX 4090 et d'un disque de 250 Go le
2026-07-31 : en fonctionnement, elle coûtait 3.4828 $ par heure ; arrêtée, elle
coûtait 0.0694 $ par heure pour le seul disque ; détruite, elle ne coûtait rien.
Cela représente une économie de 98 % tout en conservant l'environnement, les
données préparées et les checkpoints sur place.

Le tarif à l'arrêt se calcule avant la location :

```text
stopped $/hr = allocated_GB * storage_cost_per_GB_per_month / 730
             = 250 * 0.20 / 730 = $0.0694/hr
```

Comparez-le au coût d'une reconstruction : relouer, récupérer l'image,
installer et préparer de nouveau les données. Sur la même machine, une
reconstruction prenait environ 15 minutes de préparation et 43 Go de transfert
entrant, soit environ 1.00 $ au total. Face à 0.0694 $ par heure, un retour dans
les 14 heures environ favorise l'arrêt ; une absence plus longue favorise la
destruction et la reconstruction depuis la copie préparée.

Un risque rend l'arrêt dangereux pour du matériel rare : arrêter libère les
GPU. Rien ne les réserve, si bien que le redémarrage ne réussit que si l'hôte
les a toujours de disponibles. Votre disque est en sécurité, mais pas vos GPU.

## Serverless sous forme de fonction

Si vous préférez ne pas gérer de machine, Modal et Beam exécutent tous deux une
fonction Python décorée sur un GPU et reviennent à zéro lorsqu'elle se termine.
La propre suite de tests nocturnes de LibreYOLO s'exécute sur Modal, et le
fichier `tools/ci/modal_nightly.py` du dépôt de la bibliothèque est l'exemple
fonctionnel à copier.

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")   # Bibliothèques système d'OpenCV
    .pip_install("libreyolo[rfdetr]")
)
app = modal.App("libreyolo-train")
cache = modal.Volume.from_name("libreyolo-cache", create_if_missing=True)


@app.function(gpu="A100", timeout=6 * 60 * 60, volumes={"/cache": cache})
def train():
    import os

    os.environ["HF_HOME"] = "/cache/hf"          # cache les poids entre les exécutions

    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLO9s.pt")
    model.train(data="coco8.yaml", epochs=100, project="/cache/runs")
    cache.commit()                                # rend le volume persistant


@app.local_entrypoint()
def main():
    train.remote()
```

Exécutez-la avec `modal run modal_train.py`. Le système de fichiers du conteneur
est éphémère, donc tout ce qui doit être conservé va dans le volume ou est
envoyé ailleurs. Définissez explicitement `timeout=` ; c'est la seule barrière
entre une exécution bloquée et une facture sans fin.

Beam adopte la même forme avec un décorateur `@function`, un `Volume` et
`train.remote()` appelé depuis `__main__`.

## Dimensionner selon le coût par tâche

Le coût horaire est la mauvaise valeur à optimiser. Un petit modèle laisse une
grande carte à moitié inactive, si bien qu'un GPU moins cher et plus lent coûte
souvent moins par époque. Exécutez le profileur pendant quelques étapes sur la
carte louée avant de vous engager dans une longue exécution : si le verdict est
`dataloader` ou `host / launch`, un GPU plus rapide n'apporte rien, alors que
plus de workers ou un batch plus grand apportent beaucoup. Consultez les
[performances d'entraînement](/docs/train/performance).

## Pages connexes

- [Datasets](/docs/train/datasets) pour la disposition attendue de l'archive
  préparée et la commande doctor qui détecte les problèmes avant le début de la
  facturation du GPU.
- [Entraînement multi-GPU](/docs/train/multi-gpu) pour les machines à plusieurs
  cartes.
