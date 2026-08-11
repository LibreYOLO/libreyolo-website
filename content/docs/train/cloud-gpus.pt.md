---
title: Treinamento em uma GPU alugada
seo_title: Treine o LibreYOLO em uma GPU alugada na nuvem
description: >-
  Rode um treinamento do LibreYOLO em uma GPU alugada ou serverless: prepare os
  dados, instale, dispare o job, acompanhe ao vivo, recupere os pesos e pare de
  pagar.
lead: >-
  Uma GPU alugada transforma um treinamento em um job com início, fim e fatura.
  O trabalho é o mesmo de treinar localmente; o que muda é colocar os dados lá
  dentro, acompanhar de fora, tirar os pesos e desligar a máquina.
keywords:
  - treinar na gpu em nuvem
  - alugar gpu para treinar
  - vast.ai treinamento
  - modal gpu serverless
  - beam gpu
  - treinamento remoto em gpu
  - subir dataset hugging face
  - custo de gpu por época
last_verified: 1.5.0
snippets:
  install:
    - label: Na máquina
      language: bash
      code: >
        pip install libreyolo


        # Adicione apenas os extras que a execução precisa. rfdetr para treinar

        # RF-DETR, lora para fine-tuning eficiente em parâmetros, onnx para
        exportar depois.

        pip install "libreyolo[rfdetr,lora]"
    - label: Confira a GPU antes de qualquer outra coisa
      language: python
      code: |
        import torch

        print(torch.__version__, torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))

        # Uma wheel compilada para outra arquitetura reporta True e depois falha
        # no primeiro kernel de verdade, então rode um.
        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  stage:
    - label: 'Empacote e envie uma vez, da sua máquina'
      language: bash
      code: >
        tar cf my-dataset.tar my-dataset/

        huggingface-cli upload my-org/my-dataset my-dataset.tar --repo-type
        dataset
    - label: Prepare os dados na máquina
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
    - label: 'Em background, para o job sobreviver a uma desconexão'
      language: bash
      code: |
        nohup libreyolo train \
          model=LibreYOLO9s.pt \
          data=/root/data/my-dataset/data.yaml \
          epochs=100 batch=-1 imgsz=640 \
          project=/root/runs name=run1 \
          > /root/train.log 2>&1 &
    - label: 'Multi-GPU, a partir de um arquivo Python'
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="/root/data/my-dataset/data.yaml",
                epochs=100,
                batch=64,          # batch global entre todas as GPUs
                device="0,1,2,3",
                project="/root/runs",
                name="run1",
            )
  watch:
    - label: Uma leitura barata
      language: bash
      code: |
        cat /root/runs/run1/status.json
    - label: A partir de um script
      language: python
      code: |
        import json

        with open("/root/runs/run1/status.json") as handle:
            status = json.load(handle)

        print(status["state"], status["current_epoch"], status["eta_seconds"])
        print(status.get("metrics"))
    - label: 'No navegador, por um túnel SSH'
      language: bash
      code: |
        # Na máquina (faz bind em 127.0.0.1:8420 por padrão):
        libreyolo monitor /root/runs/run1 --no-browser

        # Da sua máquina, depois abra http://localhost:8420 localmente:
        #   ssh -L 8420:localhost:8420 <user>@<host>
  push:
    - label: Envie os pesos para um lugar permanente
      language: bash
      code: |
        huggingface-cli upload my-org/my-run \
          /root/runs/run1/weights/best.pt best.pt
source_hash: 75d314de06aca3b6
---

## Antes de alugar qualquer coisa

Duas decisões custam mais depois do que custam agora.

Coloque o dataset em uma CDN primeiro. Empacotá-lo como um único tar em um
repositório de dataset do Hugging Face funciona igual em qualquer provedor,
entrega rápido para todos eles, e não precisa de nada além de um `HF_TOKEN` no
ambiente do job quando o repositório é privado. Subir um dataset por uma conexão
doméstica, ou puxá-lo de uma origem lenta na máquina, é tempo de GPU faturado
esperando.

<code-tabs name="stage" />

Depois dimensione o disco. Provedores que cobram armazenamento cobram pela
capacidade alocada, não pela usada, e um disco não pode ser reduzido depois de
criado. Some os dados preparados, os checkpoints e mais ou menos 30 por cento de
folga, e pare por aí.

## Instale na máquina

<code-tabs name="install" />

Instale o PyTorch primeiro se a imagem já não trouxer um build CUDA compatível
com a placa, e só então o LibreYOLO, para o pip não resolver um torch só de CPU
por conta própria. O segundo snippet não é cerimônia opcional: uma wheel
compilada para a arquitetura de GPU errada reporta
`torch.cuda.is_available() == True` e depois falha na primeira operação de
verdade com `CUDA error: no kernel image is available for execution on the
device`. Uma multiplicação de matrizes pega isso antes que uma hora de setup
seja desperdiçada.

Aponte o `HF_HOME` para armazenamento persistente se o provedor oferecer um
volume, para que os downloads de checkpoints e datasets sobrevivam entre
execuções.

## Dispare o job

Rode o job em background. Uma sessão interativa que morre junto com a sua
conexão de rede leva o treinamento junto.

<code-tabs name="launch" />

`batch=-1` vale a pena justamente aqui, porque normalmente você está em uma
placa na qual nunca treinou. Ele sonda o modelo em modo de treinamento com um
backward pass real e escolhe a maior potência de dois que cabe, o que é mais
rápido do que descobrir o teto com um erro de falta de memória vinte minutos
depois. Veja [Hiperparâmetros](/docs/train/hyperparameters).

Em uma máquina multi-GPU, `device="0,1,2,3"` sozinho já cria um worker por GPU,
e `batch` continua sendo o batch global entre todas elas. A guarda `__main__` é
obrigatória, porque cada worker reimporta o script. Isso, e o resto do
comportamento distribuído, está em
[Treinamento multi-GPU](/docs/train/multi-gpu).

## Acompanhe de fora

Toda execução escreve um `status.json` no diretório da run, reescrito
atomicamente a cada época. É a leitura barata: algumas centenas de bytes
carregando o estado, a época atual, o ETA e as métricas mais recentes, sem
precisar parsear um log.

<code-tabs name="watch" />

Ao lado dele, o `metrics.jsonl` tem o histórico completo por época, e o
`train.log` tem a saída do console. O `libreyolo monitor` serve um dashboard de
navegador em cima dos três usando apenas a biblioteca padrão, então não precisa
de nada instalado na máquina além do próprio LibreYOLO. Chegue até ele por um
redirecionamento de porta SSH.

Nenhum deles toca no processo de treinamento, então eles se conectam a uma
execução em andamento, reabrem uma que terminou ou inspecionam uma que quebrou.

## Tire os pesos antes de parar de pagar

A máquina é descartável. Envie checkpoints em marcos intermediários, não só no
fim, porque senão uma queda, uma preempção ou o crédito acabando perde a
execução inteira.

<code-tabs name="push" />

`weights/best.pt` e `weights/last.pt` são escritos a cada época e a cada
melhora. `save_period=N` adiciona snapshots `weights/epoch_<N>.pt` por cima, que
é o que torna barato um envio no meio da execução. `summary.json` e
`results.csv`, onde a família os escreve, são pequenos e também vale a pena
levar.

Um callback em `on_train_epoch_end` é o jeito limpo de automatizar o envio. Veja
[Loggers de experimentos](/docs/train/loggers), onde os backends hospedados
também te dão as métricas sem tocar na máquina.

## Pare de pagar

Essa é a parte que custa dinheiro de verdade quando dá errado, e a regra muda
conforme o modelo do provedor.

Em um marketplace onde você aluga uma máquina crua, a cobrança corre no relógio
até a instância ser destruída. Uma GPU ociosa cobra exatamente como uma ocupada,
então matar o processo de treinamento não economiza nada por si só. Uma
instância parada continua cobrando o disco.

Em uma plataforma serverless onde o job é uma função decorada, o contêiner
escala para zero quando a função retorna, então é bem menos provável esquecer
uma máquina ligada. Um job travado sem timeout continua cobrando, então sempre
defina um.

Parar em vez de destruir é uma alavanca de verdade, e uma armadilha de verdade.
Medido em uma 8x RTX 4090 alugada com um disco de 250 GB em 2026-07-31: rodando
cobrava $3.4828 por hora, parada cobrava $0.0694 por hora só pelo disco, e
destruída não cobrava nada. Isso é uma economia de 98 por cento mantendo o
ambiente, os dados preparados e os checkpoints no lugar.

A taxa parada é uma conta que dá para fazer antes de alugar:

```text
stopped $/hr = allocated_GB * storage_cost_per_GB_per_month / 730
             = 250 * 0.20 / 730 = $0.0694/hr
```

Compare com o que custa reconstruir: alugar de novo, baixar a imagem, instalar e
preparar os dados outra vez. Nessa mesma máquina, reconstruir deu cerca de 15
minutos de setup mais 43 GB de transferência de entrada, algo em torno de $1.00
no total. Contra $0.0694 por hora, voltar dentro de umas 14 horas favorece
parar, e um intervalo maior favorece destruir e reconstruir a partir da cópia
preparada.

Um risco torna parar inseguro para hardware escasso: parar libera as GPUs. Nada
as reserva, então religar só dá certo se o host ainda as tiver livres. Seu disco
está seguro; suas GPUs não.

## Serverless, como uma função

Se você prefere não gerenciar uma máquina, tanto o Modal quanto o Beam rodam uma
função Python decorada em uma GPU e escalam para zero quando ela retorna. A
própria suíte de testes noturnos do LibreYOLO roda no Modal, e
`tools/ci/modal_nightly.py` no repositório da biblioteca é o exemplo funcional
dentro do repo para copiar.

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")   # bibliotecas de sistema do OpenCV
    .pip_install("libreyolo[rfdetr]")
)
app = modal.App("libreyolo-train")
cache = modal.Volume.from_name("libreyolo-cache", create_if_missing=True)


@app.function(gpu="A100", timeout=6 * 60 * 60, volumes={"/cache": cache})
def train():
    import os

    os.environ["HF_HOME"] = "/cache/hf"          # pesos em cache entre execuções

    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLO9s.pt")
    model.train(data="coco8.yaml", epochs=100, project="/cache/runs")
    cache.commit()                                # persiste o volume


@app.local_entrypoint()
def main():
    train.remote()
```

Rode com `modal run modal_train.py`. O sistema de arquivos do contêiner é
efêmero, então tudo que vale a pena guardar vai para o volume ou é enviado para
fora. Defina o `timeout=` explicitamente; é a única coisa entre uma execução
travada e uma conta sem fim.

O Beam tem o mesmo formato, com um decorador `@function`, um `Volume` e
`train.remote()` chamado a partir do `__main__`.

## Dimensione pelo custo por job

$/hr é o número errado para otimizar. Um modelo pequeno deixa uma placa grande
metade ociosa, então uma GPU mais barata e mais lenta costuma sair mais barata
por época. Rode o profiler por alguns passos na placa alugada antes de se
comprometer com uma execução longa: se o veredito for `dataloader` ou
`host / launch`, uma GPU mais rápida não compra nada e mais workers ou um batch
maior compram muito. Veja
[Desempenho de treinamento](/docs/train/performance).

## Relacionados

- [Datasets](/docs/train/datasets) para o layout que o arquivo preparado deve
  ter, e o comando doctor que pega os problemas antes de uma GPU estar cobrando.
- [Treinamento multi-GPU](/docs/train/multi-gpu) para máquinas com várias placas.
