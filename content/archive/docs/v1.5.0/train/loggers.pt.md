---
title: Loggers de experimentos
seo_title: Loggers de experimentos e callbacks no LibreYOLO
description: >-
  Envie as métricas de treinamento para TensorBoard, MLflow, Weights & Biases,
  Comet, ClearML, Neptune ou DVCLive, e escreva seu próprio callback nos quatro
  hooks de treinamento.
lead: >-
  Toda família treinável emite quatro eventos de treinamento. Os loggers
  embutidos são objetos de callback que escutam esses mesmos eventos, portanto
  uma integração com backend e um hook próprio usam uma única interface.
keywords:
  - tensorboard treinamento
  - mlflow tracking
  - weights and biases
  - clearml
  - comet ml
  - neptune
  - dvclive
  - callbacks de treinamento
  - métricas de treinamento csv
  - libreyolo monitor
last_verified: 1.5.0
snippets:
  logger:
    - label: Por nome
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: Instância configurada
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import MLflowLogger

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="coco8.yaml",
            epochs=10,
            loggers=[MLflowLogger(tracking_uri="sqlite:///mlflow.db"), "tensorboard"],
        )
  callback:
    - label: Uma função simples
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEpochEvent


        def on_epoch(event: TrainEpochEvent) -> None:
            print(f"epoch {event.epoch}/{event.total_epochs} loss={event.train_loss:.4f}")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
    - label: Um objeto com vários hooks
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.training import TrainEndEvent, TrainEpochEvent,
        TrainStartEvent



        class RunLog:
            def on_train_start(self, event: TrainStartEvent) -> None:
                print(f"{event.model_family}{event.model_size} -> {event.save_dir}")

            def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
                if event.is_best:
                    print(f"new best at epoch {event.epoch}: {event.best_metric}")

            def on_train_end(self, event: TrainEndEvent) -> None:
                print(f"done in {event.total_seconds:.0f}s")


        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="coco8.yaml", epochs=10, callbacks=RunLog())
  monitor:
    - label: Acompanhar uma execução no navegador
      language: bash
      code: |
        libreyolo monitor                     # a execução mais recente em runs/
        libreyolo monitor runs/train/exp      # uma execução específica
source_hash: de035acbaed32804
---

## Ative um logger

`loggers=` aceita um nome registrado, uma instância configurada ou um iterável
misturando os dois.

<code-tabs name="logger" />

Os nomes não diferenciam maiúsculas de minúsculas. O conjunto registrado é
`tensorboard`, `mlflow`, `wandb`, `comet`, `clearml`, `neptune`, `dvclive` e
`dvc`, sendo o último um alias para `dvclive`. Qualquer outra coisa gera erro na
hora e lista os nomes válidos. Não existe um valor que habilite todos, e não
existe flag de CLI: `loggers=` é um argumento Python.

## O que cada backend registra

Todos eles gravam os mesmos nomes de métrica, por isso o dashboard fica igual
qualquer que seja a sua escolha:

| Chave | Valor |
|---|---|
| `train/loss` | a loss média de treinamento da época |
| `train/loss/<component>` | cada componente da loss que a família reporta |
| `lr/<group>` | o learning rate de cada grupo de parâmetros do otimizador |
| `val/<metric>` | cada métrica de validação, com o prefixo `metrics/` removido |
| `time/epoch_seconds` | o tempo de relógio da época |

O passo é a época contada a partir de 1. A configuração de treinamento
totalmente resolvida é registrada como parâmetros no início do treinamento, e o
nome da execução tem como padrão `<family><size>-<task>`, por exemplo
`yolo9s-detect`.

No fim do treinamento, os backends que suportam artefatos enviam `results.csv`,
`train_config.yaml` e `summary.json` quando esses arquivos existem, mais
`weights/best.pt` com `log_checkpoints=True`. O TensorBoard não envia nada,
porque não tem conceito de artefato. Nenhum logger envia as imagens dos gráficos
de validação.

## Comportamento em caso de falha

Um pacote de backend ausente gera erro na construção, indicando o comando de
instalação, porque pedir um logger e silenciosamente não receber nada esconde um
bug.

Uma falha do backend durante a execução faz o oposto. A primeira exceção vinda
de um handler desabilita aquele logger pelo resto da execução, registra o
ocorrido, encerra a execução do backend como falha, e o treinamento continua. Se
o servidor de tracking cair, você não perde o treinamento.

## Os backends

Cada um precisa do seu próprio extra.

| Nome | Extra | Construtor |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`, `dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

Importe as classes de `libreyolo.training`.

Notas específicas de cada backend que vale a pena conhecer antes da primeira
execução:

Os arquivos de evento do TensorBoard vão por padrão para
`<save_dir>/tensorboard`. Visualize com `tensorboard --logdir runs/train`.

O MLflow 3.x descontinuou o file store local `./mlruns` e gera erro a menos que
`MLFLOW_ALLOW_FILE_STORE=true`. Para tracking local sem servidor, passe uma URI
de banco de dados no lugar, como no snippet acima, e leia com
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

O Weights & Biases recorre à variável de ambiente `WANDB_PROJECT` e depois a
`libreyolo`. O Comet recorre a `COMET_PROJECT_NAME` e depois a `libreyolo`, e
pega as credenciais da própria configuração; `online=False` cria um experimento
offline. O ClearML cria uma task nova, reporta a configuração em `TrainConfig` e
desabilita a captura automática de framework para que as métricas não sejam
reportadas duas vezes. O Neptune usa o cliente atual `neptune-scale` em vez do
pacote legado, e `mode="offline"` registra localmente.

O DVCLive grava em `<save_dir>/dvclive`. Ele monta sua árvore de resumo a partir
de `/`, e não consegue guardar um float em um caminho que também seja um nó pai,
então `train/loss/box` é gravado como `train/loss.box` enquanto `train/loss`
mantém o nome. O LibreYOLO também desliga os padrões usuais do DVCLive de salvar um
experimento DVC e gravar um `dvc.yaml` na raiz, de modo que um logger opcional
não cria nenhum estado de controle de versão fora do diretório da execução;
passe `save_dvc_exp=True` ou um `dvcyaml=` explícito para tê-los de volta.

O Neptune é deliberadamente excluído de `libreyolo[all]`: seu cliente estável
exige protobuf abaixo de 7, enquanto o extra do TFLite exige protobuf 7. Instale
`libreyolo[neptune]` em um ambiente sem o extra do TFLite.

## Escrevendo um callback

Os mesmos quatro eventos comandam tudo.

<code-tabs name="callback" />

| Evento | Quando | Carrega |
|---|---|---|
| `TrainStartEvent` | depois do setup, antes da época 1 | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | depois de cada época, treinamento e validação | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `current_metric`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | depois que o treinamento termina | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | se o treinamento gerar uma exceção | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

Um callable simples recebe apenas `TrainEpochEvent`. Um objeto pode implementar
qualquer subconjunto de `on_train_start`, `on_train_epoch_end`, `on_train_end` e
`on_train_exception`; métodos ausentes são ignorados.

`TrainStartEvent.config` é a configuração totalmente resolvida, os kwargs do
usuário mesclados com os padrões da família, como um mapeamento somente leitura.
Os eventos são dataclasses congeladas e seus mapeamentos são somente leitura,
portanto um callback não consegue mudar a execução escrevendo em um deles.

Uma exceção gerada em `on_train_start`, `on_train_epoch_end` ou `on_train_end`
se propaga e encerra a execução. Só `on_train_exception` é protegido, para que
não possa mascarar a falha original.

Em treinamento multi-GPU, os callbacks disparam apenas no rank 0. Com o spawn
automático de DDP eles também precisam ser picklable, o que significa uma classe
ou função de nível de módulo em vez de uma closure ou uma lambda. Veja
[Treinamento multi-GPU](/docs/train/multi-gpu).

## O que toda execução grava de qualquer jeito

Três arquivos aparecem no diretório da execução sem configuração nenhuma, em
toda família:

| Arquivo | Gravado | Conteúdo |
|---|---|---|
| `status.json` | atomicamente, a cada época e no início, no fim e na falha | `state` de `running`, `completed` ou `failed`, `current_epoch`, `total_epochs`, `progress`, `eta_seconds`, as `metrics` mais recentes, `best_metric`, `best_epoch`, e um objeto `error` em caso de falha |
| `metrics.jsonl` | acrescentado uma vez por época | uma linha JSON por época, o mesmo schema de `results.csv` |
| `train.log` | ao vivo | a saída de console da execução |

`status.json` é a leitura barata para um script ou um agente que consulta uma
execução, e a escrita atômica significa que um leitor nunca vê um arquivo
gravado pela metade.

`results.csv` e `summary.json` são separados e dependem da família. Eles são
gravados para YOLOv9, YOLOv9-E2E, YOLOv9-P2, YOLOv7, YOLO-NAS, RF-DETR, EC e
DINOv2, e não para as outras famílias. `results.csv` recebe uma linha por época
com os componentes da loss, as métricas de validação e os learning rates como
colunas, e seu cabeçalho se amplia quando uma coluna nova aparece. Ao retomar um
treinamento, ele é truncado nas linhas anteriores à época retomada, em vez de
duplicá-las.

Ao lado desses, o trainer sempre grava `train_config.yaml` no setup e os
checkpoints em `weights/`.

## Acompanhe uma execução ao vivo

<code-tabs name="monitor" />

`libreyolo monitor` serve um dashboard de navegador sobre os arquivos acima
usando apenas a biblioteca padrão: gráficos de métricas, o tail do log e
quaisquer imagens de validação, atualizando enquanto a execução está ativa. Ele
é somente leitura e nunca toca no processo de treinamento, por isso se conecta a
uma execução ao vivo, reabre uma que terminou ou inspeciona uma que quebrou.

## Relacionados

- [Validação e métricas](/docs/train/validation) para o que as chaves `val/`
  significam e como adicionar uma loss de validação.
- [Desempenho de treinamento](/docs/train/performance) para o profiler, que é
  uma ferramenta diferente com uma pergunta diferente.
