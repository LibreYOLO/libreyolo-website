---
title: Configurações
seo_title: Variáveis de ambiente e diretórios do LibreYOLO
description: >-
  Todas as variáveis de ambiente que o LibreYOLO lê, os diretórios em que ele
  escreve, os tokens de que precisa e as chaves que mudam qual caminho de código
  roda.
lead: >-
  O LibreYOLO não tem arquivo de configuração. O comportamento que não é
  argumento de função é controlado por variáveis de ambiente e por um pequeno
  número de diretórios convencionais, todos listados aqui.
keywords:
  - LIBREYOLO_DATASETS_DIR
  - LIBREYOLO_KERNELS
  - LIBREYOLO_FASTER_COCO_EVAL
  - HF_TOKEN
  - diretório de pesos libreyolo
  - cache do libreyolo
last_verified: 1.5.0
verification: >-
  Variáveis localizadas buscando os.environ e os.getenv em libreyolo/**/*.py na
  v1.5.0; semântica lida em cada ponto de uso. Convenções de diretório lidas em
  libreyolo/data/utils.py, libreyolo/utils/download.py,
  libreyolo/export/exporter.py, libreyolo/models/base/model.py e
  libreyolo/models/sam3dbody/mhr_body.py.
snippets:
  usage:
    - label: Apontar a raiz dos datasets para outro lugar
      language: bash
      code: |
        export LIBREYOLO_DATASETS_DIR=/data/datasets
        python -c "from libreyolo.data import DATASETS_DIR; print(DATASETS_DIR)"
    - label: Ler o valor resolvido no Python
      language: python
      code: >
        from libreyolo.data import DATASETS_DIR


        # O padrão é ~/datasets; LIBREYOLO_DATASETS_DIR sobrescreve no momento
        do import.

        print(DATASETS_DIR)
source_hash: 462f1288582225ce
---

## Variáveis de ambiente

| Variável | Padrão | Efeito |
|---|---|---|
| `LIBREYOLO_DATASETS_DIR` | `~/datasets` | Raiz dos datasets. Lida uma única vez no import, para `libreyolo.data.DATASETS_DIR` |
| `LIBREYOLO_FASTER_COCO_EVAL` | não definida | Sobrescreve a flag de validação `faster_coco_eval`. `1`, `true`, `yes` ou `on` força o backend mais rápido a ficar ligado, qualquer outro valor o desliga, e sem definição vale a flag da configuração |
| `LIBREYOLO_KERNELS` | não definida | Seleção de kernels. `off` ou `reference` força as implementações de referência; qualquer outro valor seleciona apenas as implementações registradas sob esse nome |
| `LIBREYOLO_QUANT_KERNELS` | não definida | Alias legado de `LIBREYOLO_KERNELS`, lido apenas quando esta não está definida |
| `LIBREYOLO_HUB_KERNELS` | não definida | `0`, `false`, `off` ou `no` desativa o carregamento de kernels do Hugging Face Hub. Qualquer outro valor, incluindo sem definição, deixa o recurso ativo |
| `LIBREYOLO_MHR_PATH` | `~/.cache/libreyolo/mhr/mhr_model.pt` | Local do modelo corporal MHR usado pela tarefa `mesh` |
| `LIBRELABEL_ENABLE_LOCATE` | não definida | Precisa ser exatamente `1`, `true`, `yes` ou `on` para expor o assistente LocateAnything na ferramenta de anotação. Qualquer outro valor o mantém desligado |
| `SAM_3D_BODY_PATH` | não definida | Caminho para o pacote SAM 3D Body da família mesh, quando ele não é passado ao construtor |
| `HF_TOKEN` | não definida | Token de acesso do Hugging Face, usado para repositórios restritos |

<code-tabs name="usage" />

`LIBREYOLO_DATASETS_DIR` é lida no momento do import, então defini-la depois de
importar `libreyolo.data` não tem efeito sobre `DATASETS_DIR`.

Os kernels do Hub são um opt-in em duas partes. A busca em runtime só acontece
quando o pacote opcional `kernels` está instalado, então instalar
`libreyolo[hub-kernels]` é o opt-in e `LIBREYOLO_HUB_KERNELS=0` é o opt-out.
Uma instalação sem o extra não é afetada de um jeito nem de outro.

A seleção de kernels também encurta os imports: quando `LIBREYOLO_KERNELS`
força `off` ou `reference`, os provedores acelerados internos nunca chegam a ser
importados. O registro que essas três variáveis controlam está documentado em
[kernels](/docs/reference/kernels).

## Variáveis que a biblioteca define

Estas são escritas, e não lidas, então defini-las na mão não é o caminho
suportado.

| Variável | Definida por |
|---|---|
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | O helper de spawn do DDP, um valor por processo worker |
| `CUDA_VISIBLE_DEVICES` | Temporariamente restringida durante a configuração distribuída, e depois restaurada |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Definida como `1` pelos treinadores EC, com `setdefault`, então um valor existente prevalece |
| `MOMENTUM_ENABLED` | Definida com `setdefault` pelo carregador da família mesh |

`LOCAL_RANK` também funciona como sinal de modo distribuído: é pela presença
dela no ambiente que o código de treinamento detecta que está rodando sob DDP.

## Variáveis dos loggers

Os loggers opcionais de treinamento recorrem a padrões do ambiente para o nome
do projeto.

| Variável | Padrão | Usada por |
|---|---|---|
| `WANDB_PROJECT` | `libreyolo` | O logger do Weights and Biases, quando nenhum projeto é passado |
| `COMET_PROJECT_NAME` | `libreyolo` | O logger do Comet, quando nenhum projeto é passado |

A autenticação nesses serviços segue as ferramentas deles, não a do LibreYOLO.

## Tokens

`HF_TOKEN` é o token de acesso do Hugging Face. Quando ele não está definido, o
token é lido de `~/.cache/huggingface/token`, que é onde um login pela CLI do
Hugging Face o escreve. Os dois caminhos funcionam.

Um token só é necessário para repositórios restritos. O SAM 3 é o exemplo que
vem na biblioteca: os pesos dele baixam de um repositório restrito sob uma
licença própria, então os termos precisam ser aceitos na página do repositório e
a sessão precisa estar autenticada.

## Diretórios

| Caminho | Conteúdo |
|---|---|
| `weights/` | Checkpoints baixados, snapshots baixados do Hugging Face e artefatos exportados |
| `~/datasets` | Raiz dos datasets, a menos que `LIBREYOLO_DATASETS_DIR` diga outra coisa |
| `~/.cache/huggingface/token` | Token do Hugging Face, quando não está em `HF_TOKEN` |
| `~/.cache/libreyolo/mhr/mhr_model.pt` | Modelo corporal MHR, a menos que `LIBREYOLO_MHR_PATH` diga outra coisa |
| `runs/track/` | Saída padrão de `model.track(save=True)` |

`weights/` é relativo ao diretório de trabalho. Um nome de arquivo simples é
resolvido através dele, então `LibreYOLO("LibreYOLO9t.pt")` procura por
`weights/LibreYOLO9t.pt` e baixa ali quando o arquivo não existe.
`model.export()` escreve no mesmo diretório quando `output_path` não é
informado. Os tiers irmãos baixam snapshots de vários arquivos em
`weights/<Prefix><size>/`.

## Comportamento de download

Os downloads de pesos são repetidos três vezes com backoff, retomam de um
arquivo parcial e são protegidos por um arquivo de lock para que dois processos
não busquem o mesmo checkpoint ao mesmo tempo. Uma família que busca de um host
de terceiros pode fixar um checksum e falhar de imediato em caso de divergência.

Alguns downloads imprimem um aviso de licença antes de começar. Esses avisos
fazem parte do caminho de download e não podem ser suprimidos por configuração.

## Backend de validação

`model.val()` aceita `faster_coco_eval=True` por padrão e recorre ao
pycocotools quando o pacote não está instalado, avisando uma vez. Definir
`LIBREYOLO_FASTER_COCO_EVAL` sobrescreve a flag por chamada, que é o que um
harness de benchmark sem acesso às configurações por execução deve usar. O
backend que de fato rodou é informado em `model.last_eval_backend`.

## Scripts de download de dataset

Um YAML de dataset pode carregar um campo `download` contendo Python. Ele não é
executado a menos que `allow_download_scripts=True` seja passado à chamada que o
lê, o que é um argumento de função em `val()` e `export()`, e não uma variável
de ambiente.
