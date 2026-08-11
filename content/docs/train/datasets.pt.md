---
title: Datasets
seo_title: Datasets de treinamento no LibreYOLO
description: >-
  O YAML de dataset que o LibreYOLO lê, o layout de pastas que ele espera, como
  funciona o download automático e o comando doctor que confere um dataset antes
  do treinamento.
lead: >-
  Um dataset do LibreYOLO é um arquivo YAML que nomeia uma raiz, seus splits e
  seus nomes de classes. Todo o resto, inclusive onde ficam os arquivos de
  rótulos, é derivado desse arquivo por convenção.
keywords:
  - formato de dataset yolo
  - data.yaml
  - treinar yolo com dataset próprio
  - formato de label yolo
  - dataset coco json
  - download automático de dataset
  - libreyolo doctor
  - verificar desbalanceamento de classes
  - vazamento entre train e val
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Um nome incluído no pacote, um caminho relativo ou um caminho absoluto
        funcionam.

        model.train(data="coco8.yaml", epochs=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10
  doctor:
    - label: Conferir um dataset
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml
    - label: Falhar um job de CI também com avisos
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml strict=true json=true
    - label: Pular a passada de decodificação das imagens
      language: bash
      code: >
        # Lê apenas os rótulos e o YAML. As verificações de corrupção,
        duplicatas

        # e vazamento entre splits precisam dos pixels, então são puladas.

        libreyolo doctor my-dataset.yaml fast=true
    - label: Python
      language: python
      code: |
        from libreyolo import doctor

        report = doctor.diagnose("my-dataset.yaml", imgsz=640)

        for finding in report.findings:
            print(finding.severity.value, finding.check_id, finding.message)

        raise SystemExit(report.exit_code(strict=False))
source_hash: 9a12a0551c8b56e9
---

## Aponte o train para um dataset

`data=` aceita um caminho de YAML ou o nome de um config que vem junto com o
pacote.

<code-tabs name="train" />

O nome é resolvido em uma ordem fixa: um caminho absoluto que exista, depois o
nome como foi dado relativo ao diretório de trabalho, depois o mesmo nome com
`.yaml` no fim, depois o diretório de configs incluídos no pacote. Quando nada
corresponde, o erro nomeia cada diretório que foi procurado e lista os configs
incluídos.

## Configs incluídos

Treze configs de dataset vêm dentro do pacote, em
`libreyolo/config/datasets/`.

| Config | Tarefa | Observações |
|---|---|---|
| `coco8.yaml` | detect | 8 imagens, baixa de uma URL simples |
| `coco128.yaml` | detect | 128 imagens |
| `coco1000.yaml` | detect | 800 de train, 200 de val |
| `coco5000.yaml` | detect | 4000 de train, 1000 de val |
| `coco.yaml` | detect | COCO 2017 completo |
| `coco-val-only.yaml` | detect | só o val2017 |
| `coco8-pose.yaml` | pose | 8 imagens, keypoints do COCO-17 |
| `coco-pose.yaml` | pose | keypoints do COCO 2017 |
| `ade20k.yaml` | semantic | 150 classes |
| `cityscapes.yaml` | semantic | 19 classes, download manual |
| `cocostuff.yaml` | semantic | 182 classes, download manual |
| `gopro.yaml` | restore | pares de remoção de desfoque |
| `sr8.yaml` | restore | pares de super-resolução |

Só o `coco8.yaml` e o `coco128.yaml` trazem uma URL de download simples. O resto
ou traz um bloco de download em Python, que precisa do opt-in descrito abaixo, ou
espera que os dados já estejam no disco.

## Onde um dataset fica no disco

A chave `path` do YAML nomeia a raiz do dataset. Um `path` absoluto é usado como
está escrito. Um relativo é procurado primeiro dentro do diretório de datasets,
depois ao lado do próprio arquivo YAML, e um dataset que está prestes a ser
baixado vai para o diretório de datasets.

Esse diretório é `~/datasets`, sobrescrito pela variável de ambiente
`LIBREYOLO_DATASETS_DIR`. Não existe arquivo de configurações para isso.

## As chaves do YAML

```yaml
path: my-dataset        # raiz do dataset
train: images/train     # obrigatório para treinar
val: images/val         # obrigatório para validar
test: images/test       # opcional
nc: 3                   # opcional; precisa bater com names
names:
  0: person
  1: helmet
  2: vest
download: https://example.com/my-dataset.zip   # opcional
```

`train`, `val` e `test` aceitam cada um um diretório de imagens, um arquivo
`.txt` listando um caminho de imagem por linha, ou uma lista misturando os dois.
As linhas de uma lista `.txt` podem ser relativas, e nesse caso são resolvidas
contra o diretório do próprio arquivo de lista, e linhas que começam com `#` são
puladas.

`names` pode ser uma lista ou um mapeamento com chaves inteiras. `nc` é
opcional; quando os dois estão presentes e discordam, o doctor reporta isso como
erro.

## Layout de diretórios e arquivos de rótulos

Detecção, segmentação, pose e caixas orientadas compartilham o mesmo layout. O
caminho do rótulo é derivado do caminho da imagem reescrevendo um componente de
diretório `images` para `labels` e trocando a extensão para `.txt`:

```text
my-dataset/
  images/train/0001.jpg   ->   labels/train/0001.txt
  images/val/0002.jpg     ->   labels/val/0002.txt
```

Só um componente de caminho `images` inteiro é reescrito, então um diretório
chamado `images_old` fica intacto.

Uma linha de detecção tem cinco campos, todos normalizados para `[0, 1]` em
relação à largura e à altura originais da imagem:

```text
<class_id> <cx> <cy> <w> <h>
```

Um arquivo de rótulos ausente ou vazio significa que a imagem não tem objetos, e
ela treina como background em vez de levantar erro. Uma linha com mais de cinco
campos é lida como um polígono e seu bounding box vira a extensão do polígono,
então uma exportação de segmentação usada para treinamento de detecção carrega
sem reclamar. O doctor informa quantas linhas seguiram esse caminho.

## Outras tarefas

A segmentação mantém o mesmo layout com linhas de polígono,
`<class_id> <x1> <y1> ... <xN> <yN>`, com pelo menos três pontos. Uma linha de
detecção de cinco campos é aceita e significa uma instância retangular.

A pose acrescenta `kpt_shape: [K, D]` e uma permutação opcional `flip_idx` ao
YAML. Cada linha tem exatamente `5 + K * D` campos: o box, depois `K` keypoints
em `x y` ou `x y v`, com visibilidade `0`, `1` ou `2`.

As caixas orientadas usam exatamente nove campos, a classe seguida de quatro
pontos de canto em coordenadas normalizadas. Nenhum ângulo é armazenado no
arquivo.

A segmentação semântica pareia cada imagem com uma máscara de canal único na
mesma resolução, resolvida substituindo `images` por `masks_dir` (padrão
`masks`). O valor de pixel `255` significa ignorar. `label_mapping` remapeia os
ids de origem para ids de treinamento no momento do carregamento.

A classificação usa uma árvore ImageFolder em vez de arquivos de rótulos, com
`train/` e `val/` contendo cada um um diretório por classe. O mapeamento de
classe para índice é a ordem dos nomes de pasta ordenados.

A restauração pareia uma entrada degradada com um alvo limpo de resolução
idêntica através de `input_dir` e `target_dir`. Profundidade, normais de
superfície e bordas pareiam cada uma uma imagem com um mapa denso através da sua
própria chave de diretório.

O contrato completo por tarefa, incluindo as convenções de escala de
profundidade e a codificação PNG de segment-id panóptico, é o
`docs/dataset_schema.md` no repositório da biblioteca.

## COCO JSON nativo

Um arquivo de anotações COCO JSON pode ser usado diretamente. Adicione um
mapeamento `annotations`, e o caminho do split vira a raiz das imagens:

```yaml
path: my-dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Quando `names` está presente, os nomes de categoria do JSON precisam bater com
ele, e `names` define os ids de rótulo que o modelo prevê. Sem `names`, os ids de
categoria do COCO são ordenados e mapeados de forma densa para `0..N-1`.

Esse caminho espera um diretório de imagens por split. Uma lista de caminhos ou
uma lista de imagens em `.txt` levanta um erro em vez de carregar silenciosamente
um conjunto diferente.

## Download automático

Um dataset conta como presente quando o caminho de `train` ou de `val` resolve
para um diretório não vazio ou um arquivo existente. Quando não resolve, e o YAML
tem uma chave `download`, o valor decide o que acontece em seguida.

Uma URL `http` ou `https` é baixada e, se for um zip, extraída na raiz do
dataset. Qualquer outra coisa é tratada como um script Python embutido e só roda
quando `allow_download_scripts=True`. Sem isso, o script é pulado com um aviso e
o treinamento continua com o que estiver no disco.

```bash
libreyolo train model=LibreYOLO9s.pt data=coco.yaml allow_download_scripts=true
```

A flag é um gate de execução de código, não um gate de rede. Os downloads por URL
acontecem de qualquer jeito; quem precisa dela são os blocos `download: |`. A CLI
imprime um aviso quando a flag está ligada, e o doctor nunca a habilita.

## Confira o dataset antes de treinar

`libreyolo doctor` lê um dataset de detecção e informa o que daria errado antes
de envolver uma GPU. Ele sai com 1 quando encontra erros, então funciona como um
gate de CI.

<code-tabs name="doctor" />

As verificações vêm em seis famílias:

| Família | Procura por |
|---|---|
| `config` | `names` faltando, `nc` que discorda de `names`, splits faltando ou vazios, nomes de classe duplicados |
| `files` | imagens sem arquivo de rótulos, rótulos sem imagem, imagens faltando listadas em um split, colisões de nome-base |
| `labels` | linhas malformadas, ids de classe fora de `[0, nc)`, coordenadas fora de `[0, 1]`, boxes de área zero, boxes minúsculos ou enormes, boxes duplicados, arquivos de rótulos idênticos byte a byte |
| `balance` | classes com zero ou poucas instâncias, razão de desbalanceamento entre classes, classes presentes em um só split, proporção de imagens de background |
| `images` | arquivos que não decodificam, rotação EXIF, layouts de canais estranhos, imagens uniformes, duplicatas exatas e quase exatas |
| `splits` | a mesma imagem aparecendo em dois splits, de forma exata ou quase idêntica |

`--only` e `--skip` aceitam um id de verificação ou um prefixo de família, então
`skip=images,labels.tiny_object` é válido. `--fast` descarta toda verificação que
precisa decodificar pixels, que são as famílias `images` e `splits`.

Vale conhecer dois comportamentos. `--strict` faz com que os avisos também façam
o código de saída falhar, além dos erros. E o doctor cobre apenas datasets de
detecção: um dataset de pose, segmentação ou caixas orientadas é rejeitado com
uma mensagem nomeando o que ele detectou, em vez de ser conferido contra o
contrato errado.

## Relacionados

- [Hiperparâmetros](/docs/train/hyperparameters) para os argumentos que `train()`
  aceita quando os dados já estão no lugar.
- [Validação e métricas](/docs/train/validation) para avaliar no split `val` ou
  `test`.
