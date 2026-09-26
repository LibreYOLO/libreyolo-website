---
title: Checkpoints upstream
seo_title: Carregando checkpoints upstream no LibreYOLO
description: >-
  Como a conversão automática transforma um checkpoint upstream publicado em um
  checkpoint LibreYOLO v1.0: os layouts que ela desembrulha, o que cada família
  reconhece e onde ela para.
lead: >-
  As famílias do LibreYOLO são portadas de projetos upstream cujos checkpoints
  publicados são quase carregáveis, mas não carregam metadados do LibreYOLO. A
  conversão automática reconhece esses arquivos, envolve-os no esquema v1.0 e
  grava o resultado ao lado do original.
keywords:
  - libreyolo autoconvert
  - carregar checkpoint upstream
  - convert_upstream_state_dict
  - pesos upstream libreyolo
  - converter checkpoint yolo para libreyolo
last_verified: 1.5.0
verification: >-
  Comportamento lido de libreyolo/models/autoconvert.py e
  BaseModel.convert_upstream_state_dict; os reconhecedores por família foram
  conferidos lendo o override de convert_upstream_state_dict de cada família,
  tudo na v1.5.0. Regras COCO do RF-DETR vindas de docs/checkpoint_schema.md.
snippets:
  usage:
    - label: Basta passar o arquivo para a factory
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Um arquivo upstream reconhecido é convertido no carregamento, e o
        # checkpoint convertido é gravado ao lado dele.
        # model = LibreYOLO("yolov9-t-converted.pt")

        # Qualquer checkpoint do LibreYOLO carrega sem alterações.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.family, model.size, model.task, model.nb_classes)
source_hash: c6022771a2a207a1
---

## O que acontece no carregamento

Quando `LibreYOLO()` encontra um arquivo `.pt` que ainda não é um checkpoint
v1.0 completo, ele chama o conversor automático, que:

1. desembrulha o dicionário de tensores dos layouts upstream mais comuns;
2. pergunta a cada família registrada se ela reconhece o layout, remapeando as
   chaves onde a nomenclatura upstream difere do port nativo;
3. envolve o vencedor em um checkpoint de metadados v1.0 estrito, lendo
   tamanho, tarefa e contagem de classes dos próprios tensores, para que
   checkpoints com fine-tuning sejam convertidos corretamente;
4. grava o resultado ao lado do original como
   `<source>-<Prefix><size>[-task].pt` e retorna esse caminho, para que a
   factory o carregue normalmente.

Nada é pedido a quem chama. Um arquivo que nenhuma família reivindica não
retorna nada, e a factory informa que não conseguiu carregá-lo.

<code-tabs name="usage" />

## Layouts que ela desembrulha

O dicionário de tensores é procurado nesta ordem de preferência, EMA primeiro,
e cada candidato é testado até que um de fato contenha tensores. Um bloco EMA
vazio ou só com metadados, portanto, não esconde pesos válidos embaixo dele.

| Chave | Observação |
|---|---|
| `ema.module` | O wrapper EMA mais comum |
| `ema` | Wrappers EMA planos e antigos que guardam os tensores diretamente |
| `ema_state_dict` | Entradas sob um prefixo `module.` têm o prefixo removido |
| `params_ema` | |
| `params` | |
| `ema_net` | |
| `net` | |
| `model` | |
| `state_dict` | |
| O próprio arquivo | Um state dict puro |

Cada candidato é então reduzido às entradas cujo valor é um tensor e
normalizado: um prefixo `module.` ou `_orig_mod.` no início é removido, e um
dicionário cujas chaves começam todas com `model.model.` tem esse prefixo
retirado.

## Quais famílias reconhecem o quê

O reconhecimento é um classmethod por família. A implementação padrão
reivindica um layout cujas chaves já batem com o port nativo. Uma família cuja
nomenclatura de chaves upstream é diferente sobrescreve esse método com um
remapeamento e não retorna nada para os layouts que não reconhece.

Famílias que trazem um reconhecedor com remapeamento: `centernet`,
`deeplabv3`, `deformable_detr`, `dexined`, `moge2`, `picodet`, `rtdetr`,
`rtdetrv2`, `rtdetrv4`, `rtmdet`, `segformer`, `swin`, `teed`, `yolo7`,
`yolo9`, `yolo9_e2e`, `yolo9_p2`.

Famílias que recusam a conversão automática por completo: `efficientdet`,
`eomt` e `pidnet` não retornam nada do reconhecedor, então seus arquivos
upstream passam por um script de conversão. `l2cs` fica de fora do
reconhecedor genérico porque é somente para inferência e tem pesos com
redistribuição restrita.

O RF-DETR mantém o próprio reconhecedor, porque precisa do checkpoint inteiro,
e não só do dicionário de tensores, para detectar o tamanho e remapear as
classes do COCO. Ele só é registrado quando suas dependências opcionais estão
instaladas.

Todas as demais famílias registradas usam o padrão: reivindicam o arquivo
quando o próprio loader já reconhece essas chaves.

## Qual família vence

Várias famílias podem reivindicar o mesmo arquivo, então a resolução espelha
as regras de dispatch da factory.

A reivindicação de uma subclasse vence a da classe base. A ordem de registro
segue a criação das classes, então uma família derivada se registra depois da
base que ela refina, e seus marcadores positivos não podem perder para o
passthrough mais amplo da base.

Depois disso, a ordem do registro decide, porque ela codifica a especificidade:
a reivindicação mais antiga é a correspondência mais específica.

O único empate que a ordem de registro não consegue desfazer é DEIM contra
D-FINE, cujas chaves de arquitetura são idênticas. Ali, e só ali, o nome do
arquivo é o sinal decisivo, e um arquivo cujo nome não dá nenhuma pista é
recusado em vez de adivinhado. O nome do arquivo não é consultado em nenhum
outro lugar, de propósito, para que uma reivindicação falso-positiva ampla
nunca seja promovida acima de uma mais específica só por causa de como o
arquivo se chama.

## Carregamento seguro

Os arquivos upstream são carregados pelo unpickler weights-only. Alguns
checkpoints de treinamento upstream embutem objetos de biblioteca que esse
unpickler rejeita. Esses objetos são metadados de treinamento, não pesos,
então cada global bloqueado é retentado com uma classe substituta inerte que
satisfaz o unpickler sem executar nada. O nome capturado é usado apenas como
um rótulo de texto, nunca importado, avaliado ou chamado.

Nomes de módulos sensíveis são recusados de imediato e nunca recebem um
substituto: `builtins`, `os`, `sys`, `posix`, `nt` e `subprocess`. O laço de
retentativas é limitado a 32 tentativas, então um arquivo construído para
introduzir uma série ilimitada de globais distintos falha de forma fechada em
vez de ficar girando. Só os tensores sobrevivem até o checkpoint convertido.

## Para onde vai o arquivo convertido

A saída é gravada ao lado do original, com o nome
`<source>-<Prefix><size>[-task].pt`. Ela é sempre reescrita em vez de
reaproveitada, o que mantém atualizados os carregamentos repetidos do mesmo
original e evita colisões com os pesos oficiais ou com outro fine-tuning da
mesma família, tamanho e tarefa no mesmo diretório.

Quando o diretório de origem é somente leitura, a conversão recorre a um
diretório temporário privado novo, criado a cada chamada, e a linha de log
informa o caminho usado. Só se isso também falhar a conversão é abandonada,
com um aviso.

## Checkpoints do LibreYOLO já existentes

Um arquivo que carrega um marcador específico do LibreYOLO,
`libreyolo_version` ou `model_family`, pertence ao caminho de carregamento
normal e não é reconvertido. Essa exclusão vale apenas para uma reivindicação
de passthrough, ou seja, aquela em que o conjunto de chaves ficou inalterado.
Uma reivindicação cuja conversão alterou o conjunto de chaves é prova de um
layout upstream estranho e é aceita mesmo em um arquivo marcado.

`schema_version` não é tratado como marcador de propósito, porque outras
ferramentas de treinamento e de exportação usam esse nome genérico, e o mesmo
vale para `names`, `nc`, `size`, `task` e `imgsz`, porque um fine-tuning
upstream também pode carregá-los. Um fine-tuning estranho que apenas carrega
uma chave `names` genérica não fica marcado, portanto, e sua reivindicação com
chaves nativas converte normalmente e deriva a contagem de classes dos
tensores da cabeça, em vez de ser carregado erradamente como se tivesse 80
classes.

## Metadados que o conversor lê

Os nomes das classes vêm de uma chave `names` de primeiro nível, ou de
`class_names` dentro de um bloco `args` ou `hyper_parameters`. Um mapa de
nomes indexado por rótulos em vez de por índice de classe é inutilizável e é
substituído por padrões gerados. Uma lista de nomes maior que a contagem de
classes detectada é cortada, porque índices fora do intervalo reprovariam no
validador estrito e abortariam a conversão em silêncio.

Os `args` upstream são levados adiante como metadados simples, e qualquer
valor que não seja string, número, booleano, lista ou dicionário é
descartado, para que nada inseguro chegue ao arquivo salvo.

## Normalização COCO do RF-DETR

Os checkpoints upstream do RF-DETR expõem uma cabeça de classificação com 91
saídas, que são as 90 classes do COCO mais o fundo. A conversão automática
normaliza um RF-DETR COCO para a convenção COCO-80, com o remapeamento
aplicado no pós-processamento.

Um checkpoint é tratado como COCO quando carrega exatamente 80 nomes, ou
declara uma contagem de 80 classes, ou tem uma dica de dataset `coco`, ou não
tem metadado nenhum de classe ou de dataset. Esse último caso importa: um
state dict upstream puro é o checkpoint canônico pré-treinado no COCO, e é o
único RF-DETR de 91 saídas sem metadados em circulação.

Um RF-DETR customizado de verdade, com 90 classes, é preservado com 90
classes. Ele é identificado por uma lista de nomes, por uma contagem de
classes explícita diferente de 80 ou por uma dica de dataset que não seja
COCO, então o fallback de checkpoint puro não dispara para ele. Placeholders
vazios são ignorados na hora de decidir se existe uma dica de dataset.

## Limites

A conversão automática reconhece os layouts upstream publicados. Ela não
reescreve uma arquitetura e não torna carregável um modelo que não foi
portado. Quando nenhuma família reivindica um arquivo, a resposta é um script
de conversão, não um argumento da factory: o repositório traz
`weights/convert_*.py` para as famílias que precisam de um, incluindo EoMT,
PIDNet e EfficientDet.

A conversão também não inventa metadados que não consegue ler. Tamanho, tarefa
e contagem de classes vêm dos tensores; os nomes vêm do arquivo quando estão
presentes e são gerados como `class_i` quando não estão.
