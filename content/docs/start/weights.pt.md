---
title: Checkpoints e pesos
seo_title: Checkpoints e pesos do LibreYOLO
description: >-
  Como o LibreYOLO encontra, baixa e verifica os pesos dos modelos, onde eles
  ficam hospedados, como rodar sem rede e o que faz um checkpoint carregar com
  segurança.
lead: >-
  Um checkpoint do LibreYOLO é um dicionário torch.save que guarda um state dict
  mais os metadados necessários para identificá-lo. Esta página cobre de onde
  vêm esses arquivos, onde eles vão parar e como são carregados.
keywords:
  - pesos libreyolo
  - checkpoints libreyolo
  - baixar pesos libreyolo
  - libreyolo offline
  - libreyolo hugging face
  - metadados de checkpoint
last_verified: 1.5.0
meta:
  - label: Hospedados em
    value: 'Um repositório do Hugging Face por checkpoint:'
    links:
      - label: huggingface.co/LibreYOLO
        href: 'https://huggingface.co/LibreYOLO'
  - label: Cache local
    value: weights/ no diretório de trabalho
    mono: true
  - label: Esquema de metadados
    value: v1.0
snippets:
  load:
    - label: Download automático
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Um nome de arquivo puro resolve para weights/LibreYOLO9t.pt e é
        # baixado ali se ainda não estiver presente.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model(SAMPLE_IMAGE).boxes)
    - label: Caminho explícito
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Um caminho com componente de diretório é usado exatamente como
        # escrito e nunca é buscado na rede.
        model = LibreYOLO("/opt/models/LibreYOLO9t.pt")
        print(model.family, model.size, model.task)
  inspect:
    - label: CLI
      language: bash
      code: |
        # Lê os metadados sem construir um modelo, e informa se eles
        # cumprem o esquema.
        libreyolo metadata path=weights/LibreYOLO9t.pt
    - label: JSON
      language: bash
      code: |
        libreyolo metadata path=weights/LibreYOLO9t.pt --json
    - label: Python
      language: python
      code: >
        from libreyolo.utils.serialization import (
            load_untrusted_torch_file,
            validate_checkpoint_metadata,
        )


        loaded = load_untrusted_torch_file("weights/LibreYOLO9t.pt")


        # Retorna uma lista de problemas. Vazia significa que o arquivo cumpre a
        v1.0.

        print(validate_checkpoint_metadata(loaded))

        print(loaded["model_family"], loaded["size"], loaded["task"],
        loaded["nc"])
source_hash: 210a12baa1417cfb
---

## Onde um checkpoint é procurado

Uma referência de modelo sem componente de diretório, como `LibreYOLO9t.pt`, é
resolvida em `weights/` relativo ao diretório de trabalho atual. Se
`weights/LibreYOLO9t.pt` existir, ele é usado; se um arquivo com esse nome
existir no próprio diretório de trabalho, ele é usado no lugar; caso contrário,
`weights/LibreYOLO9t.pt` passa a ser o destino do download.

Uma referência que de fato contém um diretório, absoluto ou relativo, é tomada
literalmente. É essa a forma a usar quando os pesos ficam em um lugar central e
nada deve ser buscado na rede.

<code-tabs name="load" />

## Download automático

Quando o caminho resolvido não existe, o LibreYOLO analisa o nome do arquivo
para recuperar a família, o tamanho e a tarefa, e pede à família correspondente
uma URL de download. A maioria das famílias a monta a partir da organização
LibreYOLO no Hugging Face, onde cada checkpoint tem seu próprio repositório com
o nome do arquivo:

```text
https://huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt
```

Um sufixo de variante de dataset continua fazendo parte do nome do repositório,
então um checkpoint treinado em algo diferente do padrão da família resolve para
o próprio repositório em vez de sobrescrever o padrão.

A transferência em si é defensiva, porque um arquivo de pesos truncado falha
mais adiante com um erro pouco útil. Os downloads são transmitidos para um
arquivo `.part` e movidos para o lugar de forma atômica só quando completos, de
modo que um processo interrompido nunca deixa um checkpoint escrito pela metade
no caminho final. Uma transferência interrompida é retomada a partir do seu
offset em bytes usando um validador HTTP, e recomeça do zero se o servidor
indicar que o objeto mudou. As falhas são repetidas três vezes com backoff
exponencial. Processos concorrentes que apontam para o mesmo caminho pegam um
arquivo de lock, então dois treinamentos que começam juntos baixam uma vez só.
Quando uma família busca de um host de terceiros em vez da organização
LibreYOLO, ela pode fixar um checksum e recusar o arquivo se ele não bater.

Se `HF_TOKEN` estiver definido, ou houver um token em cache em
`~/.cache/huggingface/token`, ele é anexado como bearer token. Ele é anexado
apenas a URLs de `huggingface.co`, então uma família que baixa de outro host
nunca o recebe.

Nem toda família faz download automático. Algumas deliberadamente não retornam
URL nenhuma porque os pesos publicados não podem ser redistribuídos, e o erro
então explica o que fornecer no lugar. Outras imprimem um aviso de licença antes
de a transferência começar. Esse aviso é o sinal, em runtime, de que os termos
de um checkpoint são mais restritos que os do código, e vale a pena lê-lo em vez
de passar batido.

## A organização no Hugging Face

Os pesos publicados ficam em
[huggingface.co/LibreYOLO](https://huggingface.co/LibreYOLO), um repositório por
checkpoint. Cada repositório carrega uma licença, e a licença não é uniforme
dentro de uma família: uma família cujo código é MIT pode ter alguns pesos que
não são. O repositório é a fonte autoritativa. A página de cada modelo lista os
checkpoints publicados daquela família e suas licenças nas seções Checkpoints e
Licenciamento.

## Trabalhando offline

Nada na biblioteca exige acesso à rede depois que os arquivos estão locais. Duas
abordagens funcionam:

Pré-popule um diretório `weights/` ao lado de onde quer que o job rode. Baixar
os checkpoints uma vez em uma máquina conectada e depois copiar o diretório já
basta; a etapa de resolução acima os encontra e nunca chega à rede.

Ou passe um caminho absoluto para um local compartilhado. Uma referência com
componente de diretório é usada como está, então um mount somente leitura de
pesos curados é uma configuração válida. Se o processo não puder escrever ao
lado de um checkpoint que precisa converter, a conversão recorre a um diretório
temporário privado em vez de falhar.

Os datasets seguem uma regra separada: eles são resolvidos em `~/datasets`, ou
no diretório indicado por `LIBREYOLO_DATASETS_DIR` quando essa variável está
definida.

## Segurança no carregamento

Checkpoints são pickles, e um pickle pode executar código arbitrário quando é
aberto. O LibreYOLO trata todo arquivo de pesos como não confiável e o carrega
pelo caminho `weights_only=True` do PyTorch, que restringe o unpickler a
tensores e a um pequeno conjunto de tipos seguros. Isso vale para o arquivo que
você passa, não só para os arquivos que o LibreYOLO baixou. Em um build do
PyTorch antigo demais para suportar esse argumento, o carregamento é recusado em
vez de ser feito de forma insegura.

Alguns checkpoints de treinamento upstream embutem objetos que o unpickler
restrito rejeita, como um objeto de configuração do framework em que foram
treinados. Esses objetos são metadados de que o LibreYOLO não precisa, então,
durante a conversão, cada classe bloqueada é substituída por um substituto
inerte que satisfaz o unpickler sem executar nada, e apenas os tensores
sobrevivem no arquivo convertido. Nomes de módulo sensíveis são recusados de
saída em vez de virarem stub, e o laço de retentativas é limitado, de modo que
um arquivo construído para introduzir uma série infinita de classes bloqueadas
falha de forma fechada. Veja [importar pesos existentes](/docs/migrate) para o
resto desse caminho.

## Metadados do checkpoint

Um checkpoint do LibreYOLO é um dicionário cuja chave `model` guarda o state
dict do PyTorch. Nove chaves são exigidas pelo esquema v1.0, e juntas elas
permitem que a factory identifique um arquivo sem analisar o nome dele nem
adivinhar pelos formatos dos tensores.

| Chave | Significado |
|---|---|
| `model` | O state dict do PyTorch |
| `schema_version` | A versão do contrato de metadados. A v1.0 usa a string `1.0` |
| `libreyolo_version` | A versão do LibreYOLO que produziu o arquivo |
| `model_family` | Um identificador de família registrado, como `yolo9` |
| `size` | A variante dentro daquela família, como `t` ou `r18` |
| `task` | Um nome canônico de tarefa |
| `nc` | Uma contagem positiva de classes |
| `names` | Um mapeamento de índice de classe para rótulo, cobrindo `0` até `nc - 1` |
| `imgsz` | Uma resolução de entrada positiva |

Tarefas com estrutura extra a registram junto dessas chaves. Checkpoints de pose
adicionam `num_keypoints` e `keypoint_dim`, e podem adicionar sigmas OKS por
keypoint. Checkpoints de OCR embutem o charset CTC completo para que o arquivo
seja autocontido. Checkpoints de restauração podem registrar o tipo de
degradação e um fator de aumento de escala. Checkpoints do trainer adicionam
estado de retomada, como `epoch`, o estado do otimizador e os pesos da EMA;
pesos de inferência publicados não devem carregar isso.

Um arquivo que cumpre as nove chaves carrega pelo caminho de metadados. Um que
não cumpre é convertido, se alguma família reconhecer seu layout, ou carregado
pelo caminho de compatibilidade com um aviso nomeando o que está faltando.

## Inspecionando um checkpoint

<code-tabs name="inspect" />

`libreyolo metadata` nunca constrói um modelo, então funciona em um arquivo cuja
família não está instalada e em um arquivo sobre o qual você não tem certeza.
