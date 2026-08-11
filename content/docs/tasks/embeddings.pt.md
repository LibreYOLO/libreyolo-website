---
title: Embeddings
seo_title: Embeddings de imagem e de região no LibreYOLO
description: >-
  A tarefa embed retorna vetores float32 normalizados em L2 para uma imagem
  inteira, para cada região detectada ou para texto. Cadastre uma galeria, faça
  a correspondência por similaridade de cosseno e busque pelo Python ou pela
  CLI.
lead: >-
  Uma única tarefa cobre todo vetor que o LibreYOLO produz. embed retorna linhas
  float32 de comprimento unitário cujo produto escalar é uma pontuação de
  similaridade, não importa se a linha descreve uma imagem inteira, um único
  rosto detectado ou uma linha de texto, e a mesma Gallery faz a correspondência
  de todas elas.
keywords:
  - embeddings de imagem python
  - embedding normalizado l2
  - busca por similaridade de cosseno
  - libreyolo embed
  - recuperação de imagens
  - cadastrar galeria de rostos
  - clip embeddings
  - dinov2 embeddings
  - reid embeddings
last_verified: 1.5.0
verification: >-
  Chave da tarefa e aliases lidos de libreyolo/tasks.py. Payloads de resultado
  das classes Embeddings e Identities em libreyolo/utils/results.py. API da
  Gallery de libreyolo/utils/gallery.py. embed e _postprocess_embeddings de
  libreyolo/models/base/model.py. Famílias suportadas localizadas buscando embed
  em SUPPORTED_TASKS dentro de libreyolo/models/**/model.py. Superfície da CLI
  de libreyolo/cli/__init__.py, libreyolo/cli/commands/special.py e
  libreyolo/cli/commands/predict.py. Intenção de projeto de
  docs/adr/0015-embed-generalization.md.
meta:
  - label: Chave da tarefa
    value: embed
    mono: true
  - label: Aliases
    value: 'face-recognition, reid, face'
    mono: true
  - label: Payloads de resultado
    value: 'Embeddings, Identities'
    mono: true
  - label: dtype das linhas
    value: 'float32, comprimento unitário'
snippets:
  predict:
    - label: Imagem inteira
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # O padrão do CLIP é classify, então peça o vetor explicitamente.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)  # (1, 512), uma linha por imagem
        print(result.boxes)                  # None: nada foi localizado
    - label: Por região
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        # A linha i descreve a região da caixa i.
        print(result.boxes.xyxy.shape)       # (N, 4)
        print(result.embeddings.data.shape)  # (N, 512)
    - label: Várias imagens de uma vez
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Todas as linhas de todos os resultados, concatenadas em um tensor.
        vectors = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(vectors.shape)  # (3, 384)
    - label: Texto
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        # Texto é um método, nunca uma fonte de predição. Uma string passada
        # para model(...) continua sendo um caminho ou uma URL.
        text = model.embed_text(["a photo of a cat", "a photo of a dog"])
        print(text.shape)  # (2, 512)
  similarity:
    - label: Comparar dois conjuntos de linhas
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        query = model.embed("query.jpg")          # (1, 512)
        pool = model.embed(["a.jpg", "b.jpg"])    # (2, 512)

        # As linhas têm comprimento unitário, então a similaridade de cosseno
        # é um produto escalar.
        scores = model("query.jpg").embeddings.similarity(pool)
        print(scores.shape)  # (1, 2)
    - label: Imagem contra texto
      language: python
      code: |
        import torch

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image = model.embed("photo.jpg")                       # (1, 512)
        text = model.embed_text(["a cat", "a dog", "a car"])   # (3, 512)

        print(torch.matmul(image, text.T))
  gallery:
    - label: Cadastrar e identificar
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("refs.npz")

        result = model("group.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # name é None abaixo do limiar
    - label: Busca top-k
      language: python
      code: |
        from libreyolo import Gallery
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        gallery = Gallery.load("refs.npz", model=model)

        result = model("query.jpg")
        matches = gallery.match(result.embeddings, top_k=5, threshold=0.4)
        print(matches[0])   # [(name, score), ...] da primeira linha
    - label: Cadastrar um vetor que você já tem
      language: python
      code: |
        from libreyolo import Gallery

        gallery = Gallery()
        gallery.enroll_embedding("ada", vector)  # normalizado na entrada
        print(gallery.identities, gallery.dim, len(gallery))
  cli:
    - label: Cadastrar uma árvore de pastas
      language: bash
      code: >
        # source/<identity>/*.jpg. Uma galeria existente é estendida no lugar.

        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=refs.npz
    - label: Identificar durante a predição
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=group.jpg \
          gallery=refs.npz gallery_threshold=0.45
    - label: Comparar duas imagens
      language: bash
      code: >
        libreyolo compare model=librefacerec-l.onnx \
          source=a.jpg source2=b.jpg threshold=0.4

        # verify é o mesmo comando com um segundo nome.

        libreyolo verify model=librefacerec-l.onnx source=a.jpg source2=b.jpg
        --json
source_hash: ffbaad5599035bc7
---

## Definição

`embed` transforma uma imagem, uma região de uma imagem ou uma string em uma
linha float32 de largura fixa cujo comprimento é um. Como toda linha é um vetor
unitário, comparar duas delas é um produto escalar, e comparar dois conjuntos
delas é uma única multiplicação de matrizes. Nada mais na tarefa é específico do
modelo: recuperação, detecção de duplicatas, reidentificação e reconhecimento
facial são todos a mesma aritmética sobre linhas diferentes.

O vetor é a saída. Não existe lista de classes, então um nome é anexado depois,
comparando com as referências que você fornece, e não por algo que a rede tenha
sido treinada para prever.

### Três formatos

| Formato | `Results.embeddings` | `Results.boxes` | Produzido por |
|---|---|---|---|
| Imagem inteira | `(1, D)` | `None` | Passar uma imagem para uma família de imagem inteira |
| Região | `(N, D)` | `(N, 4)`, alinhada por linha | Famílias que localizam primeiro, como o reconhecimento facial |
| Texto | nem chega a ser um `Results` | | `model.embed_text(texts)`, retornando `(M, D)` |

Um resultado de imagem inteira continua bidimensional mesmo para uma única
imagem. `(D,)` não é um formato de retorno permitido, então quem consome nunca
precisa tratar o caso de linha única como exceção. Texto retorna um tensor
simples em vez de um `Results`, porque uma string não é uma fonte de imagem:
passar uma para `model(...)` ainda significa um caminho ou uma URL, e a
biblioteca nunca adivinha que uma string é prosa.

A chave canônica da tarefa é `embed`. `embedding`, `embeddings`,
`face-recognition`, `facial-recognition`, `recognition`, `face`, `faceid` e
`reid` normalizam todos para ela, então `task="reid"` e `task="embed"`
selecionam exatamente a mesma coisa.

## Modelos

Quatro famílias atendem à tarefa, e elas se dividem com clareza entre as que
localizam algo primeiro e as que não localizam.

| Família | Formato | Dimensão | Também suporta |
|---|---|---|---|
| [LibreFaceRec](/docs/models/librefacerec) | Região, uma linha por rosto detectado | 512 | Nada; `embed` é sua única tarefa |
| [CLIP](/docs/models/clip) | Imagem inteira, com uma torre de texto pareada | 512 para `b32` e `b16`, 768 para `l14` | `classify`, que continua sendo seu padrão |
| [SigLIP 2](/docs/models/siglip2) | Imagem inteira, com uma torre de texto pareada | 768 para `b16`, 1152 para `so400m` | `classify`, que continua sendo seu padrão |
| [DINOv2](/docs/models/dinov2) | Imagem inteira, apenas imagem | 384 | `semantic`, `classify` |

CLIP e SigLIP 2 mantêm `classify` como tarefa padrão, então `task="embed"`
precisa ser pedido. O checkpoint `-cls` que já existe é o artefato de duas
torres compartilhado; nenhum checkpoint `-embed` duplicado é publicado para
pesos idênticos.

`embed_text` existe apenas no CLIP e no SigLIP 2, as duas famílias com torre de
texto. O DINOv2 não tem nenhuma. O embedding do DINOv2 contorna as cabeças
semântica e de classificação e lê o token CLS normalizado final em 224 pixels;
as variantes `n`, `s`, `m` e `l` compartilham todas o encoder DINOv2-S, então
todas as quatro retornam `D = 384`.

Os backbones somente de classificação adicionados nesta versão,
[ViT](/docs/models/vit), [Swin](/docs/models/swin) e [DeiT](/docs/models/deit),
declaram apenas `classify` e não atendem a esta tarefa.

<code-tabs name="predict" />

`model.embed(source, **kwargs)` é o atalho em batch: ele roda `predict` e
concatena todas as linhas de todos os resultados em um único tensor float32 de
CPU `(N_total, D)`, levantando erro se as linhas tiverem dimensões diferentes.
Uma família sem `embed` entre suas tarefas suportadas levanta
`NotImplementedError`.

## Payloads de resultado

`result.embeddings` é um payload `Embeddings`. Seu `data` é sempre `(N, D)`
float32, já normalizado em L2 pelo caminho de inferência, e uma entrada não
bidimensional levanta erro em vez de ser remodelada silenciosamente.

| Membro | Significado |
|---|---|
| `.data` | A matriz `(N, D)` |
| `.dim` | `D` |
| `.normalized` | As mesmas linhas, renormalizadas por precaução |
| `.similarity(other)` | `(N, M)` contra outro conjunto, ou `(N,)` contra um único vetor `(D,)` |
| `.verify(i, j, threshold=0.4)` | Se as linhas `i` e `j` são o mesmo sujeito |

`result.identities` é um payload `Identities`, presente apenas quando uma
galeria foi passada. É um contêiner simples, não um tensor, então mover um
`Results` entre dispositivos não o afeta.

| Membro | Significado |
|---|---|
| `.name` | Lista de nomes, `None` onde nada superou o limiar |
| `.score` | Melhor pontuação de cosseno float32 `(N,)`, mantida mesmo quando o nome é `None` |
| `.data` | Lista de tuplas `(name, score)` |

<code-tabs name="similarity" />

Os vetores ficam de fora de `summary()` e `to_json()` por padrão, já que uma
linha de 512 floats ocupa cerca de dois kilobytes por sujeito. Cada linha
reporta `embedding_dim` em vez deles, além de `identity` e `identity_score`
quando uma galeria foi usada. Passe `summary(embeddings=True)` para incluir os
números.

## Galerias

Uma `Gallery` é um conjunto nomeado de linhas de referência. Ela armazena cada
referência separadamente em vez de fazer a média delas, então um nome é
pontuado apenas pela sua melhor referência correspondente, e adicionar uma foto
ruim não desloca o centroide de uma identidade.

<code-tabs name="gallery" />

`Gallery(model)` se liga aos pesos que produzirão seus vetores.
`enroll(name, sources, select="best")` roda a predição em cada fonte e mantém a
linha de maior confiança por resultado; `select="all"` mantém todas as linhas,
que é o que você quer quando uma imagem de referência de fato contém vários
sujeitos. `enroll_embedding(name, vector)` pula a inferência e recebe um
vetor diretamente, normalizando-o e rejeitando uma linha toda zerada.

`FaceGallery` é um alias permanente da mesma classe, e os arquivos gravados por
versões anteriores, só de rostos, continuam sendo carregados normalmente.

### Correspondência e limiares

A correspondência é uma multiplicação densa de matrizes contra todas as
referências armazenadas, reduzida a uma pontuação por nome pegando o máximo.
Não há índice aproximado, o que mantém os números exatos e impõe um teto
prático ao tamanho da galeria.

Dois pontos de entrada diferem no que fazem abaixo do limiar. `match()` retorna
`[(name, score), ...]` por linha, descartando tudo abaixo do limiar, então uma
linha sem correspondência é uma lista vazia. `identify()` retorna um payload
`Identities` que sempre mantém a melhor pontuação e define o nome como `None`
quando ela está abaixo do limiar. Nenhum dos dois jamais substitui pelo nome
mais próximo abaixo do limiar.

O limiar padrão é `0.4` em todos os lugares. É um valor de cosseno, não uma
probabilidade, e o ponto de operação certo é uma propriedade dos seus dados e
da sua tolerância a correspondências falsas, então faça uma varredura dele em
pares rotulados em vez de aceitar o padrão. `libreyolo enroll` e o argumento de
predição `gallery=` usam o mesmo número.

### Persistência

`save(path)` grava um `.npz` comprimido contendo os vetores, os nomes e um
bloco de metadados que carrega a versão do formato, a dimensão do embedding e
uma impressão digital dos pesos que produziram as linhas. `Gallery.load(path,
model=...)` confere os dois antes de comparar qualquer coisa, então apontar uma
galeria para um modelo diferente levanta erro em vez de pontuar silenciosamente
vetores de dois espaços não relacionados um contra o outro. Salvar uma galeria
vazia é recusado.

## Linha de comando

| Comando | Finalidade |
|---|---|
| `libreyolo enroll` | Percorre uma árvore de uma pasta por identidade e escreve ou estende uma galeria `.npz` |
| `libreyolo compare` | Faz o embedding do sujeito principal em duas imagens e reporta a similaridade de cosseno |
| `libreyolo verify` | O mesmo comando com um segundo nome |
| `libreyolo predict gallery=...` | Anexa identidades a uma execução de predição comum |

<code-tabs name="cli" />

Todo comando do LibreYOLO aceita tanto `key=value` quanto `--key value`, então
`gallery=refs.npz` e `--gallery refs.npz` são o mesmo argumento.

`enroll` recebe `model`, `source` e `gallery`, além dos opcionais
`face-detector`, `device`, `--json` e `--quiet`. Ele lê uma pasta por
identidade, onde o nome da pasta é a identidade e cada imagem dentro dela
contribui com referências:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

Uma imagem que não produz nada é pulada com uma linha no stderr em vez de
abortar a execução, e o resumo reporta quantas referências foram armazenadas
para cada nome. Um arquivo de galeria existente é estendido no lugar, então
identidades podem ser adicionadas ao longo do tempo.

`compare` e `verify` são uma função registrada duas vezes. Eles recebem
`model`, `source`, `source2` e um `threshold` opcional, e imprimem a
similaridade de cosseno, o veredito de igual-ou-diferente e o limiar que o
produziu. `--json` imprime os mesmos três campos como um objeto.

No `predict`, `gallery` aponta para um `.npz` salvo e `gallery_threshold`
sobrescreve o padrão de `0.4`. Passar uma galeria para um modelo cuja tarefa
não é `embed` é um erro, e não um no-op silencioso, e um arquivo de galeria
ausente sugere o comando `libreyolo enroll` que a criaria.

## Rostos

O reconhecimento facial é o formato de região desta tarefa, e é a única
implementação desse formato que acompanha a biblioteca. Ele adiciona um estágio
de detecção e alinhamento antes da cabeça de embedding, além de um método
`verify()`, um argumento para trazer suas próprias caixas, números de acurácia
publicados e orientações de calibração para o limiar. Tudo isso está em
[reconhecimento facial](/docs/tasks/face-recognition), que é o passo a passo a
seguir quando o assunto são rostos. Tudo nesta página se aplica a ele sem
mudanças.

## Treinar, validar e exportar

Nada nesta tarefa treina dentro do LibreYOLO. A cabeça de embedding facial é um
artefato ONNX cujos `train()`, `val()` e `export()` levantam erro; treine uma
cabeça fora e carregue o arquivo pelo caminho. CLIP, SigLIP 2 e DINOv2 treinam
e exportam pelas suas tarefas de classificação e segmentação, não por `embed`.

Não existe validador de recuperação. Meça a acurácia de verificação em pares
rotulados fazendo uma varredura de `threshold`, e a acurácia de identificação
cadastrando uma galeria e lendo `identities.name` e `identities.score` em
imagens reservadas para teste, contando um nome `None` como rejeição.
