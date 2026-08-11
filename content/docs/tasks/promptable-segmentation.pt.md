---
title: Segmentação por prompt
seo_title: Segmentação por prompt no LibreYOLO
description: >-
  Transforme um ponto, um box ou um conceito em texto na máscara de um objeto no
  LibreYOLO. Carregue SAM, SAM 2, SAM 3, EdgeTAM, MobileSAM ou PicoSAM3 pelo
  LibreSAM.
lead: >-
  A segmentação por prompt transforma um clique em uma máscara: você aponta para
  um objeto, ou desenha um box em volta dele, e o modelo retorna o contorno. No
  LibreYOLO isso não é uma chave de tarefa separada, e sim um nível de modelos,
  carregado pela factory LibreSAM, cujos resultados são Results de segmentação
  comuns.
keywords:
  - segmentação por prompt
  - segment anything python
  - segmentação interativa python
  - sam python máscara
  - segmentar objeto com clique
  - prompt de box
  - sam2 python
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompts de ponto e de box
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Um ponto é [x, y] em pixels; as labels são 1 positivo, 0 negativo.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # polígonos
        print(result.boxes.xyxy)    # boxes justos derivados das máscaras

        # Um prompt de box dá uma máscara por box.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: 'Codificar uma vez, usar vários prompts'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # set_image roda o pesado encoder de imagem uma vez e o guarda em cache.
        model.set_image(SAMPLE_IMAGE)
        first = model.predict(points=[640, 420], labels=[1])
        second = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
    - label: Segmentar tudo
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # Sem prompt, é usada uma grade de pontos sobre a imagem inteira. A

        # grade padrão de 32 por lado dá ~1024 passagens do decoder, lenta na
        CPU.

        result = model.predict(SAMPLE_IMAGE, points_per_side=8)

        print(len(result.masks))
    - label: Máscaras de ambiguidade
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Um ponto pode ser uma manga, uma camisa ou uma pessoa. multimask=True
        # retorna as três máscaras de todo-versus-parte em vez de só a melhor.
        result = model.predict(
            SAMPLE_IMAGE, points=[640, 420], labels=[1], multimask=True
        )
        print(len(result.masks))
source_hash: bb70ff24e6c0a767
---

## Definição

A segmentação por prompt recebe uma imagem mais um prompt espacial e retorna a
máscara daquilo que o prompt aponta. Nada é classificado: não existe lista de
classes, e `result.boxes` guarda boxes justos derivados das máscaras, e não
detecções por direito próprio. `result.masks` carrega os dados das máscaras e
`result.masks.xy` os polígonos delas.

O prompt é a interface. `points` são coordenadas de pixel `[x, y]`, um conjunto
por objeto, com `labels` marcando cada ponto como positivo (1, inclua este) ou
negativo (0, exclua este). `bboxes` é `[x1, y1, x2, y2]`, uma máscara por box.
Pontos e boxes podem ser combinados, e nesse caso eles se pareiam por objeto e
precisam ter o mesmo comprimento. Omitir todos os prompts roda o caminho de
segmentar tudo, uma grade de pontos sobre a imagem.

Um único ponto é ambíguo por construção. Clicar em uma manga pode significar a
manga, a camisa ou a pessoa, então `multimask=True` retorna essas três máscaras
de todo-versus-parte por prompt em vez da única melhor. `conf` filtra pelo IoU
predito pelo modelo, uma pontuação de qualidade da máscara, não uma confiança de
detecção.

O LibreYOLO não tem uma chave de tarefa `promptable`. O nível se registra como
`segment`, a mesma chave que a segmentação de instâncias usa. O que o separa é o
formato da chamada, e é por isso que ele tem a própria factory, `LibreSAM()`,
irmã de `LibreYOLO()`, `LibreOpenVocab()` e `LibreVLM()`. Uma única assinatura
`predict(image)` não consegue expressar o laço para o qual esses modelos foram
feitos: `set_image()` roda o encoder de imagem uma vez e guarda os embeddings em
cache, cada chamada posterior de `predict()` com `source=None` paga só pela
decodificação do prompt, e `reset_image()` limpa o cache. O encoder de imagem é
o custo dominante e roda uma vez por imagem, então um segundo prompt na mesma
imagem o pula por completo.

## Modelos

Seis famílias carregam pelo `LibreSAM` por alias.

O [SAM](/docs/models/sam) é o padrão, nos tamanhos `base`, `large` e `huge`,
também escritos `b`, `l` e `h`.

O [SAM 2](/docs/models/sam-2), como `sam2-tiny`, `sam2-small`, `sam2-base-plus`
e `sam2-large`. O LibreYOLO suporta o caminho de imagem dele.

O [SAM 3](/docs/models/sam-3), como `sam3`, é a única família que aceita um
prompt de conceito em texto: `text="yellow school bus"` retorna todas as
instâncias que combinam. Passar `text=` para qualquer outra família levanta um
erro com uma mensagem citando o SAM 3. Os pesos dele vêm da Meta sob a SAM
License personalizada, em vez da licença MIT do LibreYOLO, e o repositório é
restrito: aceite os termos na página do modelo e autentique-se com
`hf auth login` antes do primeiro download. Leia
[SAM 3](/docs/models/sam-3) antes de fazer deploy dele.

O [EdgeTAM](/docs/models/edgetam), como `edgetam`, é uma variante on-device do
SAM 2. O LibreYOLO suporta o caminho de imagem dele.

O [MobileSAM](/docs/models/mobilesam), como `mobilesam`, troca o encoder ViT-H
do SAM por um TinyViT destilado.

O [PicoSAM3](/docs/models/picosam3), como `picosam3`, é uma CNN compacta para
regiões indicadas por box em sensores de borda (edge). Aqui os prompts de box
são todo o contrato: pontos, texto, máscara, multimask e segmentar tudo levantam
um erro com uma mensagem apontando para o SAM 2 ou o SAM 3.

O extra do nível cobre as quatro famílias que carregam via `transformers`:

```bash
pip install "libreyolo[sam]"
```

MobileSAM e PicoSAM3 são ports nativos do LibreYOLO e não precisam da instalação
do `transformers` para rodar.

## Predição

<code-tabs name="predict" />

`source` e `set_image()` são alternativas, não uma sequência: passe uma imagem
para `predict()` para uma chamada de uma vez só, ou chame `set_image()` antes e
depois `predict(source=None)` para cada prompt. Passar `device=` para
`predict()` move o modelo para aquela chamada e para todas as seguintes, e
invalida qualquer embedding em cache.

Segmentar tudo é o modo caro. `points_per_side` vale 32 por padrão, o que dá
aproximadamente 1024 passagens do decoder sobre a imagem; abaixe esse valor para
qualquer coisa interativa na CPU. Nesse modo, `conf` aplica o limiar de grade da
família quando é deixado sem definir, enquanto no caminho com prompt um `conf`
sem definir mantém todas as máscaras. Passe `conf=0.0` para desativar a
filtragem em qualquer um dos dois modos, e `max_det` para limitar quantas
máscaras voltam.

Prompts de máscara não são suportados nesta versão, e `masks=` levanta um erro
em vez de ser ignorado. `track()` também levanta um erro em todo o nível: estes
são segmentadores de imagem, então rode `predict()` por frame. Veja
[predição](/docs/predict) para fontes e tratamento de resultados.

## Treinamento

Nenhuma família deste nível treina dentro do LibreYOLO. `train()` levanta um
erro: faça fine-tuning upstream e carregue os pesos resultantes.

## Validação

Não existe validador para este nível, e `val()` levanta um erro. Uma máscara por
prompt não tem um conjunto fixo de classes contra o qual pontuar, então as
métricas usuais de detecção e segmentação não têm em que se apoiar. Pontuar uma
máscara por prompt significa compará-la com uma máscara de referência que você
mesmo fornece, contra os prompts que importam para você.

## Exportação

A exportação está fora do escopo do nível como um todo e `export()` levanta um
erro, com uma exceção. O [PicoSAM3](/docs/models/picosam3) exporta a CNN de
região 96x96 crua dele para ONNX como `roi_image -> mask_logits`; o recorte pelo
box e o redimensionamento da máscara de volta para as coordenadas da imagem
continuam em Python. Todas as outras famílias rodam por `predict()` no PyTorch.
Veja [exportação](/docs/export) para os formatos disponíveis no resto da
biblioteca.
