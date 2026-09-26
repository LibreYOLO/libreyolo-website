---
title: Citação
seo_title: Como citar o LibreYOLO e os autores originais
description: >-
  Como citar o LibreYOLO em um artigo e como citar os autores da família de
  modelos que você executou. Os dois entram na mesma seção de métodos.
lead: >-
  Uma citação completa do LibreYOLO tem duas partes: a biblioteca e o trabalho
  publicado por trás da família de modelos que produziu o resultado.
keywords:
  - citar libreyolo
  - libreyolo bibtex
  - libreyolo citation cff
  - como citar modelo de visão computacional
  - citar artigo de detecção de objetos
  - referência bibliográfica visão computacional
last_verified: 1.5.0
source_hash: 0f3f23e4e85e38be
---

## Como citar o LibreYOLO

O repositório publica seus metadados de citação como
[`CITATION.cff`](https://github.com/LibreYOLO/libreyolo/blob/release/CITATION.cff),
e não como um bloco BibTeX. O GitHub lê esse arquivo e oferece um botão Cite this
repository na página do repositório, que gera APA e BibTeX a partir dele. Pegue a
entrada de lá em vez de digitar uma à mão.

O arquivo completo:

```yaml
cff-version: 1.2.0
message: "If you use LibreYOLO in your research or software, please cite it as below."
title: "LibreYOLO"
type: software
authors:
  - family-names: Ceccon
    given-names: Xuban
  - name: "The LibreYOLO contributors"
license: MIT
url: "https://github.com/LibreYOLO/libreyolo"
repository-code: "https://github.com/LibreYOLO/libreyolo"
```

Ele não traz versão nem data de publicação de propósito.
[`RELEASING.md`](https://github.com/LibreYOLO/libreyolo/blob/release/RELEASING.md)
diz aos maintainers para nunca alterar a versão, a data ou o título de
`CITATION.cff` nem de `.zenodo.json` durante um release, para que toda citação
caia em um único registro em vez de se espalhar entre versões. Informe no seu
próprio texto a versão que você executou e deixe a citação como está.

## Como citar a família de modelos

O LibreYOLO é um port. Rodar `LibreRFDETRm.pt` significa rodar o RF-DETR, e as
pessoas que escreveram o RF-DETR são quem um revisor espera ver creditado.
Citar só a biblioteca atribui o trabalho delas ao projeto errado.

Tudo o que você precisa está na página da família. A linha Upstream do cabeçalho
nomeia o trabalho original e a organização por trás dele, e traz links para o
paper e para o repositório de código-fonte. A seção Citação, mais abaixo, contém
o BibTeX.

Esse BibTeX é copiado literalmente do bloco de citação dos próprios autores,
normalmente a seção Citation do README upstream ou um `CITATION.cff`, e é
exibido com um link de volta para o bloco de onde veio, para que você possa
conferir com a fonte. Ele nunca é montado a partir dos metadados do paper. Uma
entrada reconstruída à mão falha de forma silenciosa e cara: um coautor perdido,
a conferência errada, o tipo de entrada errado, um ano que pertence ao preprint.
Preprints também acabam sendo aceitos, então uma entrada pode ser um
`@inproceedings` mesmo que a versão que você leu estivesse no arXiv.

Copie o bloco como ele está. Se o seu estilo bibliográfico precisar de outro tipo
de entrada, converta a entrada em vez de redigitá-la, e mantenha a lista de
autores na ordem original.

## O que uma seção de métodos precisa

Três coisas tornam um resultado do LibreYOLO reproduzível e corretamente
atribuído:

- A biblioteca, citada a partir do `CITATION.cff`, junto com a versão que você
  executou. `libreyolo version` imprime isso, junto com as versões de Python,
  torch e CUDA contra as quais ela está rodando.
- O trabalho upstream, citado a partir da seção Citação da página da família.
- O nome exato do arquivo do checkpoint, como `LibreRFDETRm.pt`. Os tamanhos
  dentro de uma família se comportam de formas diferentes, e várias famílias
  publicam checkpoints treinados em datasets diferentes sob o mesmo prefixo,
  então o nome da família sozinho não identifica o que rodou.

A atribuição também é um termo de licença para boa parte do que o LibreYOLO
publica. Tanto a Apache-2.0 quanto a família CC BY exigem que o aviso acompanhe
os pesos que você redistribui, o que é uma obrigação separada de citar um paper.
Veja [licenciamento](/docs/licensing) para saber quais termos se aplicam a qual
checkpoint.
