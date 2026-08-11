---
title: 인용
seo_title: LibreYOLO와 상류 저자를 인용
description: 논문에서 LibreYOLO를 인용하는 방법과 실행한 모델 패밀리의 저자를 인용하는 방법. 둘 다 동일한 방법론 섹션에 포함됩니다.
lead: '완전한 LibreYOLO 인용에는 두 부분이 있습니다: 라이브러리와 결과를 생성한 모델 패밀리 뒤의 출판된 작업.'
keywords:
  - libreyolo를 인용
  - libreyolo bibtex
  - libreyolo 인용 cff
  - 모델 인용
  - 컴퓨터 비전 인용
last_verified: 1.5.0
source_hash: 0f3f23e4e85e38be
---

## LibreYOLO 인용

저장소는 BibTeX 블록이 아닌 [`CITATION.cff`](https://github.com/LibreYOLO/libreyolo/blob/release/CITATION.cff)로 인용 메타데이터를 게시합니다. GitHub는 해당 파일을 읽고 저장소 페이지에서 '이 저장소 인용' 버튼을 제공하며, 여기서 APA와 BibTeX 형식을 생성합니다. 직접 입력하지 말고 거기에서 항목을 가져오세요.

전체 파일:

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

버전과 릴리스 날짜가 의도적으로 표시되지 않습니다. [`RELEASING.md`](https://github.com/LibreYOLO/libreyolo/blob/release/RELEASING.md)는 유지보수자에게 `CITATION.cff` 또는 `.zenodo.json`를 릴리스 동안 버전 업, 날짜 변경 또는 제목 변경하지 말라고 알려주어, 모든 인용이 여러 버전에 흩어지지 않고 한 기록에 모이도록 합니다. 실행한 버전을 본문에서 보고하고, 인용은 그대로 두세요.

## 모델 계열 인용하기

LibreYOLO는 포트입니다. `LibreRFDETRm.pt`를 실행한다는 것은 RF-DETR를 실행한다는 의미이고, RF-DETR를 작성한 사람이 리뷰어가 신용을 기대하는 대상입니다. 라이브러리만 인용하면 그들의 작업을 잘못된 프로젝트에 귀속시키게 됩니다.

필요한 모든 내용은 계열 페이지에 있습니다. 헤더의 Upstream 행에는 원본 작업과 그 배후의 조직 이름이 표시되며, 논문과 소스 저장소 링크가 포함됩니다. 아래 Citation 섹션에는 BibTeX가 있습니다.

그 BibTeX는 저자들이 직접 작성한 인용 블록에서 그대로 복사한 것이며, 일반적으로 상위 README의 Citation 섹션이나 `CITATION.cff`에서 가져온 것이고, 원본 블록으로 돌아가는 링크가 표시되어 출처와 대조할 수 있습니다. 논문 메타데이터에서 조립되는 경우는 없습니다. 수동으로 항목을 재작성하면 눈에 띄지 않게 오류가 발생하고 비용이 많이 듭니다: 공동 저자가 빠지거나, 잘못된 학회/저널, 잘못된 항목 유형, 사전출판(preprint) 연도가 들어가는 경우 등입니다. 사전출판도 인용될 수 있기 때문에, 읽은 버전이 arXiv에 있었더라도 항목이 `@inproceedings`일 수 있습니다.

블록을 있는 그대로 복사하세요. 참고문헌 스타일이 다른 항목 유형을 요구하는 경우, 다시 입력하지 말고 항목을 변환하며, 저자 목록은 원래 순서를 유지하세요.

## Methods 섹션에 필요한 것

LibreYOLO 결과를 재현 가능하고 올바르게 인용되도록 만드는 세 가지 요소:

- 라이브러리, `CITATION.cff`에서 인용, 실행한 버전과 함께. `libreyolo version`는 실행 중인 Python, torch 및 CUDA 버전과 함께 이를 출력합니다.
- 업스트림 작업, 계열 페이지의 인용 섹션에서 인용.
- 정확한 체크포인트 파일 이름, 예: `LibreRFDETRm.pt`. 계열 내 크기는 다르게 동작하며, 여러 계열이 동일한 접두사 하에 다른 데이터셋에서 학습된 체크포인트를 공개하기 때문에 계열 이름만으로는 실행한 내용을 식별할 수 없습니다.

저작자 표시(attribution)는 LibreYOLO가 게시하는 많은 것들의 라이선스 조건이기도 합니다. Apache-2.0과 CC BY 계열은 재배포하는 가중치(weights)에 함께 공지를 포함할 것을 요구하며, 이는 논문을 인용하는 것과는 별도의 의무입니다. 어떤 조건이 어떤 체크포인트에 적용되는지는 [라이선스](/docs/licensing)를 참조하십시오.
