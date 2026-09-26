---
title: 인용
seo_title: LibreYOLO과 업스트림 저자들을 인용하십시오
description: 논문에서 LibreYOLO를 인용하는 방법과 실행한 모델 계열의 저자를 인용하는 방법. 둘 다 동일한 방법 섹션에 포함됩니다.
lead: '완전한 LibreYOLO 인용에는 두 부분이 있습니다: 라이브러리와 결과를 만들어낸 모델 계열 뒤에 있는 출판된 작업입니다.'
keywords:
  - libreyolo 인용
  - libreyolo 비브텍스
  - libreyolo 인용 cff
  - 모델 인용
  - 컴퓨터 비전 인용
last_verified: 1.5.0
source_hash: 0f3f23e4e85e38be
---

## LibreYOLO 인용

이 저장소는 BibTeX 블록이 아니라 [`CITATION.cff`](https://github.com/LibreYOLO/libreyolo/blob/release/CITATION.cff)로 인용 메타데이터를 게시합니다. GitHub는 해당 파일을 읽고 저장소 페이지에서 '이 저장소 인용' 버튼을 제공하며, 이를 통해 APA와 BibTeX 양식을 생성합니다. 입력하는 대신 그곳에서 항목을 가져오십시오.

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

의도적으로 버전이나 출시일이 표시되어 있지 않습니다. [`RELEASING.md`](https://github.com/LibreYOLO/libreyolo/blob/release/RELEASING.md)는 유지관리자들에게 릴리스 동안 `CITATION.cff` 또는 `.zenodo.json`의 버전을 올리거나, 날짜를 변경하거나, 제목을 바꾸지 않도록 지시합니다. 이는 모든 인용이 여러 버전에 흩어지지 않고 하나의 기록에 모이도록 하기 위함입니다. 자신이 실행한 버전을 본문에 보고하고, 인용은 그대로 두십시오.

## 모범 계열을 인용하며

LibreYOLO는 포트입니다. `LibreRFDETRm.pt`를 실행하는 것은 RF-DETR을 실행하는 것을 의미하며, RF-DETR을 작성한 사람들이 리뷰어가 크레딧을 기대하는 사람들입니다. 라이브러리만 인용하면 그들의 작업을 잘못된 프로젝트에 속하게 됩니다.

필요한 모든 것은 계열 페이지에 있습니다. 헤더의 Upstream 행은 원본 작업과 그 뒤에 있는 조직을 명시하고, 논문과 소스 저장소를 연결합니다. 아래쪽의 Citation 섹션에는 BibTeX가 있습니다.

그 BibTeX는 저자들의 자체 인용 블록, 일반적으로 상위 README의 Citation 섹션이나 `CITATION.cff`에서 단어 그대로 복사한 것이며, 원본 블록으로 돌아가는 링크와 함께 렌더링되어 소스를 확인할 수 있습니다. 이는 논문 메타데이터에서 조합된 것이 아닙니다. 수동으로 재작성된 항목은 조용히 그리고 비용이 많이 들어가며 실패합니다: 공동 저자가 빠지거나, 잘못된 학회지, 잘못된 항목 유형, 사전 인쇄물에 속한 연도 등. 사전 인쇄물도 수락되므로, 읽은 버전이 arXiv에 있더라도 항목은 `@inproceedings`일 수 있습니다.

블록을 그대로 복사하십시오. 참고문헌 양식에 다른 항목 유형이 필요하면 재작성하지 말고 항목을 변환하며, 저자 목록은 원래 순서를 유지하십시오.

## 방법 섹션에 필요한 것

세 가지가 LibreYOLO 결과를 재현 가능하고 올바르게 귀속되도록 만듭니다:

- 도서관은 `CITATION.cff`에서 인용되었으며, 실행한 버전과 함께 제공됩니다. `libreyolo version`는 Python, torch 및 실행 중인 CUDA 버전과 함께 그것을 출력합니다.
- 계열 페이지의 인용 섹션에서 인용된 업스트림 작업.
- 정확한 체크포인트 파일명, 예를 들어 `LibreRFDETRm.pt`. 같은 계열 내에서도 크기에 따라 동작이 다르며, 여러 계열에서는 동일한 접두사 아래에서 다른 데이터셋으로 학습된 체크포인트를 공개하기 때문에, 계열 이름만으로는 어떤 것이 실행되었는지 식별할 수 없습니다.

저작자 표시(Attribution)는 LibreYOLO가 게시하는 많은 것들에 대한 라이선스 조건이기도 합니다. Apache-2.0과 CC BY 계열 모두 재배포하는 가중치(weights)에 공지를 함께 전달하도록 요구하며, 이는 논문을 인용하는 것과는 별개의 의무입니다. 어떤 조건이 어떤 체크포인트에 적용되는지에 대해서는 [라이선스](/docs/licensing)를 참조하십시오.
