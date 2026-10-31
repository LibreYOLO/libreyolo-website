---
title: API open-vocabulary
seo_title: 'API LibreOpenVocab: alias và đối số'
description: >-
  Factory LibreOpenVocab, bốn họ cùng mọi alias, set_classes, giá trị conf mặc
  định theo họ và quy tắc text_threshold cùng iou.
lead: >-
  LibreOpenVocab là factory cho detector được điều kiện hóa bằng văn bản. Danh
  sách lớp là prompt thay vì head cố định, nên vocabulary được đặt bằng
  set_classes và mô hình trả về Results detection thông thường theo vocabulary
  đó.
keywords:
  - LibreOpenVocab
  - open vocabulary detection
  - Grounding DINO
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - set_classes
last_verified: 1.5.0
verification: >-
  Alias được đọc từ libreyolo/models/openvocab/__init__.py; repo, kích thước và
  ngưỡng từ grounding_dino.py, owlv2.py, omdet_turbo.py và ov_deim.py; quy tắc
  gọi từ libreyolo/models/openvocab/base.py, tất cả ở v1.5.0. Ý định thiết kế từ
  docs/adr/0008-open-vocab-detector-contract.md.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[openvocab]'
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-tiny")
        model.set_classes(["person", "skateboard", "handrail"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
source_hash: 64e4c641c6f8cde0
---

## Cài đặt

Cấp này cần gói bổ sung `openvocab`.

<code-tabs name="install" />

## Factory

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model` là alias, không phải đường dẫn. Dấu gạch dưới được chuyển thành dấu
gạch nối trước khi tra cứu, nên tên kèm họ mà danh mục CLI in ra, như
`omdet_turbo-t` và `grounding_dino-t`, được nạp nguyên như đã cho. Alias không
xác định phát `ValueError` kèm mọi alias đã biết.

Constructor nhận `size`, `nb_classes=80`, `names=None`, `device="auto"`,
`task=None` và `text_threshold=None`. Truyền `names` tương đương gọi
`set_classes` ngay sau khi nạp. Truyền `text_threshold` cho họ không hỗ trợ sẽ
phát `TypeError`.

<code-tabs name="usage" />

## Họ và alias

| Họ | Alias | Kích thước | Trọng số |
|---|---|---|---|
| Grounding DINO | `grounding-dino`, `groundingdino`, `grounding-dino-tiny`, `groundingdino-tiny`, `grounding-dino-t`, `groundingdino-t`, `grounding-dino-base`, `groundingdino-base`, `grounding-dino-b`, `groundingdino-b` | `t`, `b` | `LibreYOLO/LibreGroundingDINOt`, `LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`, `owl-v2`, `owlv2-base`, `owl-v2-base`, `owlv2-b16`, `owl-v2-b16`, `owlv2-large`, `owl-v2-large`, `owlv2-l14`, `owl-v2-l14` | `b16`, `l14` | `LibreYOLO/LibreOWLv2b16`, `LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`, `omdet`, `omdetturbo`, `omdet-turbo-tiny`, `omdet-turbo-swin-tiny`, `omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`, `ovdeim`, `ov-deim-s`, `ovdeim-s`, `ov-deim-m`, `ovdeim-m`, `ov-deim-l`, `ovdeim-l` | `s`, `m`, `l` | `LibreYOLO/LibreOVDEIMs`, `LibreYOLO/LibreOVDEIMm`, `LibreYOLO/LibreOVDEIMl` |

Alias mặc định là `grounding-dino-tiny`.

`LibreGroundingDINO`, `LibreOWLv2` và `LibreOMDetTurbo` được export ở cấp
package và có thể dựng trực tiếp bằng `size=`. OV-DEIM được truy cập qua các
alias factory ở trên.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

Đặt vocabulary cho mọi lệnh `predict()` về sau và trả về mô hình để có thể nối
lệnh gọi. Danh sách phải khác rỗng, chỉ chứa chuỗi và các mục phải duy nhất khi
so sánh không phân biệt hoa thường; nhãn trống bị từ chối. Truyền một chuỗi
thuần phát `TypeError` vì chuỗi sẽ bị duyệt thành các lớp một ký tự.

Sau lệnh gọi, `model.names` ánh xạ `0..N-1` đến nhãn theo thứ tự đã cho và
`model.nb_classes` là `N`.

## Đối số lệnh gọi

Cấp này tái sử dụng giao diện predict tiêu chuẩn với ba khác biệt.

`conf` mặc định theo giá trị riêng của họ thay vì 0.25 dùng chung:

| Họ | conf mặc định | Suppression |
|---|---|---|
| Grounding DINO | 0.25 | |
| OWLv2 | 0.1 | |
| OMDet-Turbo | 0.3 | Hậu xử lý riêng, ngưỡng 0.5, tuân theo `iou=` |
| OV-DEIM | 0.25 | Ghép one-to-one với lựa chọn top-K, không suppression |

`iou=` chỉ có ý nghĩa với họ chạy suppression. OMDet-Turbo nhận ngưỡng làm đối
số và mặc định 0.5 khi không đặt `iou=`. Ba họ còn lại không suppression, nên
truyền `iou=` sẽ phát cảnh báo và bị bỏ qua.

`text_threshold=` chỉ dành cho Grounding DINO, mặc định 0.25. Có thể truyền lúc
dựng để giữ giá trị lâu dài hoặc theo từng lệnh gọi. Giá trị theo lệnh gọi
không thể kết hợp `stream=True` vì kết quả stream được tạo lazy; hãy đặt trên
constructor. Mọi họ khác phát `TypeError` với đối số này.

`imgsz=` phát `ValueError`: pipeline tiền xử lý sở hữu việc đổi kích thước ở
cấp này. `augment=True` cũng báo lỗi vì test-time augmentation nằm ngoài phạm
vi. Kích thước đầu vào chỉ được ghi theo từng họ để tham khảo: Grounding DINO
800, OWLv2 960 và 1008, OMDet-Turbo 640, OV-DEIM 640.

## Không được hỗ trợ

`train()`, `val()`, `track()` và `export()` đều phát `NotImplementedError`.
Hãy tinh chỉnh ở upstream và nạp trọng số kết quả; chạy `predict()` theo từng
frame thay cho tracking. Xác thực cần validator riêng vì validator detection
dùng chung gọi mô hình bằng tensor ảnh trong khi cấp này cần đầu vào được điều
kiện hóa bằng văn bản.
