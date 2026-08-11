---
title: LingBot-Vision
families:
  - lingbotvision
seo_title: 'LingBot-Vision: phân đoạn ngữ nghĩa trong LibreYOLO'
description: >-
  Dùng LingBot-Vision trong LibreYOLO để phân đoạn ngữ nghĩa trên backbone ViT
  theo Apache-2.0. Cài đặt, dự đoán, huấn luyện, đánh giá và xuất các kích thước
  s/b/l.
lead: >-
  LingBot-Vision là họ backbone vision transformer tự giám sát do Robbyant phát
  hành, được huấn luyện bằng masked modeling tập trung vào biên để nhận thức
  không gian dày đặc. LibreYOLO ghép backbone với một head dày đặc và hỗ trợ cho
  một tác vụ là phân đoạn ngữ nghĩa.
keywords:
  - LingBot-Vision
  - phân đoạn ngữ nghĩa
  - vision transformer
  - tiền huấn luyện tự giám sát
  - mô hình hóa biên
  - Robbyant
  - dự đoán dày đặc
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLingBotVisions-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (linear probe)
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Backbone được đóng băng theo mặc định, khớp với giao thức đánh giá
        # thượng nguồn: chỉ head dày đặc 1x1 được huấn luyện.
        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(data="my-dataset.yaml", epochs=20, imgsz=512, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 imgsz=512 batch=16
    - label: Tinh chỉnh toàn bộ
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(
            data="my-dataset.yaml", epochs=20, imgsz=512, batch=16,
            freeze_backbone=False,
        )
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLingBotVisions-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="coreai", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLingBotVisions-sem.pt format=onnx imgsz=512
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, nên artifact đã xuất được tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreLingBotVisions-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c47b33fdc6fa1139
---

## Cài đặt

LingBot-Vision không cần extra tùy chọn. Mọi thành phần được import đều có trong
bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải từ Hugging Face ở lần dùng đầu tiên và lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

`result.semantic_mask` chứa bản đồ lớp dày đặc: `.data` là tensor `(H, W)` của
các class ID trên kích thước ảnh gốc, còn `.classes` liệt kê các class ID thực
sự xuất hiện. `result.boxes` là `None` vì không có phát hiện theo instance.
`conf` và `iou` được chấp nhận để đồng nhất API nhưng không thay đổi đầu ra vì
mô hình trả về một lớp cho mỗi pixel thay vì các phát hiện cần lọc. Xem
[dự đoán](/docs/predict) để biết về nguồn, streaming và xử lý kết quả.

## Các biến thể

Ba kích thước đã công bố là s, b và l, được chưng cất từ mô hình giáo viên
ViT-g/16 có 1,1 tỷ tham số. Bản thân mô hình giáo viên kích thước `g` có thể tải
và tinh chỉnh trong LibreYOLO, nhưng LibreYOLO không lưu trữ checkpoint `g` riêng.

<checkpoint-table />

## Huấn luyện

`train()` tinh chỉnh một checkpoint đã công bố. Công thức mặc định là linear
probe trong báo cáo thượng nguồn: backbone ViT được đóng băng và chỉ head dày đặc
1x1 được huấn luyện, khớp với cách tạo ra các trọng số do LibreYOLO lưu trữ ở
trên. Truyền `freeze_backbone=False` để tinh chỉnh toàn bộ mạng và giảm `lr0`
cho phù hợp.

<code-tabs name="train" />

Xem [huấn luyện](/docs/train) để biết về tập dữ liệu, tăng cường dữ liệu, multi-GPU và logger.

## Đánh giá

`val()` trả về một từ điển các khóa `metrics/`: mIoU và độ chính xác theo pixel,
được đo trên bất kỳ tập dữ liệu nào theo định dạng bạn đã dùng để huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`.
[Xuất](/docs/export) liệt kê các đối số mà từng định dạng chấp nhận.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này.

<checkpoint-table />

## Giấy phép

<provenance-box>

Bản phát hành thượng nguồn mô tả ViT được xây dựng trên kiến trúc DINOv2/DINOv3
do Meta AI công bố. Robbyant phân phối bản triển khai theo Apache-2.0, và bản
chuyển đổi LibreYOLO này chỉ được tạo từ repo Robbyant, không dùng mã DINOv2
hoặc DINOv3 của Meta.

</provenance-box>

## Trích dẫn

<citation-block />

