---
title: ViT
families:
  - vit
seo_title: 'ViT: chạy bộ phân loại Vision Transformer kinh điển trong LibreYOLO'
description: >-
  Dự đoán, xác thực và xuất bộ phân loại ViT bằng LibreYOLO. Trọng số AugReg
  Apache-2.0; chưa hỗ trợ tinh chỉnh.
lead: >-
  Vision Transformer kinh điển: transformer thuần túy áp dụng lên các patch ảnh
  kích thước cố định, có class token được học và không dùng phép tích chập.
  LibreYOLO cung cấp bốn kích thước được huấn luyện sẵn bằng AugReg để phân loại
  ảnh.
keywords:
  - ViT
  - Vision Transformer
  - AugReg
  - phân loại ảnh
  - bộ phân loại transformer
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTti-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreViTti-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreViTti-cls.pt")


        # data là thư mục gốc có các phần tách thư mục lớp đối tượng train/ và
        val/

        # (bố cục ImageFolder), không phải YAML dataset.

        metrics = model.val(data="imagenet-1k/")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreViTti-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreViTti-cls.pt format=onnx
        libreyolo export model=LibreViTti-cls.pt format=tensorrt half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreViTti-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: f63e98454913765a
---

## Cài đặt

ViT không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Bộ phân loại trả về `result.probs` thay cho `result.boxes`: `top1` và `top5` cho chỉ mục lớp đối tượng, còn `top1conf` và `top5conf` cho độ tin cậy tương ứng. Bước tiền xử lý đổi kích thước và cắt giữa về đầu vào cố định 224px bằng công thức đánh giá AugReg của timm: nội suy bicubic với tỷ lệ crop 0,9. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có bốn kích thước từ tiny đến large, dùng chung graph patch-16 cố định 224px và khác nhau ở độ rộng embedding cùng độ sâu transformer. LibreYOLO chỉ cung cấp họ mô hình này để inference: hỗ trợ dự đoán, xác thực top-1/top-5 kiểu ImageNet và xuất, còn công thức tinh chỉnh AugReg chưa được triển khai.

## Xác thực

`val()` chạy trên phần tách kiểu ImageFolder (thư mục có thư mục con `train/` và `val/`, mỗi lớp đối tượng một thư mục) rồi trả về độ chính xác top-1 và top-5.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`. Trang [Xuất](/docs/export) liệt kê các đối số được mọi định dạng chấp nhận và các extra mà một số định dạng bổ sung.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />


