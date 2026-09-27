---
title: Swin Transformer
families:
  - swin
seo_title: 'Swin Transformer: phân loại ảnh bằng LibreSwin của LibreYOLO'
description: >-
  Dự đoán, xác thực và xuất bộ phân loại Swin Transformer bằng LibreYOLO. Trọng
  số MIT; chưa hỗ trợ tinh chỉnh.
lead: >-
  Swin Transformer V1: vision transformer phân cấp tính attention bên trong các
  cửa sổ cục bộ dịch chuyển thay vì trên toàn ảnh. LibreYOLO cung cấp bốn kích
  thước để phân loại ảnh.
keywords:
  - Swin Transformer
  - vision transformer phân cấp
  - shifted window attention
  - phân loại ảnh
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwint-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwint-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSwint-cls.pt")


        # data là thư mục gốc có các phần tách thư mục lớp đối tượng train/ và
        val/

        # (bố cục ImageFolder), không phải YAML dataset.

        metrics = model.val(data="imagenet-1k/")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwint-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwint-cls.pt format=onnx
        libreyolo export model=LibreSwint-cls.pt format=tensorrt half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreSwint-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: faa6bbacae62d88e
---

## Cài đặt

Swin không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Bộ phân loại trả về `result.probs` thay cho `result.boxes`: `top1` và `top5` cho chỉ mục lớp đối tượng, còn `top1conf` và `top5conf` cho độ tin cậy tương ứng. Mọi kích thước được cố định ở đầu vào 224px vì giai đoạn attention cuối được xây dựng cho độ phân giải đó; dự đoán, xác thực và xuất đều phát sinh lỗi nếu bạn truyền `imgsz` khác. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có bốn kích thước từ tiny đến large, được xây dựng từ cùng shifted-window tower và khác nhau ở độ rộng embedding cùng độ sâu giai đoạn. Large được huấn luyện sẵn trên ImageNet-22k rồi tinh chỉnh trên ImageNet-1k; ba kích thước còn lại được huấn luyện trực tiếp trên ImageNet-1k. LibreYOLO chỉ cung cấp họ mô hình này để inference: hỗ trợ dự đoán, xác thực top-1/top-5 kiểu ImageNet và xuất, còn công thức huấn luyện ImageNet upstream chưa được triển khai.

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


