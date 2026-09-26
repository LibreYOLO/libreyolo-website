---
title: VGG
families:
  - vgg
seo_title: 'VGG: chạy bộ phân loại ảnh VGG-16/19 trong LibreYOLO'
description: >-
  Dự đoán, xác thực và xuất bộ phân loại VGG bằng LibreYOLO. Trọng số
  torchvision BSD-3-Clause; chưa hỗ trợ tinh chỉnh.
lead: >-
  VGG là bộ phân loại ảnh tích chập được xây dựng từ các stack đồng nhất gồm
  phép tích chập nhỏ 3x3 thay vì filter lớn hơn. LibreYOLO cung cấp kích thước
  16 và 19 lớp, dạng thuần túy và có batch normalization, để phân loại ảnh.
keywords:
  - VGG
  - VGG-16
  - VGG-19
  - mạng neural tích chập
  - phân loại ảnh
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVGG16-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreVGG16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreVGG16-cls.pt")


        # data là thư mục gốc có các phần tách thư mục lớp đối tượng train/ và
        val/

        # (bố cục ImageFolder), không phải YAML dataset.

        metrics = model.val(data="imagenet-1k/")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreVGG16-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreVGG16-cls.pt format=onnx
        libreyolo export model=LibreVGG16-cls.pt format=tensorrt half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreVGG16-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 26eb6ff5811533fd
---

## Cài đặt

VGG không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Bộ phân loại trả về `result.probs` thay cho `result.boxes`: `top1` và `top5` cho chỉ mục lớp đối tượng, còn `top1conf` và `top5conf` cho độ tin cậy tương ứng. Dự đoán chạy ở đầu vào cố định 224px và phát sinh lỗi nếu bạn truyền `imgsz` khác. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có bốn kích thước: 16 và 19 lớp tích chập, mỗi kích thước có biến thể thuần túy và được batch-normalized. Trọng số đi kèm đến từ quá trình huấn luyện ImageNet từ đầu về sau của torchvision, không phải bản chuyển đổi từ bản phát hành Caffe năm 2014 gốc của nhóm Oxford. LibreYOLO chỉ cung cấp họ mô hình này để inference: hỗ trợ dự đoán, xác thực top-1/top-5 kiểu ImageNet và xuất, còn tinh chỉnh chưa được triển khai.

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


