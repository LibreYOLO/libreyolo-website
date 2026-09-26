---
title: ConvNeXt
families:
  - convnext
seo_title: 'ConvNeXt: huấn luyện, xác thực và xuất theo Apache-2.0'
description: >-
  Dùng ConvNeXt trong LibreYOLO để phân loại ảnh. Cài đặt, dự đoán, tinh chỉnh
  bằng LoRA, xác thực và xuất LibreConvNeXt tiny/small/base.
lead: >-
  ConvNeXt là bộ phân loại ảnh được xây dựng hoàn toàn từ các phép tích chập
  tiêu chuẩn, được hiện đại hóa theo từng block từ ResNet theo các lựa chọn
  thiết kế của vision transformer. LibreYOLO hỗ trợ mô hình này cho một tác vụ:
  phân loại.
keywords:
  - ConvNeXt
  - ConvNeXt tiny
  - phân loại ảnh
  - mạng tích chập thuần túy
  - bộ phân loại ImageNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreConvNeXtt-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 epochs=5
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreConvNeXtt-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreConvNeXtt-cls.pt format=onnx
        libreyolo export model=LibreConvNeXtt-cls.pt format=tensorrt half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreConvNeXtt-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 1682cc69cf2925e6
---

## Cài đặt

ConvNeXt không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

Tinh chỉnh bằng adapter với `lora=True` là ngoại lệ và cần extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang mô hình khác chỉ cần thay đổi một dòng. Bộ phân loại không có box hay mặt nạ (mask): `result.probs` chứa dự đoán cho toàn ảnh, với `top1`, `top5`, `top1conf` và `top5conf`. `conf`, `iou` và `max_det` được chấp nhận để giữ tính tương đồng của API nhưng không có tác dụng vì không có gì để áp dụng ngưỡng hoặc loại bỏ trên một vector xác suất duy nhất. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Ba kích thước tiny/small/base đều được huấn luyện và đánh giá theo cùng một cách, vì vậy việc chọn một kích thước là sự đánh đổi trực tiếp giữa số lượng tham số và độ chính xác. Tác vụ được cố định: mọi kích thước chỉ hỗ trợ phân loại. Tên tệp trọng số của mọi kích thước đều kết thúc bằng `-cls.pt`, và hậu tố đó là thứ factory đọc để định tuyến đến họ mô hình này; không cần đối số `task=`.

## Huấn luyện

Quá trình tinh chỉnh bắt đầu từ backbone ImageNet đã phát hành và tự động xây dựng lại lớp phân loại cuối cùng theo số lượng lớp đối tượng của dataset đích.

<code-tabs name="train" />

Khi giữ nguyên thiết lập, trainer chạy 100 epoch ở `lr0=1e-3` với AdamW, batch 64 và early stopping sau 50 epoch không có cải thiện. `data` chấp nhận thư mục gốc của tập dữ liệu (dataset) (`train/` và `val/`, mỗi lớp đối tượng một thư mục), tên rút gọn đã biết như `imagenette160` hoặc URL `.zip`. Các block của ConvNeXt có những MLP `nn.Linear` mà LoRA cần, vì vậy `lora=True` được hỗ trợ ở đây và chèn adapter vào các MLP của block thay vì tinh chỉnh toàn bộ backbone.

Xem [huấn luyện](/docs/train) để biết về dataset, tăng cường dữ liệu (data augmentation), multi-GPU và logger.

## Xác thực

`val()` trả về dictionary gồm các key `metrics/`. Với tác vụ phân loại, đó là độ chính xác top-1 và top-5 trên phần tách xác thực.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`. Trang [Xuất](/docs/export) liệt kê các đối số được mọi định dạng chấp nhận và các extra mà một số định dạng bổ sung.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box>

Chỉ ConvNeXt V1 được cung cấp trong họ mô hình này. Các checkpoint nhỏ được huấn luyện sẵn (pretrained) của ConvNeXt-V2 dùng giấy phép CC-BY-NC 4.0 và được chủ ý loại trừ, vì không thể phân phối lại trọng số phi thương mại trong một thư viện MIT/thương mại.

</provenance-box>

## Trích dẫn

<citation-block />
