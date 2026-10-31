---
title: CLIP
families:
  - clip
seo_title: 'CLIP trong LibreYOLO: phân loại zero-shot và tạo embedding'
description: >-
  Dùng CLIP trong LibreYOLO để phân loại ảnh zero-shot và tạo embedding ảnh/văn
  bản. Không cần huấn luyện: set_classes() xác định tập nhãn tại runtime.
lead: >-
  CLIP là mô hình hai nhánh chấm điểm một ảnh theo các prompt văn bản thay vì
  một tập nhãn cố định. LibreYOLO hỗ trợ mô hình này cho phân loại zero-shot và
  tạo embedding ảnh/văn bản mà không cần bước huấn luyện.
keywords:
  - CLIP
  - OpenCLIP
  - phân loại ảnh zero-shot
  - embedding ảnh
  - embedding văn bản
  - mô hình open vocabulary
  - LAION-2B
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Khi không gọi set_classes(), lệnh predict của CLI dùng 1.000 tên lớp

        # ImageNet mà mô hình tải theo mặc định.

        libreyolo predict model=LibreCLIPb32-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Embedding ảnh và văn bản
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image_embed = model(SAMPLE_IMAGE).embeddings.data

        text_embed = model.embed_text("a photo of a forklift")


        # Cả hai đều được chuẩn hóa L2, vì vậy tích vô hướng chính là độ tương
        đồng cosine.

        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        # data là thư mục gốc ImageFolder có phần train/; tên thư mục của nó
        # trở thành prompt lớp zero-shot cho lần chạy này.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCLIPb32-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a forklift", "an empty aisle", "a spill"])

        model.export(format="onnx")


        # Các nhãn set_classes() hiện tại và độ phân giải đầu vào được đóng cố
        định

        # vào graph. Hãy xuất lại sau khi thay đổi một trong hai.
    - label: CLI
      language: bash
      code: |
        # Không gọi set_classes() ở đây, vì vậy thao tác này đóng cố định 1.000
        # lớp ImageNet mặc định mà mô hình tải cùng.
        libreyolo export model=LibreCLIPb32-cls.pt format=onnx
    - label: Xuất embedding
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" truy vết riêng nhánh ảnh; không cần lớp đối tượng.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: ac7cfd75ad6c0fa7
---

## Cài đặt

CLIP cần extra riêng, extra này kéo về các package mà BPE tokenizer đi kèm sử dụng để tái tạo chính xác các ID token.

```bash
pip install "libreyolo[clip]"
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

`set_classes()` là primitive duy nhất biến mô hình này thành bộ phân loại với từ vựng mở (open-vocabulary): hàm dựng từng nhãn theo mọi mẫu prompt, mã hóa rồi lấy trung bình kết quả và lưu ma trận `[K, D]` thu được làm classifier head, vì vậy ma trận không được tính lại cho từng ảnh. Gọi lại hàm bất kỳ lúc nào để thay đổi các lớp đối tượng. Nếu không gọi, LibreCLIP tải với 1.000 tên lớp đối tượng ImageNet-1k đã được thiết lập sẵn.

Với `task="embed"`, thao tác dự đoán trả về một vector ảnh được chuẩn hóa L2 cho mỗi đầu vào thay vì xác suất lớp đối tượng, còn `embed_text()` trả về các hàng văn bản được chuẩn hóa trong cùng không gian vector, vì vậy tích vô hướng thông thường giữa chúng là độ tương đồng cosine. `iou` không có tác dụng với cả hai tác vụ; không có bước NMS. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Xác thực

`val()` đọc tên thư mục lớp đối tượng trong phần tách `train/` của ImageFolder, gọi `set_classes()` với các tên đó, rồi đo độ chính xác top-1 và top-5 zero-shot. Độ chính xác phụ thuộc vào cách tên lớp đối tượng được diễn giải dưới dạng prompt, không phụ thuộc vào bất kỳ cập nhật trọng số nào vì không có gì để huấn luyện. Việc xác thực chỉ áp dụng cho `task="classify"`; `task="embed"` không có trình xác thực dataset.

<code-tabs name="val" />

## Xuất

<export-matrix />

Thao tác xuất đóng trạng thái hiện tại của mô hình vào một graph cố định. Với `task="classify"`, các nhãn được thiết lập gần nhất bởi `set_classes()` và độ phân giải tại thời điểm xuất được đóng vào lớp tuyến tính cuối cùng, do đó graph ONNX hoặc TensorRT đã xuất là một bộ phân loại ảnh `[B, K]` thông thường, không có nhánh văn bản và tokenizer; hãy xuất lại sau khi thay đổi lớp đối tượng hoặc kích thước. Thao tác xuất `task="embed"` chỉ truy vết nhánh ảnh. Cả hai đều cần ONNX opset 14 trở lên, trình xuất thiết lập giá trị này theo mặc định.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này. Cả hai đều được chuyển đổi từ các checkpoint của OpenCLIP được huấn luyện trên LAION-2B (`ViT-B-32` và `ViT-B-16`), không phải từ bất kỳ lần huấn luyện COCO nào.

<checkpoint-table />

Dữ liệu huấn luyện LAION-2B có lịch sử được ghi nhận về nội dung CSAM (Stanford Internet Observatory, tháng 12 năm 2023). Kể từ đó LAION đã phát hành Re-LAION, một bản phát hành lại đã được làm sạch; nếu bạn tiếp tục host lại các trọng số này, nên ưu tiên checkpoint bắt nguồn từ Re-LAION khi có.

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />
