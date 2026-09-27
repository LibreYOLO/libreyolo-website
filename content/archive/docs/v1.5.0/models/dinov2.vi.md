---
title: DINOv2
families:
  - dinov2
seo_title: 'DINOv2 trong LibreYOLO: phân đoạn ngữ nghĩa, phân loại và embedding'
description: >-
  Dùng DINOv2 trong LibreYOLO để phân đoạn ngữ nghĩa, phân loại và tạo embedding
  toàn ảnh trên backbone DINOv2-with-Registers. Toàn bộ dùng Apache-2.0.
lead: >-
  DINOv2 là vision transformer tự giám sát được Meta AI huấn luyện để tạo đặc
  trưng ảnh đa dụng mà không cần nhãn. LibreYOLO bọc backbone
  DINOv2-with-Registers cho ba tác vụ: phân đoạn ngữ nghĩa, phân loại và tạo
  embedding toàn ảnh.
keywords:
  - DINOv2
  - DINOv2 with registers
  - học tự giám sát
  - vision transformer
  - phân đoạn ngữ nghĩa
  - embedding ảnh
  - trích xuất đặc trưng
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Phân đoạn ngữ nghĩa
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Không có checkpoint do LibreYOLO host cho họ mô hình này: đoạn này
        # tải backbone DINOv2-with-Registers-small Apache-2.0 từ tổ chức
        # Hugging Face của Meta. Dense head bắt đầu với khởi tạo ngẫu nhiên
        # cho đến khi bạn huấn luyện nó (xem phần Huấn luyện bên dưới).
        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        result = model(SAMPLE_IMAGE)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: Phân loại
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # nb_classes= là số lớp đối tượng của dataset; linear head bắt đầu
        # với khởi tạo ngẫu nhiên cho đến khi bạn huấn luyện nó.
        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
    - label: Embedding
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Bỏ qua mọi task head: chỉ backbone là đủ, vì vậy cách này
        # không cần tinh chỉnh để trở nên hữu ích.
        model = LibreDINOv2(size="s", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (1, D), được chuẩn hóa L2
    - label: Embedding một batch
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Wrapper tiện ích: chạy predict() và xếp mọi hàng vào một
        # tensor (N, D).
        features = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(features.shape)
  train:
    - label: Phân đoạn ngữ nghĩa
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Phân loại
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Multi-GPU
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(
            data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4,
            device="0,1",
        )
  val:
    - label: Phân đoạn ngữ nghĩa
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Phân loại
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
  export:
    - label: Phân đoạn ngữ nghĩa
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.export(format="onnx")
    - label: Phân loại
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.export(format="onnx")
    - label: Embedding
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        model.export(format="tflite")
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results. Thao tác xuất
        # đặt tên tệp theo tác vụ, ở đây là LibreDINOv2s-sem.onnx.
        model = LibreYOLO("LibreDINOv2s-sem.onnx")
        result = model(SAMPLE_IMAGE)
source_hash: 4256e0a0398e5aaf
---

## Cài đặt

LibreDINOv2 chỉ đăng ký khi đã cài `transformers`, cùng dependency tùy chọn mà RF-DETR cần cho backbone DINOv2, vì vậy mô hình cần cùng extra.

```bash
pip install "libreyolo[rfdetr]"
```

## Dự đoán

LibreYOLO không phát hành checkpoint LibreDINOv2. Thay vì tải tệp, hãy khởi tạo trực tiếp wrapper: `model_path=None` (mặc định) tải backbone `facebook/dinov2-with-registers-small` Apache-2.0 của Meta từ Hugging Face trong lần sử dụng đầu tiên. `task=` chọn nội dung chạy bên trên backbone.

<code-tabs name="predict" />

`task="semantic"` và `task="classify"` thêm dense head hoặc linear head trên backbone; head đó được khởi tạo ngẫu nhiên và chỉ hữu ích sau khi bạn huấn luyện (xem [Huấn luyện](#train)). `task="embed"` bỏ qua mọi head và trả về token CLS được chuẩn hóa cuối cùng của backbone dưới dạng một hàng toàn ảnh trong `result.embeddings`, vì vậy hoàn toàn không cần huấn luyện. `result.boxes` luôn là `None`: không tác vụ nào trong ba tác vụ tạo detection theo từng thực thể. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

`size` chọn độ rộng projector kiểu RF-DETR được xếp trên backbone, không chọn chính backbone: mọi kích thước dùng chung encoder DINOv2-S (small). Phân đoạn ngữ nghĩa chạy ở lưới patch vuông nguyên bản của DINOv2; phân loại và embedding chạy ở độ phân giải phân loại nhỏ hơn dùng để huấn luyện linear probe.

## Huấn luyện

`task="semantic"` và `task="classify"` đều có thể huấn luyện; `task="embed"` không có head phụ thuộc lớp đối tượng để fit và phát sinh `NotImplementedError` nếu bạn gọi `train()` trên tác vụ này.

<code-tabs name="train" />

Các đối số keyword chính ở đây là `batch_size` và `lr`, không phải `batch` và `lr0` như hầu hết các họ mô hình khác; `batch` và `lr0` vẫn được chấp nhận và ánh xạ sang chúng, nhưng truyền cả hai sẽ phát sinh lỗi xung đột. `output_dir=` (mặc định `"runs/train"`) thay thế `project=`/`name=` làm cách chính để đặt vị trí lượt chạy, dù truyền trực tiếp `project=`/`name=` vẫn hoạt động. Xem [huấn luyện](/docs/train) để biết về dataset, tăng cường dữ liệu (data augmentation), multi-GPU và logger.

## Xác thực

`val()` trả về dictionary gồm các key `metrics/`: mIoU và độ chính xác pixel cho `task="semantic"`, độ chính xác top-1 và top-5 cho `task="classify"`. `task="embed"` không có ground truth để chấm điểm và phát sinh `NotImplementedError` nếu bạn gọi `val()` trên tác vụ này.

<code-tabs name="val" />

## Xuất

<export-matrix />

Mỗi tác vụ hỗ trợ một tập con định dạng khác nhau như hiển thị ở trên. Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`. Trang [Xuất](/docs/export) liệt kê các đối số mà mọi định dạng chấp nhận.

<code-tabs name="export" />

## Giấy phép

<provenance-box>

Hàng "Trọng số" bên trên nêu giấy phép áp dụng là Apache-2.0, nhưng thực tế không có gì được phát hành lại dưới tổ chức LibreYOLO trên Hugging Face cho họ mô hình này: LibreYOLO không host checkpoint LibreDINOv2 riêng. Nội dung mà `LibreDINOv2(model_path=None)` tải là repo `facebook/dinov2-with-registers-small` của chính Meta, không qua chỉnh sửa.

</provenance-box>

## Trích dẫn

<citation-block />
