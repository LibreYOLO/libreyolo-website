---
title: D-FINE
families:
  - dfine
seo_title: 'D-FINE: tinh chỉnh, xác thực và xuất theo MIT'
description: >-
  Dùng D-FINE trong LibreYOLO để phát hiện đối tượng và phân đoạn thực thể. Cài
  đặt, dự đoán, tinh chỉnh, xác thực và xuất với mã nguồn dùng giấy phép MIT.
lead: >-
  Một detection transformer biểu diễn lại hồi quy box thành phân phối xác suất
  trên từng cạnh của box, được tinh chỉnh qua các lớp decoder. LibreYOLO hỗ trợ
  mô hình này cho phát hiện và phân đoạn thực thể.
keywords:
  - D-FINE
  - detection transformer
  - phát hiện đối tượng thời gian thực
  - phân đoạn thực thể
  - tinh chỉnh phân phối chi tiết
  - DETR
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDFINEn.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Phân đoạn thực thể
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Hậu tố -seg trong tên tệp chọn mask head, vì vậy không cần đối số
        # task ở đây.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDFINEn.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Phân đoạn thực thể
      language: bash
      code: |
        # Tiếp tục từ trọng số phân đoạn đã phát hành, bao gồm cả mask head.
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: Phân đoạn từ trọng số phát hiện
      language: bash
      code: >
        # Trọng số phát hiện không có mask head, vì vậy đây là một transfer rõ
        ràng:

        # head bắt đầu khi chưa được huấn luyện và chỉ hữu ích sau khi huấn
        luyện. Việc

        # chỉ định task=segment ở đây chính là thao tác cho phép transfer.

        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn.pt data=my-dataset.yaml
    - label: Phân đoạn thực thể
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # mặt nạ
        print(metrics["metrics/mAP50-95(B)"])   # box
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDFINEn.pt format=onnx imgsz=640

        libreyolo export model=LibreDFINEn.pt format=tensorrt imgsz=640
        half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreDFINEn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 0216631a26185524
---

## Cài đặt

D-FINE không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

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

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang detector khác chỉ cần thay đổi một dòng. Tên tệp có `-seg` tự phân giải thành tác vụ phân đoạn, và khi đó `result.masks` chứa các mặt nạ (mask) thực thể cùng với các box. `conf` và `max_det` lọc lựa chọn query; `iou` được chấp nhận để giữ tính tương đồng của API nhưng không có tác dụng vì decoder là bộ dự đoán tập hợp không có bước NMS. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có năm kích thước. Tất cả chạy ở cùng độ phân giải đầu vào, vì vậy bảng phân tách chúng theo số lượng tham số và độ chính xác.

<benchmark-table task="detect" />

<va-embed />

Tác vụ phân đoạn dùng lại backbone, encoder và decoder phát hiện rồi bổ sung mask head, vì vậy checkpoint `-seg` nhận cùng đối số như bản phát hiện tương ứng. Họ RT-DETRv4 của LibreYOLO được viết dưới dạng lớp con của wrapper D-FINE: nó kế thừa dòng decoder này rồi cố định danh sách tác vụ trở lại tác vụ phát hiện vì không có mask head.

## Huấn luyện

Quá trình huấn luyện bắt đầu từ checkpoint đã phát hành cho cả hai tác vụ.

<code-tabs name="train" />

Khi giữ nguyên thiết lập, trainer chạy 132 epoch ở `lr0=2e-4` với `amp=False`, batch 16 và early stopping sau 50 epoch không có cải thiện. Trọng số phát hiện là điểm bắt đầu hợp lệ cho huấn luyện phân đoạn, nhưng chỉ dưới dạng transfer rõ ràng vì mask head bắt đầu khi chưa được huấn luyện và nếu không sẽ trả về các mặt nạ vô nghĩa. Truyền `task=segment` cho CLI chính là thao tác cho phép điều này. Tuyến Python hẹp hơn: phải khởi tạo trực tiếp `LibreDFINE` với `allow_detect_to_segment_transfer=True` vì factory `LibreYOLO()` không nhận đối số này, và việc khởi tạo trực tiếp không tải xuống nên tệp trọng số phải có sẵn trên đĩa.

`lora=True` áp dụng cho tác vụ phát hiện. Huấn luyện phân đoạn từ chối tùy chọn này và hướng đến `freeze='backbone'` thay thế vì mask head chưa được kiểm thử với adapter. Trên Apple silicon, trainer chuyển toàn bộ lượt chạy sang CPU: backward pass của phép nhân ma trận theo bin của Integral gặp lỗi biên dịch Metal. Inference trên MPS không bị ảnh hưởng.

Xem [huấn luyện](/docs/train) để biết về dataset, tăng cường dữ liệu (data augmentation), multi-GPU và logger.

## Xác thực

`val()` trả về dictionary được lập key theo tên metric và in kết quả cho từng lớp đối tượng khi giữ bật `verbose`.

<code-tabs name="val" />

Với checkpoint `-seg`, key `metrics/mAP50-95` thuần túy chứa điểm mặt nạ, và cùng lượt chạy đó cũng báo cáo box trong `(B)` và mặt nạ trong `(M)`, vì vậy cả hai đều có sẵn từ một lượt.

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`. Các tuyến OpenVINO, Paddle, MNN và Core AI xuất trên canvas cố định thay vì shape động. Trang [Xuất](/docs/export) liệt kê các đối số được mọi định dạng chấp nhận và các extra mà một số định dạng bổ sung.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box>

Trọng số phân đoạn có upstream thứ hai: mask decoder, phép ghép mask và mask loss đến từ ArgoHA/D-FINE-seg, cũng dùng Apache-2.0, và người bảo trì của dự án đã chấp thuận việc tái sử dụng kèm ghi công.

</provenance-box>

## Trích dẫn

<citation-block />
