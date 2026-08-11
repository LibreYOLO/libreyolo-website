---
title: EdgeCrafter
families:
  - ec
seo_title: 'EdgeCrafter: phát hiện, ước lượng tư thế và phân đoạn trong LibreYOLO'
description: >-
  Dùng EdgeCrafter trong LibreYOLO để phát hiện, ước lượng tư thế và phân đoạn
  thực thể. Cài đặt, dự đoán, xác thực và xuất với mã nguồn dùng giấy phép MIT.
lead: >-
  Một vision transformer nhỏ gọn cho dự đoán dày đặc trên phần cứng biên, được
  upstream công bố dưới dạng ba mô hình cùng họ: ECDet, ECPose và ECSeg.
  LibreYOLO nạp cả ba thành một family, với tác vụ được xác định bởi checkpoint.
keywords:
  - EdgeCrafter
  - ECDet
  - ECPose
  - ECSeg
  - vision transformer nhỏ gọn
  - phát hiện đối tượng
  - ước lượng tư thế
  - phân đoạn thực thể
  - inference trên thiết bị biên
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreECs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Tư thế
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Hậu tố -pose trong tên tệp chọn keypoint head, vì vậy không cần
        # đối số task ở đây.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.conf)
    - label: Phân đoạn thực thể
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=5e-4,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreECs.pt data=my-dataset.yaml epochs=50
        imgsz=640 batch=8 lr0=5e-4
    - label: Tư thế
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Cần dataset keypoint một lớp có data.yaml khai báo kpt_shape,
        # và imgsz bằng kích thước gốc của checkpoint.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="my-pose-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: Phân đoạn thực thể
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Cần nhãn polygon và imgsz bằng kích thước gốc của checkpoint.
        model = LibreYOLO("LibreECs-seg.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            lora=True,
        )
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs.pt data=my-dataset.yaml
    - label: Tư thế
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        metrics = model.val(data="my-pose-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: Phân đoạn thực thể
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # mặt nạ
        print(metrics["metrics/mAP50-95(B)"])   # hộp
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-seg.pt format=onnx imgsz=640
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, vì vậy artifact đã xuất được nạp
        # như mọi checkpoint khác và trả về cùng một đối tượng Results.
        model = LibreYOLO("LibreECs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 39c6975fc16b3ff1
---

## Cài đặt

EdgeCrafter không cần thành phần tùy chọn nào. Mọi nội dung mà mô hình import đều
có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

Tinh chỉnh adapter với `lora=True` là ngoại lệ và cần thành phần bổ sung `lora`.

```bash
pip install "libreyolo[lora]"
```

## Dự đoán

Trọng số được tải từ Hugging Face trong lần sử dụng đầu tiên và lưu vào cache
cục bộ.

<code-tabs name="predict" />

Tác vụ được xác định từ tên tệp, vì vậy checkpoint có `-pose` hoặc `-seg` sẽ tự
chọn head và không nhận đối số task. Cả ba đều trả về đối tượng `Results` giống
mọi family khác, với `result.keypoints` được thêm cho tư thế và `result.masks`
cho phân đoạn. Tác vụ tư thế bao gồm một lớp đối tượng là người, với 17 keypoint
COCO, và số lượng được cố định khi dựng mô hình. Nó không có box head, vì vậy
mỗi hộp tư thế là phạm vi bao của chính các keypoint, còn kênh keypoint thứ ba
là một hằng số thay vì điểm số theo từng điểm.

`conf` và `max_det` lọc quá trình chọn query; `iou` được chấp nhận để giữ tính
tương thích API nhưng không có tác dụng, vì cả ba head đều decode một tập query
không có bước NMS. Xem [dự đoán](/docs/predict) để biết về nguồn, stream và cách
xử lý kết quả.

## Biến thể

Có bốn kích thước. Tất cả đều chạy ở cùng độ phân giải đầu vào, vì vậy bảng
phân biệt chúng theo số lượng tham số và độ chính xác.

<benchmark-table task="detect" />

<va-embed />

Upstream công bố ECDet, ECPose và ECSeg dưới dạng ba mô hình riêng thay vì một mô
hình có ba head. Chúng dùng chung backbone ECViT và hybrid encoder, chỉ khác ở
head, nên LibreYOLO gộp chúng thành một family và để tên tệp checkpoint mang
thông tin tác vụ. Do đó, một chữ cái chỉ kích thước biểu thị cùng backbone và
encoder trên cả ba mô hình, còn các lệnh dự đoán, xác thực và xuất nhận cùng đối
số bất kể bạn nạp mô hình nào.

## Huấn luyện

Cả ba tác vụ đều huấn luyện qua `train()`. Phương thức này đọc tác vụ từ
checkpoint đã nạp và chọn trainer tương ứng.

<code-tabs name="train" />

Những nội dung đã được kiểm tra cho phát hiện và phân đoạn: độ tương đồng
inference với upstream ở mức 1e-5, theo từng lớp và từng kích thước, cùng khả
năng chạy loss và một bước huấn luyện trên đầu vào tổng hợp. Những nội dung chưa
được kiểm tra, theo docstring của chính `train()`: độ hội tụ của một lượt tinh
chỉnh đầy đủ, huấn luyện multi-GPU, bước nạp lại mô hình tốt nhất sau khi dừng
augmentation và ánh xạ lại lớp đối tượng từ Objects365 sang COCO. Đường dẫn tư
thế tuân theo recipe đã công bố của DETRPose, dùng Hungarian matcher trên chi
phí lớp đối tượng, keypoint L1 và OKS với cơ chế khử nhiễu keypoint tương phản;
độ hội tụ đầu cuối của nó cũng chưa được kiểm tra.

Nếu giữ nguyên, trainer chạy 74 epoch ở `lr0=5e-4` và bật mixed precision, theo
recipe upstream: AdamW, lịch flat cosine, EMA ở 0.9999 và đầu vào được chuẩn hóa
theo ImageNet. Cả tư thế lẫn phân đoạn đều yêu cầu `imgsz` bằng kích thước gốc
của checkpoint vì evaluation anchor grid được dựng khi khởi tạo mô hình; giá
trị khác sẽ phát sinh lỗi trước khi lượt chạy bắt đầu. Tư thế còn yêu cầu
dataset một lớp có `data.yaml` khai báo `kpt_shape`, với số lượng keypoint khớp
với head.

`lora=True` chỉ áp dụng cho phát hiện; tư thế và phân đoạn sẽ phát sinh
`ValueError` khi dùng tùy chọn này. Trên Apple silicon, trainer giữ lượt chạy
trên GPU và gửi một phép toán sang CPU, đó là phép lan truyền ngược grid-sample
bên trong deformable attention mà PyTorch chưa triển khai trong Metal.

Xem [huấn luyện](/docs/train) để biết về dataset, tăng cường dữ liệu (data
augmentation), multi-GPU và logger.

## Xác thực

`val()` trả về một dictionary được lập chỉ mục bằng tên metric và in kết quả
theo từng lớp đối tượng khi vẫn bật `verbose`.

<code-tabs name="val" />

Tác vụ tư thế báo cáo metric OKS của keypoint dưới `metrics/keypoints_*`. Tác vụ
phân đoạn báo cáo mặt nạ dưới key `metrics/mAP50-95` thuần và lặp lại cả hai góc
nhìn trong một lượt, hộp dưới `(B)` và mặt nạ dưới `(M)`.

## Xuất

<export-matrix />

Artifact đã xuất được nạp lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như một checkpoint và trả về cùng một đối tượng
`Results`. Tư thế và phân đoạn xuất ở đầu vào cố định 640 x 640 thay vì shape
động, và một số target phát hiện cũng dùng canvas cố định, gồm OpenVINO, Paddle,
MNN, ExecuTorch và Core AI. Phần [Xuất](/docs/export) liệt kê các đối số mà mọi
định dạng chấp nhận và những thành phần bổ sung mà một số định dạng yêu cầu.

<code-tabs name="export" />

## Checkpoint

Tất cả tệp trọng số đã công bố cho family này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />
