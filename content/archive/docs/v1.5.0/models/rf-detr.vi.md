---
title: RF-DETR
families:
  - rfdetr
seo_title: 'RF-DETR: huấn luyện, tinh chỉnh và xuất theo MIT'
description: >-
  Dùng RF-DETR trong LibreYOLO để phát hiện, phân đoạn instance, tư thế và box
  định hướng. Cài đặt, dự đoán, huấn luyện, đánh giá và xuất, tất cả theo giấy
  phép MIT.
lead: >-
  Một detection transformer dự đoán tập đối tượng cố định thay vì lưới dày đặc,
  nên không cần NMS khi suy luận. LibreYOLO hỗ trợ mô hình cho bốn tác vụ.
keywords:
  - RF-DETR
  - detection transformer thời gian thực
  - DETR
  - phát hiện đối tượng
  - phân đoạn instance
  - ước lượng tư thế
  - bounding box định hướng
last_verified: 1.5.0
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: 'LibreRFDETRs, phát hiện trên video ở 512 px.'
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRFDETRs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # Mọi nguồn thư viện chấp nhận: tệp, thư mục, URL, chỉ số webcam,
        # luồng RTSP hoặc danh sách .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # val() trả về dict thông thường, không phải đối tượng
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: Trên COCO
      language: bash
      code: >
        # YAML COCO đi kèm chứa script tải xuống nhúng sẵn, nên cần quyền rõ
        ràng

        # trừ khi tập dữ liệu đã có cục bộ.

        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)


        # Các đối số được mọi định dạng chấp nhận:

        #

        #   format    "onnx" | "torchscript" | "executorch" | "tensorrt"

        #             | "openvino" | "paddle" | "mnn" | "rknn" | "ncnn"

        #             | "tflite" | "coreml" | "coreai".

        #             "engine" is an alias for tensorrt, "litert" for tflite.

        #   imgsz     int hoặc (chiều cao, chiều rộng). Mặc định là

        #             độ phân giải gốc.

        #   batch     int, mặc định 1.

        #   half      bool, xuất ở FP16. Mặc định False.

        #   int8      bool, xuất ở INT8. Mặc định False. Cần `data`.

        #   data      đường dẫn đến YAML tập dữ liệu, dùng để hiệu chuẩn int8.

        #   fraction  float, phần tập hiệu chuẩn cần dùng. Mặc định 1.0.

        #   dynamic   bool, các trục động. Mặc định True.

        #   simplify  bool, chạy đơn giản hóa đồ thị ONNX. Mặc định True.

        #   opset     int, ONNX opset. Được chọn theo họ khi không chỉ định.

        #   device    str, thiết bị dùng để truy vết. Mặc định là thiết bị của
        mô hình.

        #   output_path  str, mặc định là tên suy ra từ checkpoint.

        #   verbose   bool, mặc định False.

        #   allow_download_scripts  bool, mặc định False. Cho phép mã Python

        #             nhúng trong YAML tập dữ liệu cần tải xuống.

        #

        # Một số định dạng nhận thêm đối số riêng, chẳng hạn nền tảng đích RKNN.

        # Các đối số đó được ghi trong trang của từng định dạng.
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512

        libreyolo export model=LibreRFDETRs.pt format=tensorrt imgsz=512
        half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Factory định tuyến theo hậu tố tệp, nên artifact đã xuất được tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreRFDETRs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
    - label: Không dùng LibreYOLO
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Chạy trực tiếp đồ thị nghĩa là bạn tự thực hiện tiền xử lý và hậu xử
        lý.

        # Hãy kiểm tra chữ ký trước khi kết nối.

        session = ort.InferenceSession("LibreRFDETRs.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 512, 512),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 8c464aa759131694
---

## Cài đặt

RF-DETR cần extra riêng, extra này cài thêm `transformers` cho backbone.

```bash
pip install "libreyolo[rfdetr]"
```

## Dự đoán

Trọng số được tải từ Hugging Face ở lần dùng đầu tiên và lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về là loại mà mọi họ đều trả về, nên việc đổi sang
detector khác chỉ cần sửa một dòng. `conf` và `max_det` lọc lựa chọn query;
không có bước NMS cần điều chỉnh. Xem [dự đoán](/docs/predict) để biết về nguồn,
streaming và xử lý kết quả.

## Các biến thể

Bốn kích thước và bốn tác vụ dùng chung một kiến trúc: phân đoạn, tư thế và box
định hướng dùng lại decoder phát hiện với head khác, nên nhận cùng các đối số.
Các kích thước có số tham số tương tự và chủ yếu khác nhau ở độ phân giải đầu vào.

<benchmark-table task="detect" />

<va-embed />

## Huấn luyện

Huấn luyện bắt đầu từ checkpoint đã công bố cho cả bốn tác vụ. RF-DETR liệt kê
`pretrained` trong các đối số mà trình huấn luyện native bỏ qua, nên truyền
`pretrained=False` không tạo mô hình khởi tạo ngẫu nhiên ở đây.

<code-tabs name="train" />

Hai đối số quan trọng hơn ở đây so với detector CNN. Giữ `lr0` ở mức `1e-4` trở
xuống vì detector transformer phân kỳ ở learning rate mà mô hình YOLO vẫn chịu
được. Giữ `imgsz` ở độ phân giải gốc của checkpoint trừ khi có lý do thay đổi.
Đầu vào phải chia hết cho kích thước patch của backbone nhân số cửa sổ; LibreYOLO
kiểm tra điều này trước khi chạy và nêu các kích thước hợp lệ gần nhất.

Xem [huấn luyện](/docs/train) để biết về tập dữ liệu, tăng cường dữ liệu, multi-GPU và logger.

## Đánh giá

`val()` trả về từ điển các khóa `metrics/` bao gồm precision, recall, mAP 50 và
mAP 50-95, được đo trên bất kỳ tập dữ liệu nào theo định dạng bạn đã huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`. Cũng
hỗ trợ chạy đồ thị trong runtime độc lập không cài LibreYOLO, nhưng khi đó bạn
phải tự viết bước tiền xử lý và hậu xử lý.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />
