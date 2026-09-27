---
title: HRNet
families:
  - hrnet
seo_title: 'HRNet: ước lượng tư thế top-down trong LibreYOLO'
description: >-
  Dùng HRNet trong LibreYOLO để ước lượng tư thế COCO-17 top-down. Cài đặt, dự
  đoán, xác thực và xuất các checkpoint W32 và W48 dùng giấy phép MIT.
lead: >-
  HRNet là mạng tích chập duy trì luồng đặc trưng độ phân giải cao qua phép hợp
  nhất đa tỷ lệ lặp lại, thay vì khôi phục độ phân giải sau khi downsample.
  LibreYOLO bọc biến thể tư thế top-down chính thức để inference và xác thực.
keywords:
  - HRNet
  - ước lượng tư thế người
  - tư thế top-down
  - keypoint COCO-17
  - mạng độ phân giải cao
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Khi không cung cấp nguồn người: HRNet tự động ghép với detector
        # LibreYOLO9t nhẹ và ghi log lựa chọn đó một lần.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreHRNetw32-pose.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Nguồn người
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        # Bỏ qua hoàn toàn phát hiện: coi toàn bộ ảnh là một người.
        result = model(SAMPLE_IMAGE, cropped=True)

        # Hoặc đưa cho HRNet các box từ detector bạn đã chạy.
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        # Hoặc ghép với detector LibreYOLO cụ thể thay cho
        # LibreYOLO9t mặc định.
        result = model(SAMPLE_IMAGE, person_detector="rfdetr")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreHRNetw32-pose.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreHRNetw32-pose.pt format=onnx
    - label: Dùng tệp đã xuất
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Graph đã xuất chỉ là heatmap head trên canvas cố định: nhận một batch

        # crop người đã cắt, đã chuẩn hóa và trả về heatmap thô. Phát hiện
        người,

        # hình học crop, giải mã heatmap và loại bỏ bằng OKS không thuộc graph
        này;

        # chạy ngoài LibreYOLO nghĩa là bạn phải tự triển khai lại bước giải mã
        đó.

        session = ort.InferenceSession("LibreHRNetw32-pose.onnx")

        name = session.get_inputs()[0].name

        heatmaps = session.run(
            None, {name: np.zeros((1, 3, 256, 192), dtype=np.float32)}
        )[0]
source_hash: 5a5540fd54ee6f23
---

## Cài đặt

HRNet không cần extra ngoài package cơ sở.

```bash
pip install libreyolo
```

Detector người mặc định là checkpoint LibreYOLO9t nhẹ, được tự động tải xuống lần đầu HRNet ghép với nó.

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

HRNet là mô hình ước lượng tư thế top-down: mô hình cần box người trước khi pose head có thể chạy, vì vậy mỗi lệnh gọi đều phân giải một box. Khi giữ nguyên thiết lập, mô hình tự ghép với detector LibreYOLO9t trong lần đầu và ghi log lựa chọn đó. `cropped=True` bỏ qua phát hiện và coi toàn ảnh là một người; `person_boxes` nhận các box từ detector bạn đã chạy; `person_detector` nhận `"auto"`, `"rfdetr"`, mọi mô hình phát hiện LibreYOLO hoặc callable thuần túy. `flip_test=True` cũng chạy mô hình trên crop lật ngang rồi lấy trung bình hai heatmap, đây là tăng cường khi kiểm thử riêng của HRNet; `augment=True` chung không được định nghĩa ở đây. Nguồn nhiều ảnh chạy tuần tự: detector của HRNet và số người thay đổi theo ảnh không hỗ trợ dự đoán xếp chồng. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có hai kích thước `w32` và `w48`, cả hai dự đoán tập keypoint COCO-17 tiêu chuẩn từ crop người có độ phân giải cố định; `w48` là backbone rộng hơn.

Model zoo upstream báo cáo độ chính xác tư thế cho mỗi kích thước bằng detector người riêng, thiết lập kiểm thử lật riêng và giao thức đánh giá COCO chính thức. Cặp ghép mặc định của LibreYOLO dùng detector khác, vì vậy lượt xác thực ở đây đo tổ hợp đó chứ không phải tổ hợp upstream; để khớp số liệu upstream cần cùng box người, điểm detector và thiết lập lật mà đánh giá gốc sử dụng.

## Xác thực

`val()` chạy keypoint OKS-AP kiểu COCO và chấp nhận `data.yaml` YOLO-pose hoặc JSON keypoint COCO cùng thư mục ảnh. Backend metric mặc định là faster-coco-eval, tự động dùng `pycocotools` khi chưa cài faster-coco-eval; `faster_coco_eval=False` buộc dùng tuyến `pycocotools`.

<code-tabs name="val" />

Quá trình xác thực điều khiển `predict()` riêng của HRNet ở bên trong, vì vậy sử dụng detector người mà mô hình được xây dựng hoặc gọi cùng. Hãy khởi tạo mô hình với `person_detector=` rõ ràng để giữ nguồn đó cố định giữa các lượt chạy, thay vì để mỗi lệnh gọi phân giải lại giá trị mặc định.

## Xuất

<export-matrix />

Hợp đồng xuất của HRNet chỉ bao quát ONNX, TorchScript, OpenVINO và TensorRT; mọi định dạng khác phát sinh lỗi trước khi bắt đầu trace. Mỗi bản xuất chỉ là heatmap head canvas cố định, batch một FP32, nhận crop người và trả về heatmap thô: hình học crop affine phía trước cùng bước giải mã heatmap, khôi phục lật và loại bỏ OKS phía sau vẫn nằm trong Python, vì vậy pipeline hoàn chỉnh từ ảnh đến keypoint vẫn cần LibreYOLO ở đầu kia.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />


