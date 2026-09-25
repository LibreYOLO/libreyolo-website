---
title: "Các lựa chọn thay thế YOLOv8 tốt nhất năm 2026"
description: "RF100-VL đo khả năng tinh chỉnh của detector sau khi huấn luyện trên COCO. RF-DETR và YOLO-NAS đạt điểm cao hơn YOLOv8. Cách chạy cả hai trong LibreYOLO."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, YOLOv8 alternative, YOLO8 alternative, RF-DETR, YOLO-NAS, object detection, RF100-VL]
faq:
  - q: "Lựa chọn thay thế YOLOv8 tốt nhất năm 2026 là gì?"
    a: "RF-DETR-S nếu dùng GPU. Mô hình đạt 60.41 mAP50-95 trong chiến dịch RF100-VL của LibreYOLO. Bài báo RF100-VL báo cáo YOLOv8m đạt 56.9. YOLO-NAS-S là lựa chọn convolutional, đạt 58.00."
  - q: "RF-DETR chuyển giao tốt hơn YOLOv8 không?"
    a: "Có, trên RF100-VL. LibreYOLO đo được RF-DETR-S đạt 60.41 và RF-DETR-M đạt 61.13. Roboflow báo cáo lần lượt là 60.2 và 61.2. Bài báo báo cáo YOLOv8m đạt 56.9."
  - q: "Tôi có thể dùng YOLO-NAS cho mục đích thương mại không?"
    a: "Trọng số pretrained của Deci không được dùng cho mục đích thương mại. Kiến trúc dùng giấy phép Apache-2.0. Xem bài viết về YOLO-NAS để biết chi tiết giấy phép."
---

COCO thường là tiêu chí để chọn detector. Một góc nhìn khác là mức độ mô hình tinh chỉnh tốt trên dữ liệu khác. [RF100-VL](https://arxiv.org/abs/2505.20612) đo lường điều đó: một checkpoint được huấn luyện sẵn trên COCO được huấn luyện trong 100 epoch trên từng tập dữ liệu trong số 100 tập dữ liệu, sau đó được chấm điểm trên từng tập test. Con số là mAP50-95 trung bình.

YOLOv8m đạt 56.9 theo [phụ lục bài báo](https://papers.neurips.cc/paper_files/paper/2025/file/1013f8ff40a194f3f12a6bcc5221bb34-Paper-Datasets_and_Benchmarks_Track.pdf). Hai mô hình chúng tôi chạy đạt điểm cao hơn: RF-DETR và YOLO-NAS. Các lần chạy gốc có tại [huggingface.co/datasets/LibreYOLO/rf100-vl-results](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results).

## RF-DETR

RF-DETR-S đạt 60.41 và RF-DETR-M đạt 61.13 trong [chiến dịch của LibreYOLO](/articles/rf100vl-benchmark). [Roboflow báo cáo](https://rfdetr.roboflow.com/latest/learn/benchmarks/) lần lượt là 60.2 và 61.2. Các phiên bản từ Nano đến Large dùng giấy phép Apache-2.0.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=512, batch=8)
```

[Tài liệu](/docs/models/rf-detr) | [Trọng số](https://huggingface.co/LibreYOLO/LibreRFDETRs) | [Lần chạy `20260814-rfdetr-s-1b1190ee`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/rfdetr-s/20260814-rfdetr-s-1b1190ee)

## YOLO-NAS

YOLO-NAS-S đạt 58.00, là lựa chọn convolutional nếu bạn muốn một CNN theo phong cách YOLO. Trọng số do Deci công bố không được dùng cho mục đích thương mại. LibreYOLO không lưu trữ các trọng số này. Chi tiết: [YOLO-NAS vẫn được duy trì](/articles/yolo-nas-with-libreyolo).

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

[Tài liệu](/docs/models/yolo-nas) | [Lần chạy `20260809-yolonas-s-02926964`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/yolonas-s/20260809-yolonas-s-02926964)

Cài đặt bằng `pip install "libreyolo[rfdetr]"` cho RF-DETR, hoặc `pip install libreyolo` cho YOLO-NAS. Trang chiến dịch: [RF100-VL](/articles/rf100vl-benchmark).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Tài liệu](https://www.libreyolo.com/docs)
