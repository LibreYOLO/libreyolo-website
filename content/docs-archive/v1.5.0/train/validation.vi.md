---
title: Xác thực và metric
seo_title: Xác thực và metric trong LibreYOLO
description: >-
  Chạy val() trên mọi mô hình, đọc các key metric mà từng tác vụ trả về, chọn
  backend đánh giá và bật validation loss bên cạnh metric độ chính xác.
lead: >-
  Xác thực chạy mô hình trên một split của dataset qua val() và trả về
  dictionary phẳng gồm các key metric và giá trị float. Các key là chuỗi
  literal, và tập key nhận được phụ thuộc vào tác vụ chứ không phải family.
keywords:
  - map50-95
  - đánh giá coco
  - metric xác thực
  - faster-coco-eval
  - pycocotools
  - validation loss
  - miou
  - chất lượng panoptic
  - độ chính xác top1
last_verified: 1.5.0
snippets:
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["speed/total_ms"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Trên split khác
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml", split="train", batch=4)

        print(metrics)
  valloss:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, val_loss=True)
  json:
    - label: Ghi dự đoán theo định dạng COCO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
source_hash: d907183492fa3f57
---

## Chạy xác thực

`val()` nhận dataset và trả về các metric.

<code-tabs name="val" />

Giá trị trả về là `dict[str, float]` thuần. Mọi key đều là literal, vì vậy hãy
đọc theo tên thay vì vị trí.

Các đối số chính là `data`, `split`, `batch`, `imgsz`, `conf`, `iou`, `workers`,
`device`, `augment`, `save_json` và `verbose`. `conf` mặc định là `0.001` và
`iou` là `0.6`, đều lỏng hơn nhiều so với giá trị mặc định khi dự đoán vì một
lượt quét mAP cần phần đuôi có độ tin cậy thấp. `imgsz` mặc định theo kích thước
đầu vào riêng của mô hình thay vì một số cố định. `split` nhận `val`, `test`
hoặc `train` và không nhận giá trị nào khác.

Mọi trường khác của cấu hình xác thực được truyền qua dưới dạng đối số keyword,
gồm `save_dir`, `max_det`, `eval_max_det`, `half`, `amp_dtype`, `cache` và
`save_plots`.

## Key metric theo từng tác vụ

Phát hiện trả về nhóm số liệu COCO:

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

Hai key trong đó dễ gây nhầm lẫn. `metrics/precision` và `metrics/recall` là các
alias được giữ để tương thích ngược: chúng mang giá trị mAP 50-95 và AR@100,
không phải một cặp precision và recall. Hãy dùng các key có tên cụ thể.

Phân đoạn thực thể trả về các số liệu mAP và AR ở trên dưới dạng số liệu mặt nạ
trong các key không có hậu tố, với phiên bản hộp dưới hậu tố `(B)` và phiên bản
mặt nạ được lặp lại dưới `(M)`. Với tác vụ này, precision và recall chỉ tồn tại
ở dạng có hậu tố `metrics/precision(B)`/`metrics/recall(B)` và
`metrics/precision(M)`/`metrics/recall(M)`, cả hai cặp mang cùng giá trị alias
như detect: cặp `(B)` là mAP50-95 và AR@100 của hộp, cặp `(M)` là mAP50-95 và
AR@100 của mặt nạ.

| Tác vụ | Key |
|---|---|
| detect | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, cùng phân tích theo kích thước và recall ở trên |
| segment | phiên bản mặt nạ của các key detect ở trên (key không hậu tố là mặt nạ); `precision`/`recall` chỉ có dạng `(B)`/`(M)`, cả hai được alias theo cùng cách |
| pose | `metrics/keypoints_mAP50-95`, `metrics/keypoints_mAP50`, `metrics/keypoints_mAP75`, `metrics/keypoints_mAP_M`, `metrics/keypoints_mAP_L`, và các key `keypoints_AR` tương ứng |
| obb | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, `metrics/precision`, `metrics/recall`, cùng các bản sao có hậu tố `(OBB)` |
| classify | `metrics/accuracy_top1`, `metrics/accuracy_top5` |
| semantic | `metrics/mIoU`, `metrics/pixel_accuracy` |
| panoptic | `metrics/PQ`, `metrics/SQ`, `metrics/RQ`, `metrics/PQ_things`, `metrics/PQ_stuff`, `metrics/categories` |
| depth | `metrics/abs_rel`, `metrics/rmse`, `metrics/delta1`, `metrics/delta2`, `metrics/delta3` |
| normal | `metrics/mean_angular_error`, `metrics/median_angular_error`, `metrics/within_11_25`, `metrics/within_22_5`, `metrics/within_30` |
| edge | `metrics/ODS`, `metrics/OIS`, `metrics/best_threshold` |
| restore | `metrics/PSNR`, `metrics/SSIM` |
| matte | `metrics/MAE`, `metrics/Smeasure` |
| ocr | `metrics/det_precision`, `metrics/det_recall`, `metrics/det_hmean`, `metrics/e2e_precision`, `metrics/e2e_recall`, `metrics/e2e_f1`, `metrics/rec_1-NED` |
| point | `metrics/precision`, `metrics/recall`, `metrics/f1`, `metrics/MLE`, `metrics/MAE`, `metrics/RMSE`, cùng một key quét mAP |

`metrics/precision` và `metrics/recall` của OBB không phải alias: chúng là
precision và recall thực ở IoU 0.50, lấy tại operating point lỏng nhất (mọi dự
đoán vượt qua `conf`, mặc định `0.001`). Các bản sao có hậu tố `(OBB)` lặp lại
cùng bốn giá trị dưới tên theo tác vụ, cùng quy ước với `(B)` và `(M)` ở trên.

`accuracy_top5` thực ra là top-`min(5, num_classes)`, vì vậy trên dataset ba lớp,
nó là top-3 mà mọi sample đều thỏa mãn, nên giá trị là 1.0.

Key quét của tác vụ point được dựng từ các ngưỡng khoảng cách, vì vậy với giá
trị mặc định, nó là `metrics/mAP@[0.01:0.10]` và key một ngưỡng là
`metrics/mAP@0.01`. Truyền `dist_thresholds` sẽ thay đổi cả hai chuỗi.

Phần lớn tác vụ còn trả về key `fitness`, là số đơn mà cơ chế chọn checkpoint
tốt nhất sử dụng theo mặc định. Phát hiện, phân đoạn và OBB không có key này;
family của chúng được chọn theo `metrics/mAP50-95`, vốn có trong dictionary trả
về. Tư thế không trả về `fitness` lẫn `metrics/mAP50-95`; thay vào đó, trainer
của tác vụ đặt `best_metric_key` thành `metrics/keypoints_mAP50-95`.

## Key tốc độ

Mọi validator đều thêm thời gian:

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

Đây là số mili giây trên mỗi ảnh, lấy trung bình trong lượt chạy. Chúng mô tả
máy và cài đặt bạn đã chạy, vì vậy một số liệu lấy từ đây chỉ có ý nghĩa khi
được báo cáo cùng phần cứng, kích thước batch và precision.

## Backend đánh giá

Metric phát hiện và phân đoạn được tính qua COCO evaluator, còn
`faster_coco_eval=True`, giá trị mặc định, chọn backend C++ khi đã cài package
`faster-coco-eval`. Nếu chưa cài, lượt chạy quay về pycocotools với một cảnh báo
cho mỗi process:

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

Backend thực sự chạy được ghi lại trên mô hình dưới dạng `last_eval_backend`, và
CLI báo cáo nó trong đầu ra cho các tác vụ kiểu phát hiện. Đặt
`LIBREYOLO_FASTER_COCO_EVAL` để ghi đè giá trị cấu hình từ môi trường.

`iou_thresholds` chỉ được áp dụng trên đường dẫn OBB. Đường dẫn COCO đánh giá
theo lượt quét cố định riêng từ 0.50 đến 0.95 và bỏ qua giá trị này.

## Validation loss

Theo mặc định, xác thực chỉ báo cáo độ chính xác. `val_loss=True` còn tính
objective huấn luyện của family trên các batch xác thực.

<code-tabs name="valloss" />

Nó phát ra `metrics/loss` cùng một `metrics/loss/<component>` cho mỗi thành phần,
được gán trọng số chính xác như khi huấn luyện, vì vậy các thành phần cộng lại
thành tổng. Qua logger, chúng xuất hiện dưới dạng `val/loss` và
`val/loss/<component>`, còn `libreyolo monitor` đặt `metrics/loss` chồng lên
`train/loss`.

Các thành phần là thành phần riêng của family:

| Tác vụ | Family | Thành phần |
|---|---|---|
| detect | `yolo9`, `yolo9_p2`, `yolo9_e2e` | `box`, `cls`, `dfl` |
| detect | `yolonas` | `cls`, `iou`, `dfl` |
| detect | `rfdetr` | `ce`, `bbox`, `giou` |
| detect | `rtdetr`, `rtdetrv2` | `vfl`, `bbox`, `giou` |
| detect | `dfine` | `vfl`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `domedetr` | `vfl`, `bbox`, `giou`, `fgl`, `ddf`, `defe_density`, `defe_reg` |
| detect | `deim`, `deimv2`, `rtdetrv4`, `ec` | `mal`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `rtmdet` | `cls`, `bbox` |
| detect | `picodet` | `cls`, `bbox`, `dfl` |
| detect | `yolox` | `iou`, `obj`, `cls`, `l1` |
| detect | `yolo7` | `iou`, `obj`, `cls` |
| point | `fomo` | `ce` |
| classify | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` | `ce` |
| semantic | `segformer`, `lingbotvision`, `dinov2` | `sem` |
| restore | `nafnet` | `restore` |

Tùy chọn này mặc định tắt vì việc gán target làm tăng thời gian và bộ nhớ xác
thực. Validator dùng lại đầu ra mô hình đã tạo cho metric độ chính xác thay vì
chạy forward pass thứ hai, chạy trong `no_grad` trên mô hình đánh giá hoặc EMA,
và trong huấn luyện multi-GPU, nó được tính cục bộ trên rank 0 mà không có
collective. Việc chọn checkpoint tốt nhất vẫn dựa trên metric độ chính xác.

Có ba điều được chủ ý không thực hiện. Thứ nhất, nó không bao giờ gồm các thành
phần khử nhiễu tương phản vì chúng cần ground truth tại thời điểm forward, còn
forward xác thực không nhận dữ liệu đó. Thứ hai, nó báo cáo mô hình ở evaluation
mode, vì vậy khi forward train và eval của một family thực sự khác nhau ở
statistic BatchNorm hoặc stochastic depth, con số phản ánh eval mode; đây là
phép so sánh có chủ ý. Thứ ba, tác vụ chưa được family triển khai sẽ phát sinh
lỗi cấu hình khi thiết lập thay vì âm thầm bỏ qua:

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

FOMO là ngoại lệ không thay đổi gì: validator của family này luôn tính loss đó,
và `val_loss=True` chỉ ảnh hưởng tới các key dùng để công bố.

Không thể kết hợp xác thực có augmentation với validation loss; yêu cầu cả hai
sẽ phát sinh lỗi.

## Các tệp được ghi khi xác thực

`val()` luôn ghi `config.yaml` vào thư mục lưu, mặc định là
`runs/val/<model>_<size>_<timestamp>` khi không cung cấp `save_dir`.

<code-tabs name="json" />

`save_json=True` ghi `predictions.json` cho phát hiện, còn phân đoạn ghi
`predictions_bbox.json` cùng `predictions_masks.json`. OBB không hỗ trợ và sẽ
báo rõ.

`save_plots=True` ghi vào thư mục con `plots/`. Phát hiện tạo
`box_metrics.png`, biểu đồ AP và recall theo từng lớp, đường cong
precision-recall và confidence, confusion matrix, cùng ảnh sample đã chú thích
khi cài OpenCV. Phân đoạn thêm bản sao tương ứng cho mặt nạ, còn tư thế có bộ
metric và đường cong riêng. Các validator khác không triển khai biểu đồ; phân
loại, ngữ nghĩa, toàn cảnh, độ sâu, pháp tuyến, cạnh, phục hồi, matting, OCR, OBB
và point đều không ghi gì ở đó. Lỗi vẽ biểu đồ sẽ cảnh báo và không bao giờ dừng
lượt chạy.

## Xác thực trong khi huấn luyện

Quá trình huấn luyện xác thực mỗi `eval_interval` epoch trên split `val` của
dataset, và các metric tạo ra điều khiển việc chọn `best.pt`, early stop theo
`patience` và các key `val/` trong mọi logger. Việc xác thực chạy trên trọng số
EMA khi bật EMA.

Xem [Siêu tham số](/docs/train/hyperparameters) để biết `eval_interval`,
`patience` và `save_plots`, và [Logger thí nghiệm](/docs/train/loggers) để biết
các con số được đưa tới đâu.

## Nội dung liên quan

- Xem [Dataset](/docs/train/datasets) để biết các key split và định dạng mà
  validator đọc.
