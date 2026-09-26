---
title: libreyolo val
seo_title: Tham chiếu lệnh libreyolo val
description: >-
  Đánh giá một checkpoint trên một split của tập dữ liệu (dataset) từ dòng lệnh:
  mọi tham số kèm giá trị mặc định, và các khóa metric (chỉ số) mà từng tác vụ
  trả về.
lead: >-
  Đánh giá một mô hình trên một split của dataset rồi in ra các metric. Tập
  metric phụ thuộc vào tác vụ của mô hình, và các con số này chính là thứ dựng
  nên một dòng trong bảng benchmark.
keywords:
  - libreyolo val cli
  - lệnh đánh giá mô hình libreyolo
  - đánh giá yolo bằng cli
  - tính mAP50-95 dòng lệnh
  - tham số libreyolo val
last_verified: 1.5.0
meta:
  - label: Lệnh
    value: libreyolo val
    mono: true
  - label: Bắt buộc
    value: 'model, data'
    mono: true
  - label: Đầu ra
    value: >-
      Metric trên stdout. Biểu đồ và COCO JSON nằm trong runs/val/exp khi được
      yêu cầu
snippets:
  examples:
    - label: Cơ bản
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Biểu đồ và COCO JSON
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml \
          imgsz=640 batch=8 save_json=true save_plots=true \
          project=runs/val name=yolo9s-coco8 exist_ok=true
    - label: Máy đọc được
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml json=true quiet=true
source_hash: f6507840568c3725
---

## Cú pháp

```bash
libreyolo val model=<name|path> data=<dataset.yaml> [key=value ...]
```

Tham số là các cặp `key=value`, và dạng POSIX cũng dùng được, nên `batch=8` và
`--batch 8` là cùng một tham số.

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `model` | | Đường dẫn trọng số mô hình hoặc tên CLI. Bắt buộc |
| `data` | | Đường dẫn tới YAML của dataset (định dạng YOLO, ví dụ `coco8.yaml`). Bắt buộc |
| `data_dir` | | Thư mục dataset trực tiếp, bỏ qua đường dẫn ghi trong YAML |
| `split` | `val` | Split của dataset: `val`, `test`, `train` |
| `batch` | `16` | Kích thước batch |
| `imgsz` | | Kích thước ảnh: `640` (vuông) hoặc `480x640` (HxW). Dùng kích thước đầu vào của chính mô hình khi không đặt |
| `conf` | `0.001` | Ngưỡng độ tin cậy |
| `iou` | `0.6` | Ngưỡng IoU của NMS |
| `max_det` | `300` | Số dự đoán tối đa mỗi ảnh sau NMS |
| `eval_max_det` | | Giới hạn của bộ đánh giá COCO. Theo quy ước AP@100 của pycocotools khi không đặt |
| `faster_coco_eval` | `true` | Dùng backend C++ faster-coco-eval cho các metric COCO khi đã cài; nếu không thì quay về pycocotools |
| `half` | `false` | Suy luận (inference) FP16 |
| `amp_dtype` | `float16` | Kiểu dữ liệu autocast CUDA khi `half=true`: `float16` hoặc `bfloat16` |
| `save_json` | `false` | Lưu kết quả JSON theo định dạng COCO |
| `save_plots` | `false` | Lưu biểu đồ đánh giá: metric, AP theo từng lớp đối tượng, ma trận nhầm lẫn, ảnh mẫu |
| `workers` | `4` | Số worker của dataloader |
| `device` | `auto` | Thiết bị |
| `project` | `runs/val` | Thư mục gốc chứa đầu ra |
| `name` | `exp` | Tên thí nghiệm |
| `exist_ok` | `false` | Dùng lại thư mục đầu ra |
| `allow_download_scripts` | `false` | Cho phép mã Python nhúng trong khối download của YAML dataset |
| `json` | `false` | Xuất JSON ra stdout |
| `quiet` | `false` | Chặn stderr |
| `verbose` | `true` | Đầu ra chi tiết |
| `help_json` | `false` | In schema của lệnh dưới dạng JSON rồi thoát |

## Ví dụ

<code-tabs name="examples" />

## Ghi chú

### Các metric là gì

Tập metric được in ra phụ thuộc vào tác vụ của mô hình, và đầu ra JSON dùng
đúng các khóa đó.

Phát hiện đối tượng, phân đoạn và hộp xoay (OBB) báo cáo `mAP50`, `mAP50_95`,
`precision` và `recall`. Khi một mô hình dự đoán nhiều hơn một loại đầu ra, các
nhóm theo từng loại xuất hiện kèm theo dưới dạng `box_metrics`, `mask_metrics`
và `obb_metrics`, mỗi nhóm mang đúng bốn khóa đó.

Phân loại báo cáo `accuracy_top1` và `accuracy_top5`. Phát hiện điểm báo cáo
`precision`, `recall`, `f1`, `MLE`, `MAE`, `RMSE` và `mAP_sweep`. Ước lượng độ
sâu báo cáo `abs_rel`, `rmse`, `delta1`, `delta2` và `delta3`. Phân đoạn ngữ
nghĩa báo cáo `mIoU` và `pixel_accuracy`. Phục hồi ảnh báo cáo `PSNR` và
`SSIM`.

Kết quả JSON còn mang theo `eval_backend`, cho biết thư viện đánh giá COCO và
phiên bản đã tạo ra các con số, nhờ đó có thể so sánh hai lần chạy mà biết rõ
liệu cùng một backend có chấm cả hai hay không.

### Ngưỡng

Các giá trị mặc định ở đây là mặc định cho đánh giá, không phải mặc định cho dự
đoán: `conf` là `0.001` và `iou` là `0.6`, trong khi
[`libreyolo predict`](/docs/cli/predict) dùng `0.25` và `0.45`. Nâng `conf` lên
mức ngưỡng hiển thị sẽ làm giảm recall và kéo theo mAP, nên một con số tạo ra
theo cách đó không so sánh được với con số đã công bố.

`imgsz` mặc định không được đặt, nghĩa là dùng kích thước đầu vào của chính mô
hình. Đặt giá trị này sẽ đánh giá ở kích thước được chỉ định, và đó là cách một
checkpoint được đo ở ngoài độ phân giải gốc của nó.

### Dataset tự tải về

Một YAML dataset có trường `download` là URL sẽ tải về ở lần dùng đầu tiên mà
không cần thêm quyền gì. Loại mang theo script Python tải về nhúng sẵn thì cần
`allow_download_scripts=true`, và lệnh sẽ cảnh báo trên stderr rằng việc thực
thi mã cục bộ đã được bật. Hai tệp `coco8.yaml` và `coco128.yaml` đi kèm đều
dựa trên URL, nên chúng không cần gì cả.

### Đầu ra và mã thoát

stdout mang các metric; tiến trình đi ra stderr. `json=true` in ra một đối
tượng duy nhất kèm `schema_version`, còn `quiet=true` làm im stderr.

Mã thoát là `0` khi thành công, `2` khi dùng sai hoặc lỗi cấu hình, `3` khi
không tìm thấy dataset, `4` khi không tải được mô hình, và `1` cho các lỗi
runtime khác.

Liên quan: [`libreyolo train`](/docs/cli/train), lệnh chạy đúng quy trình đánh
giá này theo lịch riêng của nó qua `eval_interval`.
