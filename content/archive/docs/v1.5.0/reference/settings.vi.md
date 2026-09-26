---
title: Thiết lập
seo_title: Biến môi trường và thư mục LibreYOLO
description: >-
  Mọi biến môi trường LibreYOLO đọc, các thư mục nó ghi, token cần thiết và công
  tắc thay đổi code path được chạy.
lead: >-
  LibreYOLO không có file cấu hình. Hành vi không phải đối số hàm được điều
  khiển bằng biến môi trường và một số ít thư mục quy ước, tất cả được liệt kê
  tại đây.
keywords:
  - LIBREYOLO_DATASETS_DIR
  - LIBREYOLO_KERNELS
  - LIBREYOLO_FASTER_COCO_EVAL
  - HF_TOKEN
  - thư mục trọng số libreyolo
  - cache libreyolo
last_verified: 1.5.0
verification: >-
  Các biến được tìm bằng cách tra libreyolo/**/*.py cho os.environ và os.getenv
  ở v1.5.0; ngữ nghĩa được đọc tại từng nơi sử dụng. Quy ước thư mục được đọc từ
  libreyolo/data/utils.py, libreyolo/utils/download.py,
  libreyolo/export/exporter.py, libreyolo/models/base/model.py và
  libreyolo/models/sam3dbody/mhr_body.py.
snippets:
  usage:
    - label: Trỏ thư mục gốc dataset sang vị trí khác
      language: bash
      code: |
        export LIBREYOLO_DATASETS_DIR=/data/datasets
        python -c "from libreyolo.data import DATASETS_DIR; print(DATASETS_DIR)"
    - label: Đọc giá trị đã phân giải từ Python
      language: python
      code: |
        from libreyolo.data import DATASETS_DIR

        # Mặc định là ~/datasets; LIBREYOLO_DATASETS_DIR override lúc import.
        print(DATASETS_DIR)
source_hash: 462f1288582225ce
---

## Biến môi trường

| Biến | Mặc định | Tác dụng |
|---|---|---|
| `LIBREYOLO_DATASETS_DIR` | `~/datasets` | Thư mục gốc dataset. Đọc một lần lúc import vào `libreyolo.data.DATASETS_DIR` |
| `LIBREYOLO_FASTER_COCO_EVAL` | chưa đặt | Override cờ xác thực `faster_coco_eval`. `1`, `true`, `yes` hoặc `on` ép bật backend nhanh hơn, mọi giá trị khác ép tắt, không đặt thì dùng cờ cấu hình |
| `LIBREYOLO_KERNELS` | chưa đặt | Chọn kernel. `off` hoặc `reference` ép dùng bản triển khai tham chiếu; mọi giá trị khác chỉ chọn bản triển khai được đăng ký dưới tên đó |
| `LIBREYOLO_QUANT_KERNELS` | chưa đặt | Alias cũ cho `LIBREYOLO_KERNELS`, chỉ đọc khi biến kia chưa đặt |
| `LIBREYOLO_HUB_KERNELS` | chưa đặt | `0`, `false`, `off` hoặc `no` tắt việc nạp kernel Hugging Face Hub. Mọi giá trị khác, kể cả chưa đặt, giữ trạng thái bật |
| `LIBREYOLO_MHR_PATH` | `~/.cache/libreyolo/mhr/mhr_model.pt` | Vị trí mô hình cơ thể MHR dùng bởi tác vụ `mesh` |
| `LIBRELABEL_ENABLE_LOCATE` | chưa đặt | Phải đúng bằng `1`, `true`, `yes` hoặc `on` để hiển thị trợ lý LocateAnything trong công cụ gán nhãn. Giá trị khác giữ trạng thái tắt |
| `SAM_3D_BODY_PATH` | chưa đặt | Đường dẫn đến package SAM 3D Body cho họ mesh khi không truyền vào constructor |
| `HF_TOKEN` | chưa đặt | Access token Hugging Face dùng cho repo giới hạn truy cập |

<code-tabs name="usage" />

`LIBREYOLO_DATASETS_DIR` được đọc lúc import, nên đặt sau khi import
`libreyolo.data` không ảnh hưởng đến `DATASETS_DIR`.

Hub kernel có cơ chế opt-in hai phần. Runtime chỉ lấy kernel khi package tùy
chọn `kernels` được cài, vì vậy cài `libreyolo[hub-kernels]` là opt-in và
`LIBREYOLO_HUB_KERNELS=0` là opt-out. Bản cài không có gói bổ sung không bị ảnh
hưởng theo cả hai hướng.

Lựa chọn kernel còn short-circuit import: khi `LIBREYOLO_KERNELS` ép `off` hoặc
`reference`, provider tăng tốc trong cây mã không được import. Registry do ba
biến này điều khiển được mô tả tại [kernel](/docs/reference/kernels).

## Các biến do thư viện đặt

Các biến này được ghi thay vì đọc, nên tự đặt chúng không phải cách được hỗ trợ.

| Biến | Thành phần đặt |
|---|---|
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | Helper spawn DDP, một giá trị cho mỗi tiến trình worker |
| `CUDA_VISIBLE_DEVICES` | Tạm thời thu hẹp trong thiết lập phân tán, sau đó khôi phục |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Được trainer EC đặt thành `1` bằng `setdefault`, nên giá trị có sẵn được ưu tiên |
| `MOMENTUM_ENABLED` | Được loader họ mesh đặt bằng `setdefault` |

`LOCAL_RANK` đồng thời là tín hiệu chế độ phân tán: mã huấn luyện phát hiện
đang chạy dưới DDP bằng sự hiện diện của biến này trong môi trường.

## Biến logger

Các logger huấn luyện tùy chọn dùng giá trị mặc định từ môi trường cho tên dự án.

| Biến | Mặc định | Thành phần sử dụng |
|---|---|---|
| `WANDB_PROJECT` | `libreyolo` | Logger Weights and Biases khi không truyền project |
| `COMET_PROJECT_NAME` | `libreyolo` | Logger Comet khi không truyền project |

Xác thực cho các dịch vụ này theo công cụ riêng, không theo LibreYOLO.

## Token

`HF_TOKEN` là access token Hugging Face. Khi biến chưa đặt, token được đọc từ
`~/.cache/huggingface/token`, vị trí lệnh đăng nhập Hugging Face CLI ghi vào.
Cả hai cách đều hoạt động.

Token chỉ cần cho repo giới hạn truy cập. SAM 3 là ví dụ đi kèm: trọng số được
tải từ repo giới hạn theo giấy phép tùy chỉnh, nên bạn phải chấp nhận điều khoản
trên trang repo và phiên làm việc phải được xác thực.

## Thư mục

| Đường dẫn | Nội dung |
|---|---|
| `weights/` | Checkpoint đã tải, snapshot Hugging Face đã tải và artifact đã xuất |
| `~/datasets` | Thư mục gốc dataset, trừ khi `LIBREYOLO_DATASETS_DIR` quy định khác |
| `~/.cache/huggingface/token` | Token Hugging Face khi không nằm trong `HF_TOKEN` |
| `~/.cache/libreyolo/mhr/mhr_model.pt` | Mô hình cơ thể MHR, trừ khi `LIBREYOLO_MHR_PATH` quy định khác |
| `runs/track/` | Đầu ra mặc định cho `model.track(save=True)` |

`weights/` tương đối với thư mục làm việc. Tên file thuần được phân giải qua
đó, nên `LibreYOLO("LibreYOLO9t.pt")` tìm `weights/LibreYOLO9t.pt` và tải xuống
đó khi không có. `model.export()` ghi vào cùng thư mục nếu không cung cấp
`output_path`. Các cấp sibling tải snapshot nhiều file vào
`weights/<Prefix><size>/`.

## Hành vi tải xuống

Việc tải trọng số được thử lại ba lần với backoff, tiếp tục từ file dở dang và
được bảo vệ bằng lock file để hai tiến trình không lấy cùng checkpoint một lúc.
Họ lấy từ host bên thứ ba có thể cố định checksum và thất bại an toàn khi không
khớp.

Một số lần tải in thông báo giấy phép trước khi bắt đầu. Thông báo thuộc pipeline
tải xuống và không thể tắt bằng cấu hình.

## Backend xác thực

`model.val()` mặc định nhận `faster_coco_eval=True` và quay về pycocotools khi
package chưa được cài, đồng thời cảnh báo một lần. Đặt
`LIBREYOLO_FASTER_COCO_EVAL` override cờ theo từng lệnh gọi, phù hợp cho harness
benchmark không thể thay cấu hình theo lần chạy. Backend thực sự chạy được báo
trong `model.last_eval_backend`.

## Script tải dataset

YAML dataset có thể chứa trường `download` với mã Python. Mã này không được
thực thi trừ khi truyền `allow_download_scripts=True` cho lệnh đọc file, đây là
đối số hàm trên `val()` và `export()` chứ không phải biến môi trường.
