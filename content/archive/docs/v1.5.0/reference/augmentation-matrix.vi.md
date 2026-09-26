---
title: Ma trận tăng cường dữ liệu
seo_title: Họ LibreYOLO nào hỗ trợ phép tăng cường dữ liệu nào
description: >-
  Hỗ trợ điều khiển tăng cường dữ liệu theo từng họ: mười sáu điều khiển
  TrainConfig, ba trạng thái, sáu kiểu pipeline và các điều khiển mà một họ âm
  thầm bỏ qua.
lead: >-
  Việc đặt điều khiển tăng cường dữ liệu không bảo đảm nó đến được pipeline.
  Trang này ghi cách từng họ có thể huấn luyện xử lý mỗi điều khiển trong
  TrainConfig, dùng bảng khai báo do thư viện phân phối làm nguồn chân lý duy
  nhất.
keywords:
  - tăng cường dữ liệu libreyolo
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - no_aug_epochs
  - ma trận hỗ trợ tăng cường dữ liệu
  - điều khiển TrainConfig
last_verified: 1.5.0
verification: >-
  Danh sách điều khiển, trạng thái, kiểu pipeline, sai khác theo họ và hàm trợ
  giúp được đọc từ libreyolo/data/augment/spec.py ở v1.5.0. Bảng đó được gắn với
  pipeline thực bằng tests/unit/test_augment_spec.py.
snippets:
  usage:
    - label: Truy vấn spec trực tiếp
      language: python
      code: |
        from libreyolo.data.augment.spec import (
            AUG_KNOBS,
            aug_support,
            ignored_aug_params,
            uses_mosaic_gating,
        )

        print(sorted(AUG_KNOBS))

        table = aug_support("yolo9")
        print(table["mixup_prob"].status, table["mixup_prob"].note)

        print(sorted(ignored_aug_params("dfine")))
        print(uses_mosaic_gating("yolo9"), uses_mosaic_gating("yolonas"))
source_hash: d2e1b9f5c81072e1
---

## Các điều khiển

Đây là tên trường `TrainConfig`, không phải cách viết CLI. CLI ánh xạ các bí danh
riêng vào chúng, nên `--mosaic` đặt `mosaic_prob`.

| Điều khiển | Ý nghĩa |
|---|---|
| `mosaic_prob` | Xác suất tạo mẫu mosaic 4 ảnh |
| `mixup_prob` | Xác suất trộn mẫu thứ hai |
| `hsv_prob` | Xác suất nhiễu màu HSV |
| `flip_prob` | Xác suất lật ngang |
| `degrees` | Phạm vi xoay ngẫu nhiên cho biến đổi affine, theo độ |
| `translate` | Tỉ lệ tịnh tiến ngẫu nhiên cho biến đổi affine |
| `mosaic_scale` | Phạm vi scale ngẫu nhiên cho biến đổi affine |
| `mixup_scale` | Phạm vi nhiễu scale áp dụng cho ảnh đối tác MixUp |
| `shear` | Phạm vi shear ngẫu nhiên cho biến đổi affine, theo độ |
| `perspective` | Độ lớn biến đổi phối cảnh cho biến đổi affine |
| `flipud` | Xác suất lật dọc |
| `no_aug_epochs` | Các epoch cuối được huấn luyện khi tắt tăng cường dữ liệu mạnh |
| `auto_augment` | Chính sách AutoAugment phân loại: randaugment, autoaugment hoặc augmix |
| `erasing` | Xác suất RandomErasing phân loại |
| `mixup` | Xác suất batch-MixUp phân loại với nhãn mềm |
| `cutmix` | Xác suất batch-CutMix phân loại với nhãn mềm |

Bốn mục cuối là gói phân loại. Các họ phát hiện bỏ qua chúng. `mixup` là điều
khiển chỉ có trong API: `--mixup` của CLI là bí danh cho `mixup_prob` phát hiện.

<code-tabs name="usage" />

## Ba trạng thái

| Trạng thái | Ý nghĩa |
|---|---|
| `used` | Điều khiển đến pipeline huấn luyện của họ và thay đổi mẫu |
| `gated_by_mosaic` | Điều khiển chỉ áp dụng cho mẫu đi qua nhánh mosaic, nên không bao giờ kích hoạt khi `mosaic_prob == 0` |
| `ignored` | Điều khiển không bao giờ đến pipeline; việc đặt nó không làm gì |

`ignored` là trạng thái đáng kiểm tra trước khi chạy vì không có gì thất bại.
CLI cảnh báo khi tham số huấn luyện được đặt rõ nhưng họ đã chọn bỏ qua, còn trình
huấn luyện cảnh báo khi `mixup_prob > 0` không thể kích hoạt vì họ giới hạn MixUp
trên mosaic và `mosaic_prob` bằng 0.

## Các kiểu pipeline

Mọi họ được bao quát tuân theo một trong sáu pipeline, với một số sai khác theo họ được liệt kê bên dưới.

| Điều khiển | Kiểu YOLOX | YOLO-NAS | Kiểu DETR | Phân loại | Semantic | Khôi phục |
|---|---|---|---|---|---|---|
| `mosaic_prob` | dùng | bỏ qua | bỏ qua | bỏ qua | bỏ qua | bỏ qua |
| `mixup_prob` | có điều kiện | dùng | bỏ qua | bỏ qua | bỏ qua | bỏ qua |
| `hsv_prob` | dùng | dùng | bỏ qua | bỏ qua | bỏ qua | bỏ qua |
| `flip_prob` | dùng | dùng | dùng | bỏ qua | bỏ qua | bỏ qua |
| `degrees` | có điều kiện | dùng | bỏ qua | bỏ qua | bỏ qua | bỏ qua |
| `translate` | có điều kiện | dùng | bỏ qua | bỏ qua | bỏ qua | bỏ qua |
| `mosaic_scale` | có điều kiện | dùng | bỏ qua | bỏ qua | bỏ qua | bỏ qua |
| `mixup_scale` | có điều kiện | dùng | bỏ qua | bỏ qua | bỏ qua | bỏ qua |
| `shear` | có điều kiện | dùng | bỏ qua | bỏ qua | bỏ qua | bỏ qua |
| `perspective` | có điều kiện | dùng | bỏ qua | bỏ qua | bỏ qua | bỏ qua |
| `flipud` | dùng | dùng | bỏ qua | bỏ qua | bỏ qua | bỏ qua |
| `no_aug_epochs` | dùng | dùng | dùng | dùng | dùng | dùng |
| `auto_augment` | bỏ qua | bỏ qua | bỏ qua | dùng | bỏ qua | bỏ qua |
| `erasing` | bỏ qua | bỏ qua | bỏ qua | dùng | bỏ qua | bỏ qua |
| `mixup` | bỏ qua | bỏ qua | bỏ qua | dùng | bỏ qua | bỏ qua |
| `cutmix` | bỏ qua | bỏ qua | bỏ qua | dùng | bỏ qua | bỏ qua |

Trong pipeline kiểu YOLOX, tiền xử lý theo mẫu áp dụng nhiễu HSV và lật, còn biến
đổi affine cùng MixUp chỉ chạy trong nhánh mosaic. YOLO-NAS thay vào đó chạy
affine theo mẫu luôn bật, bỏ qua mosaic và áp dụng MixUp độc lập, dùng lại
`mosaic_scale` làm phạm vi scale affine.

Pipeline kiểu DETR là biến đổi passthrough không có mosaic. Biến dạng quang học,
zoom-out và crop theo IoU là hằng số công thức thay vì điều khiển có thể cấu hình,
vì vậy `hsv_prob` và các điều khiển hình học không bao giờ đến được pipeline.
Pipeline phân loại dùng biến đổi ImageFolder có xác suất lật ngang cố định 0.5
thay vì `flip_prob`. Nhiễu scale và HSV semantic đến từ thuộc tính lớp của họ
thay vì điều khiển cấu hình, còn phép lật khôi phục là thao tác đầu vào-và-đích
được ghép với xác suất cố định 0.5.

`no_aug_epochs` được tuân theo ở mọi nơi dù phần bị tắt khác nhau: mosaic và
MixUp cho kiểu YOLOX, affine và MixUp cho YOLO-NAS, tăng cường quang học cùng crop
mạnh và phần đuôi learning rate cho kiểu DETR, phần đuôi scheduler cho phần còn lại.

## Các họ theo kiểu

| Kiểu | Các họ |
|---|---|
| YOLOX-style | `yolox`, `yolo7`, `yolo9`, `yolo9_e2e`, `yolo9_p2`, `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2`, `fomo` |
| YOLO-NAS | `yolonas` |
| DETR-style | `dfine`, `domedetr`, `deim`, `deimv2`, `rtdetrv4`, `rfdetr`, `ec`, `dinov2` |
| Phân loại | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` |
| Semantic | `segformer` |
| Khôi phục | `nafnet` |

Hai mươi lăm họ được bao quát. Họ ngoài danh sách trả về tập ignored rỗng, nên không phát cảnh báo.

## Sai khác

| Họ | Khác biệt so với kiểu |
|---|---|
| `rtmdet` | Bỏ qua `flipud`: biến đổi không có lật dọc |
| `picodet` | Bỏ qua `flipud` |
| `rtdetr` | Bỏ qua `flipud` |
| `rtdetrv2` | Bỏ qua `flipud` |
| `fomo` | Bỏ qua `perspective` và `flipud` |
| `ec` | Dùng `hsv_prob`, `degrees` và `translate` chỉ cho `task="pose"`; detect và segment dùng công thức quang học cố định |
| `dinov2` | Dùng gói phân loại chỉ cho `task="classify"` |

`ec` và `dinov2` là họ đa tác vụ, nên một điều khiển chỉ được đánh dấu ignored
khi mọi tác vụ có thể huấn luyện của họ đều bỏ qua. Điều này giúp cảnh báo CLI
không bao giờ sai cho một tác vụ trong khi đúng cho tác vụ khác.

Dome-DETR kế thừa nguyên các biến đổi của D-FINE. Điều duy nhất mô hình không
nhận được là huấn luyện multi-scale, bị cấu hình tắt thay vì spec tăng cường dữ liệu.

## Điều khiển riêng theo họ

Một số họ chứa điều khiển tăng cường dữ liệu trên lớp con `TrainConfig` riêng
thay vì lớp cơ sở. CLI không cung cấp các mục này; hãy đặt qua Python API.

| Họ | Điều khiển | Ý nghĩa |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Xác suất tăng cường copy-paste instance, chỉ `task="segment"` |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Nguồn copy-paste: `flip` phản chiếu cùng mẫu, `mixup` dùng mẫu thứ hai |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Xác suất xoay ngẫu nhiên 90 độ |
| `rfdetr` | `copy_paste` | Xác suất copy-paste cho `task="segment"`, chỉ chế độ `flip` |
| `rfdetr` | `copy_paste_mode` | Chế độ nguồn copy-paste cho `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Xác suất crop-resize ngẫu nhiên trong pipeline native |
| `dfine` | `crop_resize_prob` | Xác suất crop-resize ngẫu nhiên, `task="segment"` |
| `ec` | `crop_resize_prob` | Xác suất crop-resize ngẫu nhiên, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Xác suất nhiễu độ sáng và tương phản, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Xác suất affine nhận biết keypoint, `task="pose"` |

`rot90` áp dụng cho detect và OBB trên `yolo9`.

## Truy vấn spec

| Hàm trợ giúp | Trả về |
|---|---|
| `aug_support(family)` | Bảng điều khiển đến `Support` hoặc `None` cho họ không xác định |
| `ignored_aug_params(family)` | Tập tên điều khiển bị họ bỏ qua; rỗng cho họ không xác định |
| `uses_mosaic_gating(family)` | MixUp của họ có chỉ kích hoạt trên mẫu mosaic hay không |
| `display_name(family)` | Tên họ hướng đến người dùng được dùng trong cảnh báo |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | Văn bản cảnh báo khi MixUp không bao giờ kích hoạt, nếu không là `None` |

`Support` là named tuple gồm `status` và `note`, trong đó ghi chú giải thích vì sao điều khiển bị bỏ qua hoặc có điều kiện đối với họ đó.

## Điều kiện mosaic

Với họ kiểu YOLOX, `mixup_prob=0.5` cùng `mosaic_prob=0` tắt hoàn toàn MixUp vì
MixUp chỉ áp dụng cho mẫu mosaic. Tổ hợp đó dễ xuất hiện khi tắt mosaic ở cuối
quá trình huấn luyện. Trình huấn luyện ghi cảnh báo nêu tên họ, còn
`mixup_gating_warning` là hàm thuần đứng sau cảnh báo.

