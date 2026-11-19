---
title: Ma trận tăng cường dữ liệu
seo_title: Họ LibreYOLO nào hỗ trợ phép tăng cường dữ liệu nào
description: >-
  Mọi nút điều chỉnh tăng cường dữ liệu cùng phạm vi và cách viết trên CLI,
  trạng thái của từng nút trong mỗi họ của 31 họ mô hình có thể huấn luyện, và
  giá trị mặc định đi kèm của từng họ.
lead: >-
  Việc đặt một nút điều chỉnh tăng cường dữ liệu (augmentation) không bảo đảm
  nút đó đến được pipeline. Trang này ghi lại cách từng họ mô hình có thể huấn
  luyện xử lý mỗi nút trên TrainConfig, cùng giá trị khởi đầu của từng họ, được
  sinh từ thư viện tại đúng bản phát hành mà trang mô tả.
keywords:
  - tăng cường dữ liệu libreyolo
  - tham số tăng cường dữ liệu
  - giá trị mặc định tăng cường dữ liệu
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - mosaic_scale
  - no_aug_epochs
  - ma trận hỗ trợ tăng cường dữ liệu
  - tham số TrainConfig
last_verified: 1.8.0
verification: >-
  Được sinh bởi scripts/build-augmentation/collect.py từ LibreYOLO 1.8.0: danh
  sách nút, trạng thái và ghi chú lấy từ libreyolo/data/augment/spec.py, vốn
  được các test của thư viện gắn chặt với các pipeline thực; giá trị mặc định
  lấy từ subclass TrainConfig của từng họ; và cách viết CLI lấy từ lệnh
  libreyolo train. Các phạm vi và chỗ duy nhất mà spec và một pipeline không
  khớp nhau được cùng script đó kiểm tra với thư viện.
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
    - label: Giá trị mặc định của một họ
      language: python
      code: |
        from libreyolo.training.config import YOLO9Config

        config = YOLO9Config()
        print(config.mosaic_scale, config.degrees, config.mixup_prob)
source_hash: a88efd1537732e6c
---

## Các nút điều chỉnh

Đây là tên trường của `TrainConfig`. `train()` và CLI cũng chấp nhận các cách
viết ngắn trong cột CLI, nên `mosaic` đặt `mosaic_prob`. Trên CLI, `mixup` được
định tuyến theo tác vụ: trộn batch phân loại cho bộ phân loại, `mixup_prob` cho
bộ phát hiện. `perspective` không có flag CLI và được đặt từ Python.

<aug-knobs />

Mặc định cơ sở là giá trị trên chính `TrainConfig`. Hầu hết các họ ghi đè một số
giá trị trong đó, và [Giá trị mặc định theo họ](#defaults-by-family) cho biết
các giá trị mà một lần chạy thực sự bắt đầu từ đó. Để xem hình minh họa tác dụng
của từng nút, hãy xem [Tăng cường dữ liệu](/docs/train/augmentations).

Phạm vi trong ngoặc là khoảng, nên `[0, 1)` bao gồm 0 và không bao gồm 1. Các
nút phân loại được kiểm tra khi pipeline của chúng được dựng và phát sinh
`ValueError` nếu nằm ngoài phạm vi. Các nút phát hiện đối tượng không được
`TrainConfig` kiểm tra, nên một giá trị ngoài phạm vi vẫn được dùng để huấn
luyện mà không có thông báo nào. Ngoại lệ là một tỉ lệ lấy ngẫu nhiên từ 0 trở
xuống và một giá trị `perspective` âm, cả hai đều phát sinh `ValueError` khi một
mẫu được biến đổi.

<code-tabs name="usage" />

## Ba trạng thái

| Trạng thái | Ý nghĩa |
|---|---|
| `used` | Nút đi tới pipeline huấn luyện của họ mô hình và thay đổi mẫu |
| `gated_by_mosaic` | Nút chỉ áp dụng cho các mẫu đã đi qua nhánh mosaic, nên với `mosaic_prob == 0` nó không bao giờ kích hoạt |
| `ignored` | Nút không bao giờ tới pipeline; việc đặt nó không có tác dụng gì |

`ignored` là trạng thái đáng kiểm tra trước một lần chạy, vì sẽ không có gì thất
bại. CLI cảnh báo khi một tham số huấn luyện được đặt tường minh là tham số bị họ
mô hình được chọn bỏ qua, và trình huấn luyện cảnh báo khi `mixup_prob > 0`
không thể kích hoạt vì họ đó ràng buộc MixUp vào mosaic và `mosaic_prob` bằng 0.

## Họ nào áp dụng nút nào

<aug-matrix />

Mọi họ được bao quát đều theo một trong sáu pipeline. Trong pipeline kiểu YOLOX,
tiền xử lý theo từng mẫu áp dụng jitter HSV và phép lật, còn phép biến đổi
affine và MixUp chỉ chạy bên trong nhánh mosaic. Ngược lại, YOLO-NAS chạy một
phép biến đổi affine theo từng mẫu luôn bật, bỏ qua mosaic và áp dụng MixUp độc
lập, dùng lại `mosaic_scale` làm phạm vi tỉ lệ của phép biến đổi affine.

Pipeline kiểu DETR là một transform truyền thẳng (pass-through) không có mosaic.
Các bước biến dạng photometric, zoom-out và IoU-crop của pipeline này là hằng số
của công thức chứ không phải nút có thể cấu hình, đó là lý do `hsv_prob` và các
nút hình học không bao giờ tới được nó. Phân loại dùng `flip_prob` cho lật ngang
và `flipud` cho lật dọc. Với phân đoạn ngữ nghĩa, jitter tỉ lệ và HSV đến từ
thuộc tính class của họ mô hình thay vì nút cấu hình, còn phép lật trong phục
hồi ảnh là thao tác gắn liền đầu vào với đích, với xác suất cố định 0.5.

`no_aug_epochs` được tuân theo ở mọi nơi, dù thứ nó tắt khác nhau: mosaic và
MixUp với kiểu YOLOX, phép biến đổi affine và MixUp với YOLO-NAS, các
augmentation photometric và cắt ảnh mạnh cùng phần cuối của learning rate với
kiểu DETR, và auto-augmentation, erasing, MixUp và CutMix với phân loại. Phép
cắt và phép lật của phân loại vẫn bật.

Khi một họ khác với các họ còn lại cùng pipeline, lý do là một trong các trường
hợp sau:

| Họ | Khác biệt |
|---|---|
| `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2` | Bỏ qua `flipud`: transform của các họ này không có phép lật dọc |
| `fomo` | Bỏ qua `perspective` và `flipud` |
| `ppyoloe` | Bỏ qua `flipud`: công thức gốc dùng phép xoay ngẫu nhiên 90 độ thay thế |
| `yolo9`, `yolo9_e2e`, `yolo9_p2`, `yolo7`, `rtdetr`, `rtdetrv2` | Bỏ qua `mixup_scale`: MixUp của các họ này trộn các khung hình nguyên vẹn ở một tỉ lệ ngẫu nhiên mà không đổi tỉ lệ khung hình nào |
| `gtr` | Dùng `mosaic_prob` và `mixup_prob` trong `mosaic_epochs` epoch đầu; `degrees`, `translate` và `mosaic_scale` định hình phép biến đổi affine theo sau mosaic |
| `rfdetr` | Bỏ qua `hsv_prob`: pipeline gốc không có jitter HSV |
| `ec` | Dùng `hsv_prob`, `degrees` và `translate`, chỉ cho `task="pose"`; detect và segment dùng công thức photometric cố định |
| `dinov2` | Dùng gói phân loại, chỉ cho `task="classify"` |

`ec` và `dinov2` là các họ đa tác vụ, nên một nút chỉ được đánh dấu là bị bỏ qua
khi mọi tác vụ có thể huấn luyện của họ đó đều bỏ qua nút này. Nhờ vậy, cảnh báo
của CLI không bao giờ sai với tác vụ này trong khi đúng với tác vụ khác.

Dome-DETR kế thừa nguyên vẹn các transform của D-FINE. Điều duy nhất mô hình này
không nhận được là huấn luyện đa tỉ lệ (multi-scale), do cấu hình của nó tắt chứ
không phải đặc tả (spec) augmentation.

Ba mươi mốt họ được bao quát. Với một họ nằm ngoài danh sách này,
`ignored_aug_params` trả về một tập rỗng, nên không có cảnh báo nào được phát ra
cho họ đó.

## Giá trị mặc định theo họ

Giá trị khởi đầu trong cấu hình của từng họ phát hiện đối tượng. Ô trống nghĩa
là họ đó bỏ qua nút tương ứng, nên giá trị mặc định của nút không thay đổi gì.

<aug-defaults />

Hai quy luật quan trọng hơn bất kỳ giá trị đơn lẻ nào. RT-DETR, RT-DETRv2,
PicoDet và FOMO bắt đầu với `mosaic_prob=0`, nên với các họ này, những nút được
đánh dấu là bị ràng buộc bởi mosaic trong ma trận không có tác dụng gì cho đến
khi mosaic được bật. YOLOv9 bắt đầu với phép xoay, shear và MixUp đều tắt và với
`mosaic_scale=(0.5, 1.5)`, trong khi mặc định cơ sở là `(0.1, 2.0)`.

EdgeCrafter chỉ đọc `hsv_prob`, `degrees` và `translate` với `task="pose"`, nên
hàng của họ này hiển thị giá trị mặc định của pose cho ba nút đó.

Các họ phân loại bắt đầu từ mặc định cơ sở: `flip_prob=0.5`, `flipud=0.0`,
`scale=(0.5, 1.0)`, `crop_pct` riêng của từng họ, và `auto_augment`, `erasing`,
`mixup` cùng `cutmix` đều tắt.

## Nút riêng theo họ

Một số họ mang các nút augmentation trên subclass `TrainConfig` riêng thay vì
trên class cơ sở. CLI không cung cấp các nút này; hãy đặt chúng qua Python API.

| Họ | Nút | Ý nghĩa |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Xác suất augmentation copy-paste thực thể, chỉ với `task="segment"` |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Nguồn copy-paste: `flip` lật đối xứng chính mẫu đó, `mixup` dùng một mẫu thứ hai |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Xác suất xoay ngẫu nhiên 90 độ |
| `rfdetr` | `copy_paste` | Xác suất copy-paste cho `task="segment"`, chỉ chế độ `flip` |
| `rfdetr` | `copy_paste_mode` | Chế độ nguồn copy-paste cho `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Xác suất cắt và đổi kích thước ngẫu nhiên trong pipeline gốc |
| `dfine` | `crop_resize_prob` | Xác suất cắt và đổi kích thước ngẫu nhiên, `task="segment"` |
| `ec` | `crop_resize_prob` | Xác suất cắt và đổi kích thước ngẫu nhiên, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Xác suất jitter độ sáng và độ tương phản, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Xác suất phép biến đổi affine có xét đến keypoint, `task="pose"` |

`rot90` áp dụng cho detect và OBB trên `yolo9`.

## Truy vấn spec

| Hàm trợ giúp | Trả về |
|---|---|
| `aug_support(family)` | Bảng ánh xạ nút sang `Support`, hoặc `None` với họ không xác định |
| `ignored_aug_params(family)` | Tập tên các nút mà họ đó bỏ qua; rỗng với họ không xác định |
| `uses_mosaic_gating(family)` | MixUp của họ đó có chỉ kích hoạt trên mẫu mosaic hay không |
| `display_name(family)` | Tên họ hiển thị cho người dùng, dùng trong cảnh báo |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | Nội dung cảnh báo khi MixUp không bao giờ có thể kích hoạt, nếu không thì `None` |

`Support` là một named tuple gồm `status` và `note`, trong đó note giải thích vì
sao một nút bị bỏ qua hoặc bị ràng buộc bởi mosaic với họ đó. Trong 1.8.0, spec
vẫn ghi `mixup_scale` là bị ràng buộc bởi mosaic với các họ huấn luyện qua
wrapper mosaic của YOLOv9, nên `aug_support` báo nó là bị ràng buộc ở đó, dù ma
trận ở trên, vốn được đối chiếu với pipeline, đánh dấu nó là bị bỏ qua.

## Ràng buộc mosaic

Với một họ kiểu YOLOX, `mixup_prob=0.5` cùng `mosaic_prob=0` tắt hoàn toàn
MixUp, vì MixUp chỉ áp dụng cho mẫu mosaic. Tổ hợp này dễ xảy ra khi tắt mosaic
ở giai đoạn cuối quá trình huấn luyện. Trình huấn luyện ghi một cảnh báo nêu tên
họ mô hình, và `mixup_gating_warning` là hàm thuần (pure function) đứng sau cảnh
báo đó.
