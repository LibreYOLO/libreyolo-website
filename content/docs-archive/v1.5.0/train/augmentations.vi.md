---
title: Tăng cường dữ liệu
seo_title: Tăng cường dữ liệu khi huấn luyện trong LibreYOLO
description: >-
  Các nút điều chỉnh augmentation trên TrainConfig, bốn dạng pipeline phía sau
  chúng và bảng theo từng family cho biết nút nào được dùng, bị ràng buộc hoặc
  bị bỏ qua.
lead: >-
  Augmentation được cấu hình bằng các nút điều chỉnh trên TrainConfig, nhưng mỗi
  model family chạy pipeline huấn luyện riêng, và pipeline không có nhánh mosaic
  sẽ bỏ qua mosaic_prob thay vì mô phỏng gần đúng.
keywords:
  - data augmentation yolo
  - mosaic augmentation
  - mixup
  - hsv jitter
  - random affine
  - copy paste augmentation
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_prob=1.0,
            mixup_prob=0.15,
            hsv_prob=1.0,
            flip_prob=0.5,
            no_aug_epochs=15,
        )
    - label: CLI
      language: bash
      code: |
        # CLI viết mosaic_prob thành mosaic và mixup_prob thành mixup.
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  support:
    - label: Đọc bảng hỗ trợ cho một family
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: Chỉ những nút bị bỏ qua
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: Bộ tùy chọn phân loại
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(
            data="my-classification-dataset",
            epochs=50,
            auto_augment="randaugment",
            erasing=0.25,
            mixup=0.2,
            cutmix=0.2,
        )
source_hash: 47461cd13aab580c
---

## Thiết lập các nút điều chỉnh

Các nút điều chỉnh tăng cường dữ liệu (augmentation) là đối số `train()` thông
thường.

<code-tabs name="train" />

Hai nút có cách viết CLI ngắn hơn: `mosaic` ánh xạ tới `mosaic_prob` và `mixup`
ánh xạ tới `mixup_prob`. Mọi nút khác được viết giống hệt ở cả hai nơi.

## Ba trạng thái, không phải hai

Một nút có tác dụng hay không phụ thuộc vào family. Thư viện duy trì một bảng
khai báo về điều đó, và mỗi mục có một trong ba trạng thái.

`used` nghĩa là nút đi tới pipeline và thay đổi sample. `ignored` nghĩa là nó
không bao giờ đến pipeline, vì vậy việc đặt giá trị không có tác dụng.
`gated_by_mosaic` nghĩa là nó chỉ áp dụng cho sample đi qua nhánh mosaic, vì vậy
với `mosaic_prob=0`, nó không bao giờ kích hoạt dù đã được nối vào pipeline.

Trạng thái thứ ba thường gây bất ngờ. Trên pipeline kiểu YOLOX, affine warp chạy
trên canvas mosaic và MixUp trộn một sample mosaic, vì vậy `mosaic_prob=0` âm
thầm tắt đồng thời `degrees`, `translate`, `shear`, `perspective`,
`mosaic_scale`, `mixup_prob` và `mixup_scale`. Trainer ghi cảnh báo riêng cho
trường hợp MixUp:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

CLI cũng cảnh báo về các nút bị bỏ qua và chỉ liệt kê những nút bạn thực sự nhập:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

## Bốn dạng pipeline

Các family tập trung thành bốn pipeline huấn luyện, và pipeline quyết định gần
như toàn bộ câu trả lời.

Pipeline mosaic kiểu YOLOX áp dụng HSV jitter và phép lật theo từng sample, sau
đó chạy affine và MixUp bên trong nhánh mosaic. Pipeline này bao gồm YOLOX,
YOLOv7, YOLOv9 cùng các biến thể E2E và P2, RTMDet, PicoDet, RT-DETR, RT-DETRv2
và FOMO.

Pipeline pass-through kiểu DETR không có mosaic và không có affine warp.
Photometric distortion, zoom-out và IoU crop của nó là hằng số recipe thay vì
nút cấu hình, nên chỉ `flip_prob` và `no_aug_epochs` có tác dụng. Pipeline này
bao gồm D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4, EC và RF-DETR với một thay
đổi.

Pipeline ImageFolder cho phân loại bỏ qua mọi nút của phát hiện. Phép lật ngang
là hằng số 0.5 mà `flip_prob` không tác động tới. Thay vào đó, pipeline có bộ
nút riêng được mô tả bên dưới.

YOLO-NAS có dạng riêng: hoàn toàn không có mosaic, affine theo từng sample luôn
bật, và MixUp được áp dụng độc lập thay vì bị ràng buộc. Giá trị `mosaic_scale`
của nó được dùng lại làm phạm vi tỷ lệ affine.

SegFormer và NAFNet đều chạy pipeline riêng theo tác vụ, với tính ngẫu nhiên
được cố định trong family thay vì có thể cấu hình. Với SegFormer, các nút có tác
dụng là thuộc tính class `semantic_scale_jitter` và `semantic_hsv_prob`, không
phải `mosaic_scale` và `hsv_prob`. Phép crop và lật của NAFNet là các thao tác
ghép cặp đầu vào với target ở xác suất cố định 0.5.

## Mỗi family áp dụng nút nào

Bảng bên dưới là đặc tả được phân phối tại
`libreyolo/data/augment/spec.py`, được các test riêng của thư viện đối chiếu với
cách nối pipeline thực. Hãy đọc bảng ở đó thay vì suy luận từ kiến trúc.

<code-tabs name="support" />

Tóm tắt theo pipeline cho các nút cơ sở:

| Nút | Kiểu YOLOX | YOLO-NAS | Kiểu DETR | Phân loại |
|---|---|---|---|---|
| `mosaic_prob` | được dùng | bị bỏ qua | bị bỏ qua | bị bỏ qua |
| `mixup_prob` | bị ràng buộc bởi mosaic | được dùng | bị bỏ qua | bị bỏ qua |
| `hsv_prob` | được dùng | được dùng | bị bỏ qua | bị bỏ qua |
| `flip_prob` | được dùng | được dùng | được dùng | bị bỏ qua |
| `flipud` | được dùng | được dùng | bị bỏ qua | bị bỏ qua |
| `degrees` | bị ràng buộc bởi mosaic | được dùng | bị bỏ qua | bị bỏ qua |
| `translate` | bị ràng buộc bởi mosaic | được dùng | bị bỏ qua | bị bỏ qua |
| `shear` | bị ràng buộc bởi mosaic | được dùng | bị bỏ qua | bị bỏ qua |
| `perspective` | bị ràng buộc bởi mosaic | được dùng | bị bỏ qua | bị bỏ qua |
| `mosaic_scale` | bị ràng buộc bởi mosaic | được dùng | bị bỏ qua | bị bỏ qua |
| `mixup_scale` | bị ràng buộc bởi mosaic | được dùng | bị bỏ qua | bị bỏ qua |
| `no_aug_epochs` | được dùng | được dùng | được dùng | được dùng |

Các ngoại lệ trong những cột đó đều thu hẹp phạm vi:

- RTMDet, PicoDet, RT-DETR, RT-DETRv2 và FOMO không có phép lật dọc, vì vậy
  `flipud` bị bỏ qua. Wrapper mosaic của FOMO cũng được dựng không có
  perspective.
- Pipeline gốc của RF-DETR không có HSV jitter, vì vậy `hsv_prob` bị bỏ qua bên
  cạnh các mục trong cột kiểu DETR.
- EC áp dụng `hsv_prob`, `degrees` và `translate`, nhưng chỉ cho `task="pose"`,
  nơi transform nhận biết keypoint đọc các giá trị này. Đường dẫn detect và
  segment dùng recipe photometric cố định.
- DINOv2 tuân theo cột kiểu DETR cho tác vụ detect và semantic, đồng thời bổ
  sung bộ tùy chọn phân loại cho `task="classify"`.

`no_aug_epochs` được `used` ở mọi nơi nhưng không mang cùng ý nghĩa ở mọi nơi.
Trên pipeline mosaic, nó tắt mosaic và MixUp trong các epoch cuối. Trên pipeline
kiểu DETR, nó dừng augmentation photometric, zoom-out và crop, đồng thời định
hình phần đuôi của lịch. Trên pipeline phân loại và ngữ nghĩa, nó chỉ định hình
phần đuôi.

## Bộ tùy chọn phân loại

Bốn nút điều khiển pipeline phân loại và không điều khiển gì khác. Các family
phát hiện bỏ qua cả bốn.

<code-tabs name="classify" />

`auto_augment` nhận `"randaugment"`, `"autoaugment"`, `"augmix"` hoặc `None`.
`erasing` là xác suất RandomErasing. `mixup` và `cutmix` là xác suất theo batch
tạo nhãn mềm; tối đa một loại chạy trên mỗi batch, MixUp chạy trước, vì vậy hai
giá trị được cộng lại và tổng không nên vượt quá 1.

Cả bốn đều mặc định tắt, vì vậy quá trình huấn luyện phân loại không thay đổi
nếu bạn không yêu cầu.

Cần nêu rõ một xung đột tên: trên CLI, `mixup` là alias của `mixup_prob` dành
cho phát hiện. Trường `mixup` dành cho phân loại không có cách viết CLI riêng và
chỉ truy cập được qua `model.train(mixup=...)` trong Python.

## Nút riêng theo family

Một số nút nằm trên subclass cấu hình của family thay vì class cơ sở, vì vậy
chúng chỉ tồn tại cho family đó và không có flag CLI.

| Family | Nút | Tác dụng |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Xác suất augmentation thực thể copy-paste, chỉ cho `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` dùng lại cùng sample đã lật, `"mixup"` lấy sample thứ hai |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Xác suất xoay ngẫu nhiên 90 độ |
| YOLOv9 | `max_labels` | Giới hạn ground truth trên mỗi ảnh trong transform huấn luyện, mặc định 100 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste cho `task="segment"`, chỉ ở chế độ `"flip"` |
| RF-DETR, D-FINE, EC | `crop_resize_prob` | Xác suất crop-resize ngẫu nhiên |
| EC, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Xác suất jitter và affine nhận biết keypoint trên đường dẫn tư thế |

`max_labels` là nút âm thầm làm mất dữ liệu. Các hộp vượt quá giới hạn bị loại
mà không phát sinh lỗi, vì vậy cần tăng giá trị cho ảnh dày đặc như ảnh hàng
không.

Mosaic và MixUp bị tắt cho quá trình huấn luyện hộp xoay bất kể giá trị các nút,
vì augmentation nhận biết góc cho hộp xoay chưa được triển khai.

## Nội dung liên quan

- Xem [Siêu tham số](/docs/train/hyperparameters) để biết `no_aug_epochs` dưới
  dạng đối số lịch và phần còn lại của `train()`.
- Xem [Dataset](/docs/train/datasets) để biết các định dạng nhãn mà những
  transform này sử dụng.
