---
title: Tăng cường dữ liệu
seo_title: 'Tăng cường dữ liệu khi huấn luyện trong LibreYOLO, kèm ví dụ'
description: >-
  Tác dụng của từng nút điều chỉnh tăng cường dữ liệu trong LibreYOLO, minh họa
  trên một ảnh thật bằng chính mã huấn luyện của thư viện, kèm phạm vi, giá trị
  mặc định và các họ mô hình đọc nút đó.
lead: >-
  Tăng cường dữ liệu (augmentation) khi huấn luyện là các phép biến đổi ngẫu
  nhiên được áp dụng cho mọi mẫu huấn luyện. Chúng được cấu hình bằng các nút
  điều chỉnh trên TrainConfig, và mỗi họ mô hình chạy pipeline riêng, nên một
  nút chỉ thay đổi mẫu ở những nơi pipeline đọc nút đó.
keywords:
  - tăng cường dữ liệu yolo
  - mosaic augmentation
  - mixup
  - hsv augmentation
  - random affine
  - random erasing
  - phát hiện đối tượng nhỏ
  - ví dụ tăng cường dữ liệu
  - copy paste augmentation
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.8.0
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
        # CLI viết mosaic_prob thành mosaic và mixup_prob thành mixup
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  small:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLOX bắt đầu từ mosaic_scale=(0.1, 2.0), có thể thu nhỏ một đối tượng
        # còn một phần mười kích thước. (0.5, 1.5) dừng ở một nửa
        model = LibreYOLO("LibreYOLOXs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_scale=(0.5, 1.5),
        )
    - label: CLI
      language: bash
      code: |
        # Trên CLI, một cặp giá trị là hai số cách nhau bằng dấu phẩy
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=100 mosaic_scale=0.5,1.5
  support:
    - label: Đọc bảng hỗ trợ của một họ mô hình
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
    - label: Gói phân loại
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
source_hash: 72d22a38d647de0a
---

## Thiết lập các nút điều chỉnh

Các nút điều chỉnh augmentation là đối số `train()` thông thường.

<code-tabs name="train" />

Ba nút trong số đó có cách viết ngắn hơn, được cả `train()` lẫn CLI chấp nhận:
`mosaic` ánh xạ tới `mosaic_prob`, `fliplr` tới `flip_prob`, và với phát hiện
đối tượng, `mixup` ánh xạ tới `mixup_prob`. Truyền một cách viết ngắn cùng
trường tương ứng của nó với hai giá trị khác nhau sẽ phát sinh lỗi.
`perspective` không có flag CLI trong 1.8.0, nên được đặt từ Python.

Các nút phân loại được kiểm tra khi pipeline của chúng được dựng, và CLI cũng
kiểm tra `flip_prob`, `flipud`, `cutmix` và `erasing`. `TrainConfig` không kiểm
tra phạm vi của các nút phát hiện đối tượng, nên `degrees=100` sẽ huấn luyện với
góc xoay lên tới 100 độ. [Ma trận tăng cường dữ
liệu](/docs/reference/augmentation-matrix) cho biết phạm vi của mọi nút.

## Ba trạng thái, không phải hai

Một nút có tác dụng hay không phụ thuộc vào họ mô hình. Thư viện duy trì một
bảng khai báo về điều đó, và mỗi mục mang một trong ba trạng thái.

`used` nghĩa là nút đi tới pipeline và thay đổi mẫu. `ignored` nghĩa là nút
không bao giờ tới pipeline, nên việc đặt nó không có tác dụng gì.
`gated_by_mosaic` nghĩa là nút chỉ áp dụng cho các mẫu đã đi qua nhánh mosaic,
nên với `mosaic_prob=0` nó không bao giờ kích hoạt dù đã được nối vào pipeline.

Trạng thái thứ ba là trạng thái hay khiến người dùng bất ngờ. Trên pipeline kiểu
YOLOX, phép biến đổi affine chạy trên canvas mosaic và MixUp trộn một mẫu
mosaic, nên `mosaic_prob=0` âm thầm tắt cùng lúc `degrees`, `translate`,
`shear`, `perspective`, `mosaic_scale`, `mixup_prob` và `mixup_scale`. Trình
huấn luyện ghi một cảnh báo riêng cho trường hợp MixUp:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

CLI cũng cảnh báo về các nút bị bỏ qua, và chỉ liệt kê những nút bạn thực sự
nhập:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

Khi được gọi từ Python, `train()` không cảnh báo về các nút bị bỏ qua. [Ma trận tăng cường dữ
liệu](/docs/reference/augmentation-matrix) có trạng thái của mọi nút trong mọi
họ mô hình và giá trị khởi đầu của từng họ. Bảng này chính là đặc tả (spec) đi
kèm thư viện tại `libreyolo/data/augment/spec.py`, được chính các test của thư
viện đối chiếu với các pipeline, và có thể đọc trực tiếp:

<code-tabs name="support" />

## Tác dụng của từng nút

Các hình minh họa dưới đây được tạo bằng chính các transform huấn luyện của
LibreYOLO 1.8.0, chạy trên ảnh mẫu thông qua một tập dữ liệu (dataset) chỉ có
một ảnh, nên mọi ô mosaic và mọi ảnh được MixUp trộn vào đều là cùng một bức
ảnh. Hộp trong hình đánh dấu người mà LibreYOLO9t phát hiện trên ảnh đó. Dưới
mỗi hình là thiết lập và kích thước của hộp lớn nhất, tính bằng pixel ở độ phân
giải huấn luyện 640 pixel. Khi không có augmentation, người đó có kích thước
291 × 217 px.

Một số nút hình học là phạm vi: `degrees=10` nghĩa là một góc được lấy ngẫu
nhiên ở bất kỳ đâu từ -10 đến +10 độ. Các hình của những nút này cố định lần lấy
ngẫu nhiên ở giá trị ghi trong chú thích, nên chúng cho thấy hiệu ứng mạnh nhất
mà một thiết lập cho phép chứ không phải một mẫu điển hình. Các hình minh họa
phát hiện đối tượng lấy từ pipeline YOLOv9. YOLOv7 huấn luyện qua pipeline đó
mà không thay đổi gì, còn RT-DETR và RT-DETRv2 dùng wrapper mosaic của pipeline
này với transform theo từng mẫu của riêng chúng. YOLOX, RTMDet, PicoDet và FOMO
dựng mosaic và phép biến đổi affine theo cùng cách trong một wrapper riêng, với
MixUp khác.

### mosaic_prob

Xác suất để một mẫu là mosaic: bốn ảnh được đặt trên một canvas có kích thước
gấp đôi kích thước đầu vào, rồi phép biến đổi affine thu canvas này về kích
thước đầu vào. Trên các pipeline kiểu YOLOX, phép biến đổi affine và MixUp chỉ
chạy trên mẫu mosaic, nên cả hai đều dừng khi giá trị này là 0. RT-DETR,
RT-DETRv2, PicoDet và FOMO có giá trị mặc định là 0.

<aug-figure name="mosaic" />

Hình thứ hai cố định tỉ lệ của phép biến đổi affine ở 0.5×, mức mà toàn bộ
canvas vừa trong khung hình. Hai hình cuối là các lần lấy ngẫu nhiên bình thường
với giá trị mặc định của YOLOv9, và chú thích ghi tỉ lệ mà mỗi hình đã lấy được.

### mosaic_scale

Phạm vi để lấy ngẫu nhiên hệ số thu phóng của phép biến đổi affine. Ở 1×, một
đối tượng giữ nguyên kích thước như khi suy luận (inference), và ở 0.5× thì bằng
một nửa kích thước đó. Mặc định cơ sở là `(0.1, 2.0)`, được YOLOX, YOLOv7,
RTMDet, PicoDet, RT-DETR, RT-DETRv2 và FOMO giữ nguyên, nên đối tượng trong các
họ này có thể thu nhỏ còn một phần mười kích thước. YOLOv9 mặc định là
`(0.5, 1.5)`. YOLO-NAS và PP-YOLOE không có mosaic và dùng lại phạm vi này làm
tỉ lệ cho phép biến đổi affine theo từng mẫu của chúng.

<aug-figure name="mosaic_scale" />

Trên 1×, đầu ra chỉ hiển thị phần trên bên trái của canvas, nên các đối tượng bị
cắt ở mép hoặc ra khỏi khung hình.

### degrees

Phạm vi xoay: mỗi mẫu qua phép biến đổi affine được xoay một góc lấy ngẫu nhiên
từ `-degrees` đến `+degrees`. Tâm xoay là góc trên bên trái của canvas, nên một
góc xoay lớn sẽ đẩy phần lớn ảnh ra khỏi khung hình. YOLOX và YOLOv7 giữ mặc
định cơ sở là 10, còn YOLOv9 mặc định là 0.

<aug-figure name="degrees" />

### translate

Phạm vi dịch chuyển, tính theo tỉ lệ của cạnh đầu vào: mỗi mẫu qua phép biến đổi
affine dịch chuyển tối đa `translate` nhân 640 px trên mỗi trục khi
`imgsz=640`. Mặc định cơ sở là 0.1. YOLO-NAS và PP-YOLOE mặc định là 0.25, còn
RTMDet, PicoDet và FOMO là 0.

<aug-figure name="translate" />

### shear

Phạm vi shear tính bằng độ, được lấy ngẫu nhiên riêng cho từng trục. YOLOX và
YOLOv7 giữ mặc định cơ sở là 2, và mọi họ khác có đọc nút này đều mặc định là 0.

<aug-figure name="shear" />

### perspective

Một phép biến đổi phối cảnh chồng lên phép xoay, tỉ lệ, shear và dịch chuyển.
Mỗi góc canvas dịch vào trong một lượng ngẫu nhiên tối đa
`min(perspective * 100, 0.2)` lần nửa cạnh canvas, theo từng trục, nên
`perspective=0.001` dịch một góc tối đa 5 phần trăm cạnh ở mọi độ phân giải, và
mọi giá trị từ `0.002` trở lên đều đạt cùng mức tối đa. Mặc định `0` giữ phép
biến đổi ở dạng affine, và một giá trị âm phát sinh `ValueError` khi mẫu đầu
tiên được biến đổi. Các hình dùng chung một lần lấy ngẫu nhiên, được nhân theo
giá trị. YOLOv9 cùng các biến thể E2E và P2, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE,
RTMDet, PicoDet, RT-DETR và RT-DETRv2 áp dụng nút này.

<aug-figure name="perspective" />

### hsv_prob

Xác suất áp dụng jitter màu HSV. Chỉ xác suất là nút điều chỉnh. Jitter dịch hue
tối đa 5 trên thang hue 0 đến 179 của OpenCV, dịch saturation và value tối đa 30
trên thang 0 đến 255, và mỗi phép dịch trong ba phép dịch này bị bỏ qua với xác
suất một nửa. Mọi pipeline phát hiện đối tượng có đọc `hsv_prob` đều dùng các
mức này. Ba hình cuối là những kết quả dễ thấy nhất trong 119 lần lấy ngẫu nhiên.

<aug-figure name="hsv" />

### flip_prob và flipud

Xác suất lật ngang và lật dọc. `flip_prob` bắt đầu ở 0.5 trong mọi họ trừ FOMO,
họ có mặc định là 0. `flipud` bắt đầu ở 0. RTMDet, PicoDet, RT-DETR, RT-DETRv2,
FOMO, PP-YOLOE và các họ kiểu DETR không có phép lật dọc và bỏ qua `flipud`.
Công thức của PP-YOLOE dùng phép xoay ngẫu nhiên 90 độ thay thế.

<aug-figure name="flip" />

### mixup_prob và mixup_scale

MixUp trộn một mẫu thứ hai vào mẫu hiện tại và giữ hộp của cả hai. Trong
pipeline YOLOv9, mẫu được trộn vào là một mẫu nguyên vẹn không có mosaic, tỉ lệ
trộn được lấy ngẫu nhiên đều trong khoảng từ 0 đến 1, và `mixup_scale` không bao
giờ được đọc, dù spec của thư viện ghi nó là bị ràng buộc bởi mosaic. Trong
wrapper YOLOX, được YOLOX, RTMDet, PicoDet và FOMO sử dụng, và trong wrapper của
YOLO-NAS, mẫu được trộn vào trước tiên được đổi tỉ lệ theo một hệ số lấy từ
`mixup_scale`, rồi hai ảnh được lấy trung bình với tỉ lệ cố định 0.5. Mặc định
cơ sở của `mixup_prob` là 1.0, còn YOLOv9 mặc định là 0.

<aug-figure name="mixup" />

Hai hình cuối trộn một mẫu thứ hai vào một mosaic ở tỉ lệ 0.5×. Chú thích của
hai hình này ghi tỉ lệ trộn λ, tức trọng số của mẫu mosaic.

### no_aug_epochs

`no_aug_epochs` có trạng thái `used` ở mọi nơi, nhưng không mang cùng một ý
nghĩa ở mọi nơi. Trên các pipeline mosaic, nút này tắt mosaic và MixUp trong các
epoch cuối. Trên các pipeline kiểu DETR, nút này dừng các augmentation
photometric, zoom-out và cắt ảnh, đồng thời định hình phần cuối của lịch trình.
Trên các pipeline phân loại, nút này tắt auto-augmentation, erasing, MixUp và
CutMix, còn phép cắt và phép lật vẫn được giữ.

## Công thức kiểu DETR

D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4 và TinyFormer huấn luyện qua một
transform torchvision duy nhất, không có mosaic và không có phép biến đổi affine,
và EdgeCrafter (EC) cũng vậy khi huấn luyện phát hiện đối tượng. Ba thao tác
mạnh của transform này là hằng số của công thức chứ không phải nút điều chỉnh:

- `RandomPhotometricDistort` áp dụng từng thay đổi màu (độ sáng 0.875 đến 1.125,
  độ tương phản và độ bão hòa 0.5 đến 1.5, hue ±0.05, và một phép xáo trộn kênh
  màu) với xác suất 0.5.
- `RandomZoomOut`, với xác suất 0.5, đặt ảnh lên một canvas đen có cạnh tối đa
  gấp bốn lần cạnh của ảnh, nên đối tượng có thể thu nhỏ còn một phần tư kích
  thước.
- `RandomIoUCrop`, được áp dụng với xác suất 0.8, cắt ảnh về một vùng có mỗi
  cạnh từ 0.3 đến 1.0 lần cạnh tương ứng, được chọn sao cho giữ lại ít nhất một
  phần các hộp.

Sau các bước này, những hộp có một cạnh từ 1 px trở xuống bị loại bỏ. Cả ba thao
tác dừng ở epoch dừng của họ đó, muộn nhất là khi `no_aug_epochs` epoch cuối bắt
đầu. Các nút duy nhất mà pipeline này đọc là `flip_prob` và `no_aug_epochs`.

<aug-figure name="detr_ops" />

Ba mẫu từ đầu một lần huấn luyện D-FINE, khi cả ba thao tác đều đang bật:

<aug-figure name="detr_samples" />

RF-DETR có pipeline riêng: một phép lật ngang, sau đó với xác suất
`crop_resize_prob` là một phép đổi kích thước cạnh ngắn hơn về 400, 500 hoặc
600 px và một phép cắt vuông ngẫu nhiên từ 384 đến 600 px, và cuối cùng là phép
đổi kích thước thẳng về độ phân giải huấn luyện hình vuông, kéo giãn ảnh thay vì
letterbox. `crop_resize_prob` mặc định là 0.5 và được đặt từ Python.

<aug-figure name="rfdetr_crop" />

## Đối tượng nhỏ

`erasing` không áp dụng cho phát hiện đối tượng. Random Erasing thuộc gói phân
loại và không pipeline phát hiện đối tượng nào có bước erasing, nên một bộ phát
hiện được huấn luyện như nhau với bất kỳ giá trị nào của nút này, và không có gì
để tắt. Đây là các bước làm thu nhỏ hoặc loại bỏ đối tượng nhỏ:

- `mosaic_scale`. Trên các pipeline mosaic, kích thước của một đối tượng được
  nhân với tỉ lệ lấy ngẫu nhiên, nên cận dưới quyết định kích thước nhỏ nhất mà
  đối tượng có thể có. Với mặc định cơ sở `(0.1, 2.0)`, một đối tượng 20 px có
  thể chỉ còn 2 px. Sau đó pipeline YOLOv9 loại bỏ các hộp có một cạnh từ 2 px
  trở xuống, và transform cuối của nó, giống của YOLOX, loại bỏ các hộp có một
  cạnh từ 1 px trở xuống. Các pixel của đối tượng vẫn nằm trong ảnh nhưng không có nhãn. Cận
  dưới 0.5, như trong `(0.5, 1.5)` của YOLOv9, giữ mọi đối tượng ở mức từ một
  nửa kích thước khi inference trở lên.
- `degrees`, `translate`, `shear` và `perspective` đẩy đối tượng ra mép, nơi hộp
  của chúng bị cắt bớt và đi qua cùng bộ lọc kích thước đó.
- Trong các công thức kiểu DETR, `RandomZoomOut` có thể thu nhỏ đối tượng còn
  một phần tư kích thước và `RandomIoUCrop` có thể cắt chúng ra khỏi ảnh. Cả hai
  đều không phải nút điều chỉnh, và cả hai đều dừng ở epoch dừng.
- Phép cắt của RF-DETR có thể cắt đối tượng ở mép vùng cắt. `crop_resize_prob=0`
  tắt phép cắt này.
- Độ phân giải huấn luyện co giãn mọi đối tượng: letterbox đổi kích thước ảnh
  theo hệ số `imgsz` chia cho cạnh dài hơn của ảnh, nên ở `imgsz=1280` mọi đối
  tượng có chiều rộng và chiều cao gấp đôi so với ở 640.
  [YOLOv9](/docs/models/yolov9) cũng có biến thể P2 với head stride-4 dành cho
  đối tượng nhỏ.

<code-tabs name="small" />

## Gói phân loại

Sáu nút chỉ tác động lên pipeline phân loại. Các họ phát hiện đối tượng bỏ qua
cả sáu nút này.

<code-tabs name="classify" />

`scale` là phạm vi diện tích của phép cắt ngẫu nhiên. `scale=0.5` nghĩa là
`(0.5, 1.0)`, và một cặp giá trị tường minh đặt cả hai cận. Một cận dưới nhỏ có
thể cắt mất hoàn toàn đối tượng, như trong hình thứ ba.

<aug-figure name="cls_crop" />

`auto_augment` nhận `"randaugment"`, `"autoaugment"`, `"augmix"` hoặc `None`, và
áp dụng một lần lấy ngẫu nhiên của chính sách đó cho mỗi mẫu.

<aug-figure name="auto_augment" />

`erasing` là xác suất của Random Erasing, trong `[0, 1)`. Bước này chạy sau
chuẩn hóa và tô một hình chữ nhật chiếm 2 đến 33 phần trăm ảnh bằng giá trị 0,
mà sau chuẩn hóa chính là màu trung bình của ImageNet.

<aug-figure name="erasing" />

`mixup` và `cutmix` là các xác suất theo batch tạo ra nhãn mềm. Mỗi batch chạy
tối đa một phép, MixUp trước, nên hai giá trị được cộng dồn và tổng của chúng
phải không vượt quá 1. Trong các hình, A và B là hai vùng cắt từ cùng một bức
ảnh, và mỗi chú thích ghi nhãn mềm mà ảnh đã trộn dùng để huấn luyện.

<aug-figure name="batch_mix" />

`auto_augment`, `erasing`, `mixup` và `cutmix` mặc định tắt, nên quá trình huấn
luyện phân loại không thay đổi trừ khi bạn yêu cầu.
`train()` và CLI định tuyến `mixup` theo tác vụ: phân loại dùng trộn batch, còn
phát hiện đối tượng dùng `mixup_prob`.

`crop_pct=None` giữ tỉ lệ resize khi đánh giá của họ mô hình đó. Giá trị ghi đè
thay đổi phép cắt khi đánh giá của lượt xác thực chạy trong lúc huấn luyện, còn
khi xuất mô hình vẫn giữ tiền xử lý gốc của họ đó.

CLI cung cấp `auto_augment`, `erasing`, `cutmix`, `scale`, `crop_pct`, `fliplr`
và `flipud`. Với phân loại, `mixup` nghĩa là trộn batch và mặc định là 0.0. Lật
ngang mặc định là 0.5 và lật dọc là 0.0.

`no_aug_epochs` tắt auto-augmentation, erasing, MixUp và CutMix trong các epoch
cuối, nhưng vẫn giữ phép cắt và phép lật. Các công thức phân loại mặc định đặt
phần cuối này là 0.

## Nút riêng theo họ

Một số nút nằm trên subclass cấu hình của họ mô hình thay vì trên class cơ sở,
nên chỉ tồn tại cho họ đó và không có flag CLI.

| Họ | Nút | Tác dụng |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Xác suất augmentation copy-paste thực thể, chỉ với `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` dùng lại chính mẫu đó sau khi lật đối xứng, `"mixup"` lấy một mẫu thứ hai |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Xác suất xoay ngẫu nhiên 90 độ |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `max_labels` | Giới hạn số ground truth trên mỗi ảnh trong các transform huấn luyện, mặc định 300 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste cho `task="segment"`, chỉ ở chế độ `"flip"` |
| RF-DETR | `crop_resize_prob` | Xác suất cắt và đổi kích thước ngẫu nhiên trong pipeline gốc, mặc định 0.5 |
| D-FINE, EdgeCrafter | `crop_resize_prob` | Xác suất cắt và đổi kích thước ngẫu nhiên, chỉ với `task="segment"` |
| EdgeCrafter, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Xác suất jitter trên nhánh pose và xác suất phép biến đổi affine có xét đến keypoint |

`max_labels` là nút làm mất dữ liệu. Các hộp vượt quá giới hạn bị loại bỏ kèm
một cảnh báo chỉ hiện một lần thay vì một lỗi, nên ảnh dày đặc như ảnh chụp từ
trên không cần tăng giá trị này. Bộ nhớ cho label assignment phụ thuộc vào ảnh
dày đặc nhất trong batch, không phụ thuộc vào giới hạn.

Mosaic và MixUp bị tắt khi huấn luyện hộp xoay (OBB) bất kể giá trị các nút, vì
augmentation có xét đến các góc của hộp xoay chưa được hiện thực.

## Nội dung liên quan

- [Ma trận tăng cường dữ liệu](/docs/reference/augmentation-matrix) để xem trạng
  thái của mọi nút trong từng họ mô hình, các phạm vi và giá trị mặc định của
  từng họ.
- [Siêu tham số](/docs/train/hyperparameters) để biết `no_aug_epochs` dưới dạng
  đối số lịch trình và phần còn lại của `train()`.
- [Tập dữ liệu](/docs/train/datasets) để biết các định dạng nhãn mà những
  transform này sử dụng.
