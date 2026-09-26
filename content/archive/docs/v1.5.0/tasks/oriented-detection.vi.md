---
title: Phát hiện hộp xoay
seo_title: Phát hiện hộp xoay trong LibreYOLO
description: >-
  Phát hiện đối tượng bị xoay trong LibreYOLO: các family phục vụ hộp xoay, dòng
  nhãn bốn góc và các lời gọi dự đoán, huấn luyện, xác thực cùng xuất.
lead: >-
  Phát hiện đối tượng có hướng định vị mỗi thực thể bằng hình chữ nhật xoay thay
  vì hình chữ nhật thẳng trục, nhờ vậy đối tượng nghiêng được bao sát thay vì
  nằm trong hộp chứa nhiều background. Key tác vụ là obb.
keywords:
  - phát hiện oriented bounding box
  - phát hiện đối tượng xoay
  - OBB python
  - dataset DOTA
  - phát hiện vật thể hàng không
  - rotated IoU
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        # Cần thành phần bổ sung rfdetr: pip install "libreyolo[rfdetr]"
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Hậu tố -obb trong tên tệp chọn tác vụ, vì vậy không cần
        # đối số task.
        model = LibreYOLO("LibreRFDETRs-obb.pt")
        result = model(SAMPLE_IMAGE, save=True)

        obb = result.obb
        print(obb.xywhr)   # (N, 5): tâm x, tâm y, chiều rộng, chiều cao, radian
        print(obb.conf, obb.cls)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRFDETRs-obb.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Dùng góc thay cho số đo góc
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreRFDETRs-obb.pt")(SAMPLE_IMAGE)
        obb = result.obb

        print(obb.xyxyxyxy.shape)    # điểm góc (N, 4, 2) theo pixel
        print(obb.xyxyxyxyn.shape)   # cùng dữ liệu đã chuẩn hóa
        print(obb.xyxy.shape)        # hộp thẳng trục bao ngoài (N, 4)
    - label: Checkpoint nhỏ hơn
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRn-obb.pt")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr.shape)
    - label: RT-DETRv2
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Trọng số DOTA v1.0, 15 lớp hàng không ở 1024 px. Graph hộp xoay

        # được nhận dạng từ tensor riêng của checkpoint, vì vậy không cần đối số
        task.

        model = LibreYOLO("LibreRTDETRv2n-obb.pt")

        result = model("aerial.png", save=True)


        obb = result.obb

        print(obb.xywhr)

        print(result.names)   # máy bay, tàu, cảng, trực thăng và 11 lớp khác
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Tiếp tục từ trọng số hộp xoay đã công bố. data phải trỏ tới

        # dataset có dòng nhãn mang bốn góc.

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        model.train(data="my-obb-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: Từ trọng số phát hiện
      language: bash
      code: |
        # Trọng số phát hiện không có dự đoán góc, vì vậy đây là thao tác
        # transfer tường minh. Yêu cầu task=obb chính là điều cho phép.
        libreyolo train model=LibreRFDETRs.pt data=my-obb-dataset.yaml \
          task=obb epochs=50 imgsz=512
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        # val() trả về dict thuần, không phải đối tượng.
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml
    - label: RT-DETRv2
      language: bash
      code: |
        libreyolo val model=LibreRTDETRv2n-obb.pt data=my-obb-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRFDETRs-obb.pt format=onnx imgsz=512
    - label: RT-DETRv2
      language: bash
      code: >
        # ONNX và TorchScript là các target đã xác thực tại đây, ở FP32,

        # batch 1, trên canvas 1024 x 1024 cố định.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, vì vậy artifact đã xuất được nạp
        # như checkpoint và trả về cùng một đối tượng Results.
        model = LibreYOLO("LibreRFDETRs-obb.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr)
source_hash: 0d605d956f3ea025
---

## Định nghĩa

Phát hiện hộp xoay thêm một số vào kết quả phát hiện: góc. Mỗi thực thể nhận
hình chữ nhật xoay, lớp đối tượng và điểm số. Lợi ích là độ khít. Con tàu nghiêng
45 độ, mái nhà kho hay hàng xe tải đỗ: hộp thẳng trục bao quanh chúng chủ yếu là
background, còn hai hộp lân cận chồng lên nhau ngay cả khi đối tượng không chồng
lấn. Đó là lý do tác vụ phổ biến trong ảnh hàng không và bố cục tài liệu, đồng
thời là lý do dataset tham chiếu là DOTA.

`obb` là key tác vụ chuẩn, còn hậu tố `-obb` trong tên tệp checkpoint chọn tác
vụ, vì vậy không cần `task=` khi nạp trọng số đã công bố.

`predict()` điền `result.obb`. `.xywhr` là dạng chuẩn `(N, 5)`: tâm x, tâm y,
chiều rộng, chiều cao và một góc theo radian biểu thị phép xoay cạnh chiều rộng
quanh tâm. `.conf` và `.cls` mang điểm số cùng index lớp vào `result.names`, còn
`.id` mang track id khi theo dõi. `.xyxyxyxy` chuyển mỗi dòng thành bốn điểm góc
dạng pixel `(N, 4, 2)`, `.xyxyxyxyn` chuẩn hóa các góc đó, còn `.xyxy` cho hộp
thẳng trục bao ngoài, là dạng cần dùng khi mã downstream chỉ hiểu hình chữ nhật.
`result.boxes` cũng được điền bằng dạng thẳng trục.

## Mô hình

Hai family phục vụ tác vụ này, và lựa chọn phụ thuộc vào việc có cần huấn luyện
hay không.

[RF-DETR](/docs/models/rf-detr) là family có thể huấn luyện. Nó dự đoán, huấn
luyện, xác thực và xuất hộp xoay, đồng thời cung cấp checkpoint hộp xoay đã công
bố ở bốn kích thước n, s, m và l. Nó cần thành phần bổ sung riêng,
`pip install "libreyolo[rfdetr]"`, còn trang mô hình trình bày giấy phép và nguồn
gốc trọng số.

Hãy đọc phần bên dưới về nội dung thực sự được các checkpoint này dự đoán trước
khi lập kế hoạch dựa vào chúng.

[RT-DETRv2](/docs/models/rt-detr) là family có trọng số hàng không. Nó công bố
`LibreRTDETRv2n-obb.pt` đến `LibreRTDETRv2x-obb.pt`, các checkpoint DOTA v1.0
một tỷ lệ chính thức được chuyển sang định dạng LibreYOLO, bao gồm 15 lớp DOTA ở
1024 px. Nó không cần thành phần bổ sung ngoài package cơ sở, graph hộp xoay
được nhận dạng từ tensor riêng của checkpoint, còn dự đoán, xác thực và xuất
ONNX cùng TorchScript đều được hỗ trợ. Huấn luyện thì không: tác vụ hộp xoay chỉ
dành cho inference trên family này, `train()` phát sinh lỗi, và không có
transfer từ trọng số phát hiện vốn dùng backbone khác. Theo dõi và test-time
augmentation cũng không khả dụng cho hộp xoay.

Tóm lại: dùng RT-DETRv2 cho category DOTA có sẵn, dùng RF-DETR cho nhãn hộp xoay
riêng.

## Dự đoán

Trọng số được tải từ Hugging Face trong lần sử dụng đầu tiên và lưu vào cache
cục bộ.

<code-tabs name="predict" />

Cần biết checkpoint RF-DETR đã công bố là gì trước khi chạy. Dù DOTA là
benchmark tham chiếu cho tác vụ này, các trọng số đó không được huấn luyện trên
DOTA. Cả bốn được khởi tạo từ trọng số phát hiện RF-DETR và tinh chỉnh trên một
dataset Roboflow Universe duy nhất gồm cảnh quay UAV, với sáu lớp phương tiện:
bike, bus, car, other_vehicle, taxi và truck. Model card mô tả chúng là trọng số
phát triển, được tạo trong khi xác thực hỗ trợ huấn luyện hộp xoay, và nêu rằng
không nên xem chúng là trọng số chính thức cho production hoặc benchmark.

Trong thực tế, chúng là điểm bắt đầu hoạt động được cho hộp xoay của phương tiện
nhìn từ trên cao và để xác minh pipeline chạy đầu cuối. Mọi domain khác cần huấn
luyện trên nhãn hộp xoay riêng, còn với các category hàng không vốn gắn với
DOTA, checkpoint RT-DETRv2 mới là mô hình thực sự được huấn luyện trên dữ liệu
đó. `conf` và `max_det` định hình đầu ra giống như khi phát hiện. Xem [dự
đoán](/docs/predict) để biết về nguồn, stream và cách xử lý kết quả.

## Định dạng dataset

Bố cục giống bố cục phát hiện: mỗi ảnh có một tệp nhãn `.txt`, được tìm bằng
cách thay `images` thành `labels` trong đường dẫn ảnh và đổi phần mở rộng.

```text
dataset/
  data.yaml
  images/
    train/P0001.png
    val/P0101.png
  labels/
    train/P0001.txt
    val/P0101.txt
```

Một dòng có đúng chín trường, gồm index lớp rồi bốn điểm góc theo thứ tự:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Bốn điểm là số float chuẩn hóa trong `[0, 1]` và phải tạo thành hình chữ nhật
xoay không suy biến. Tệp nhãn không lưu góc: loader suy ra `xywhr` chuẩn từ các
góc. Parser mặc định nghiêm ngặt và từ chối tọa độ ngoài phạm vi, còn quá trình
nạp dataset cùng xác thực có thể cắt về `[0, 1]` trước cho nhãn ở biên crop nhưng
hợp lệ, sau đó vẫn từ chối hộp suy biến.

Việc parse dòng nhận biết tác vụ. Chín trường chỉ biểu thị hộp xoay trong chế độ
`obb`; trong chế độ `segment`, cùng dòng được đọc như polygon bốn điểm.

YAML giống YAML phát hiện:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: plane
  1: ship
```

COCO JSON gốc cũng được nạp, với mapping `annotations` từ tên split tới tệp
JSON. Nhãn được đọc theo thứ tự ưu tiên: trường `obb` gồm tám tọa độ góc theo
pixel, trường `obb` dạng `[cx, cy, w, h, angle]` với góc theo radian, polygon
`segmentation` hoặc RLE được fit lại thành hình chữ nhật diện tích nhỏ nhất,
hoặc `bbox` COCO thuần được xử lý như hình chữ nhật thẳng trục và chuẩn hóa về
`xywhr`.

Parser dòng chuẩn là `libreyolo.data.parse_yolo_obb_label_line`.

## Huấn luyện

<code-tabs name="train" />

Huấn luyện tác vụ này nghĩa là dùng RF-DETR. Theo mặc định, quá trình huấn luyện
tiếp tục từ checkpoint `-obb` đã công bố. Bắt đầu từ trọng số phát hiện là
transfer có chủ ý: các trọng số đó không dự đoán góc, còn truyền `task=obb` là
điều cho phép thay đổi. Giữ `lr0` ở hoặc dưới `1e-4`, giống các tác vụ khác của
family. Không thể tinh chỉnh checkpoint hộp xoay RT-DETRv2; hãy dùng nguyên
trạng hoặc huấn luyện RF-DETR trên nhãn riêng. Xem [huấn luyện](/docs/train) để
biết về dataset, augmentation, multi-GPU và logger.

## Xác thực

`val()` trả về dictionary thuần gồm các key `metrics/`. Phép khớp dùng rotated
IoU, được tính giữa hình chữ nhật xoay thay vì hộp thẳng trục bao ngoài, vì vậy
dự đoán đúng vị trí nhưng sai góc sẽ được tính là trượt.

<code-tabs name="val" />

`metrics/mAP50-95` là mean average precision lấy trung bình trên các ngưỡng IoU
từ 0.50 đến 0.95 theo bước 0.05 và là con số chính. Khác với đường dẫn COCO dùng
cho phát hiện, tác vụ này áp dụng `iou_thresholds` trong cấu hình xác thực, vì
vậy có thể thay đổi lượt quét. `metrics/mAP50` và `metrics/mAP75` là các phiên
bản một ngưỡng. `metrics/precision` và `metrics/recall` là precision và recall
thực tại IoU 0.50, được đọc ở operating point lỏng nhất: mọi dự đoán vượt qua
ngưỡng độ tin cậy đều được tính, còn ngưỡng đó mặc định là 0.001 khi xác thực.
Vì vậy, tăng `conf` sẽ thay đổi chúng, trong khi số liệu mAP dùng toàn bộ đường
cong precision-recall giữ nguyên. Bốn key lặp lại dưới hậu tố `(OBB)` gồm
`metrics/mAP50-95(OBB)`, `metrics/mAP50(OBB)`, `metrics/precision(OBB)` và
`metrics/recall(OBB)`, đây là cách bên gọi phân biệt kết quả hộp xoay với kết
quả thẳng trục khi cả hai nằm trong cùng bảng. `metrics/mAP75` không có bản sao
có hậu tố.

Hai tùy chọn không có tác dụng trên tác vụ này. `save_json` và `save_plots` được
chấp nhận và ghi cảnh báo: chưa triển khai dump dự đoán hộp xoay và biểu đồ xác
thực.

## Xuất

<code-tabs name="export" />

Artifact đã xuất được nạp lại qua `LibreYOLO()` theo hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`. Phạm
vi định dạng khác nhau theo tác vụ trên cùng family, còn ma trận trên trang mô
hình được tạo từ tập đã xác thực và nêu lý do target không khả dụng. Xem [xuất
và triển khai](/docs/export) để biết các định dạng, thành phần bổ sung và ràng
buộc.
