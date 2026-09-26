---
title: Phân đoạn thực thể
seo_title: Phân đoạn thực thể trong LibreYOLO
description: >-
  Phân đoạn từng đối tượng trong LibreYOLO: các family phục vụ tác vụ, định dạng
  nhãn polygon và các lời gọi dự đoán, huấn luyện, xác thực cùng xuất.
lead: >-
  Phân đoạn thực thể định vị từng thực thể đối tượng và trả về mặt nạ theo pixel
  cho mỗi thực thể, bên cạnh hộp, lớp đối tượng và điểm số mà detector trả về.
  Key tác vụ là segment.
keywords:
  - phân đoạn thực thể python
  - dự đoán mặt nạ đối tượng
  - huấn luyện mô hình segmentation
  - nhãn polygon
  - thư viện segmentation MIT
  - mask mAP
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Hậu tố -seg trong tên tệp chọn mask head, vì vậy không cần

        # đối số task.

        model = LibreYOLO("LibreDFINEn-seg.pt")

        result = model(SAMPLE_IMAGE, save=True)


        print(result.masks.data.shape)   # (N, H, W), mỗi kết quả phát hiện một
        mặt nạ

        print(result.boxes.xyxy.shape)   # (N, 4), cùng N dòng
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn-seg.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Đường viền mặt nạ
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDFINEn-seg.pt")

        result = model(SAMPLE_IMAGE)


        # .xy là danh sách contour (P, 2) theo pixel, .xyn là cùng dữ liệu đã
        chuẩn hóa.

        for name, contour in zip(result.boxes.cls, result.masks.xy):
            print(result.names[int(name)], contour.shape)
    - label: 'Family khác, cùng lời gọi'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Tiếp tục từ trọng số phân đoạn đã công bố, gồm cả mask head.

        # data phải trỏ tới dataset có nhãn chứa polygon.

        model = LibreYOLO("LibreDFINEn-seg.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Từ trọng số phát hiện
      language: bash
      code: |
        # Trọng số phát hiện không có mask head, vì vậy đây là thao tác
        # transfer tường minh: head bắt đầu chưa được huấn luyện. Yêu cầu
        # task=segment chính là điều cho phép thao tác đó.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])       # mặt nạ
        print(metrics["metrics/mAP50-95(M)"])    # mặt nạ, tường minh
        print(metrics["metrics/mAP50-95(B)"])    # hộp
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn-seg.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn-seg.pt format=onnx imgsz=640
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, vì vậy artifact đã xuất được nạp
        # như checkpoint và trả về cùng một đối tượng Results.
        model = LibreYOLO("LibreDFINEn-seg.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 33e331eac0f9b0af
---

## Định nghĩa

Phân đoạn thực thể là phát hiện cộng hình dạng. Mỗi thực thể đối tượng vẫn nhận
một hộp, lớp đối tượng và điểm số, đồng thời nhận mặt nạ nhị phân bao phủ các
pixel thuộc về nó. Mặt nạ có thể chồng lấn, còn pixel không thuộc đối tượng nào
được để không gán, đây là điểm phân biệt tác vụ với [phân đoạn ngữ
nghĩa](/docs/tasks/semantic-segmentation) và [phân đoạn toàn
cảnh](/docs/tasks/panoptic-segmentation).

`segment` là key tác vụ chuẩn, còn hậu tố `-seg` trong tên tệp checkpoint chọn
tác vụ, vì vậy không cần `task=` khi nạp trọng số đã công bố.

`predict()` điền `result.masks` bên cạnh `result.boxes`. `.data` là stack
`(N, H, W)` trên canvas ảnh gốc, các dòng được căn theo hộp, vì vậy mặt nạ `i`
thuộc về hộp `i`. `.xy` chuyển mỗi mặt nạ thành contour ngoài lớn nhất dưới dạng
mảng pixel `(P, 2)`, còn `.xyn` cung cấp cùng contour đã chuẩn hóa.

## Mô hình

Bốn family vừa huấn luyện vừa dự đoán mặt nạ: [RF-DETR](/docs/models/rf-detr),
[EdgeCrafter](/docs/models/edgecrafter), [D-FINE](/docs/models/d-fine) và
[RTMDet](/docs/models/rtmdet). RF-DETR cần thành phần bổ sung riêng,
`pip install "libreyolo[rfdetr]"`; ba family còn lại chạy trên package cơ sở.

[Mask R-CNN](/docs/models/mask-rcnn) dự đoán, xác thực và xuất mặt nạ, nhưng
`train()` phát sinh `NotImplementedError`.

[EoMT](/docs/models/eomt) dự đoán và xác thực mặt nạ, cũng không thể huấn luyện,
còn phạm vi xuất hẹp hơn nữa: `export()` chỉ nhận tác vụ semantic và phát sinh
`NotImplementedError` cho `segment` cùng `panoptic`, vì hợp đồng runtime
query-mask mà hai tác vụ cần chưa được định nghĩa. Hãy dùng EoMT cho mặt nạ thực
thể trong Python, không qua graph đã xuất.

Một nhóm riêng phân đoạn theo prompt thay vì danh sách lớp: cú nhấp, hộp hoặc
cụm từ chọn đối tượng và mô hình trả về mặt nạ. [SAM](/docs/models/sam), [SAM
2](/docs/models/sam-2), [SAM 3](/docs/models/sam-3),
[MobileSAM](/docs/models/mobilesam), [EdgeTAM](/docs/models/edgetam) và
[PicoSAM3](/docs/models/picosam3) hoạt động theo cách này, cũng như
[SenseNova-Vision](/docs/models/sensenova-vision), có phân đoạn referring: mô
hình nhận cụm từ đặt tên một đối tượng. Chúng được nạp qua factory và thành phần
bổ sung riêng, còn từng trang mô hình chứa lời gọi chính xác.

## Dự đoán

Trọng số được tải từ Hugging Face trong lần sử dụng đầu tiên và lưu vào cache
cục bộ.

<code-tabs name="predict" />

`conf` và `max_det` định hình đầu ra giống như với phát hiện, còn mặt nạ được lọc
cùng các hộp tương ứng. Xem [dự đoán](/docs/predict) để biết về nguồn, stream và
cách xử lý kết quả.

## Định dạng dataset

Bố cục giống bố cục phát hiện: mỗi ảnh có một tệp nhãn `.txt`, được tìm bằng
cách thay `images` thành `labels` trong đường dẫn ảnh và đổi phần mở rộng.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Phần thay đổi là nội dung dòng. Một segment là index lớp đối tượng, sau đó là
polygon phẳng:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

Có ít nhất ba điểm, vì vậy số tọa độ sau index lớp phải chẵn và ít nhất là sáu,
còn polygon không được suy biến. Tọa độ là số float trong `[0, 1]` tương đối với
chiều rộng và chiều cao ảnh gốc. Dòng phát hiện năm trường cũng được chấp nhận
trong dataset phân đoạn và được đọc như segment hình chữ nhật, nhờ vậy dataset
chỉ có hộp có thể được nạp mà không cần bước chuyển đổi.

YAML giống YAML phát hiện:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

COCO JSON gốc cũng hoạt động: thêm mapping `annotations` từ tên split tới tệp
JSON, còn đường dẫn split cho biết thư mục gốc của ảnh.

## Huấn luyện

<code-tabs name="train" />

Theo mặc định, quá trình huấn luyện tiếp tục từ checkpoint `-seg` đã công bố.
Có thể bắt đầu từ trọng số phát hiện nhưng đây là transfer có chủ ý: các trọng
số đó không có mask head, vì vậy head bắt đầu chưa được huấn luyện, còn việc
truyền `task=segment` là điều cho phép thay đổi. Xem [huấn luyện](/docs/train) để
biết về dataset, augmentation, multi-GPU và logger.

## Xác thực

`val()` trả về dictionary thuần gồm các key `metrics/`. Hộp và mặt nạ được tính
điểm riêng, cả hai dùng đánh giá COCO, còn số liệu mặt nạ là chính.

<code-tabs name="val" />

Các key không hậu tố chứa kết quả mặt nạ: `metrics/mAP50-95`, `metrics/mAP50`,
`metrics/mAP75`, sau đó `metrics/mAP_small`, `metrics/mAP_medium` và
`metrics/mAP_large` theo diện tích đối tượng, cùng `metrics/AR1`, `metrics/AR10`,
`metrics/AR100`, `metrics/AR_small`, `metrics/AR_medium`, `metrics/AR_large` cho
average recall. `metrics/AR_max_det` và `metrics/max_det` ghi giới hạn phát hiện
mà lượt chạy đã dùng.

Bốn số liệu còn được công bố dưới hậu tố tường minh, `(M)` cho mặt nạ và `(B)`
cho hộp, để phép so sánh không bao giờ phụ thuộc vào con số family chọn làm
chính: `metrics/mAP50-95(M)` và `metrics/mAP50-95(B)`, `metrics/mAP50(M)` và
`metrics/mAP50(B)`, `metrics/precision(M)` và `metrics/precision(B)`,
`metrics/recall(M)` và `metrics/recall(B)`. Tác vụ này không có
`metrics/precision` hoặc `metrics/recall` không hậu tố.

Hãy đọc kỹ các key precision và recall. Chúng được giữ để tương thích ngược và
là alias, không phải operating point: `metrics/precision(M)` giữ cùng giá trị
với `metrics/mAP50-95(M)`, còn `metrics/recall(M)` giữ cùng giá trị với mask AR
ở 100 kết quả phát hiện, và `(B)` hoạt động tương tự cho hộp. Vẽ một cặp key này
sẽ báo cáo cùng một số hai lần.

## Xuất

<code-tabs name="export" />

Artifact đã xuất được nạp lại qua `LibreYOLO()` theo hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`. Phạm
vi phân đoạn hẹp hơn phạm vi phát hiện trên cùng family. Ma trận trên từng trang
mô hình được tạo từ tập đã xác thực và nêu lý do target không khả dụng. Xem
[xuất và triển khai](/docs/export) để biết các định dạng, thành phần bổ sung và
ràng buộc.
