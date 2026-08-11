---
title: Ước lượng tư thế
seo_title: Ước lượng tư thế trong LibreYOLO
description: >-
  Dự đoán keypoint trên mỗi thực thể trong LibreYOLO: các family phục vụ tác vụ,
  định dạng nhãn và các lời gọi dự đoán, huấn luyện, xác thực cùng xuất.
lead: >-
  Ước lượng tư thế định vị từng thực thể và trả về một tập keypoint có tên theo
  thứ tự, vì vậy đầu ra mang cấu trúc bên trong của đối tượng thay vì chỉ phạm
  vi của nó. Key tác vụ là pose.
keywords:
  - ước lượng tư thế python
  - phát hiện keypoint
  - mô hình human pose
  - COCO keypoint
  - OKS mAP
  - huấn luyện mô hình pose
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Hậu tố -pose trong tên tệp chọn keypoint head, vì vậy không cần
        # đối số task.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy.shape)   # tọa độ pixel (N, K, 2)
        print(result.boxes.xyxy.shape)     # (N, 4), cùng N thực thể
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs-pose.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Chỉ keypoint hiển thị
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreECs-pose.pt")(SAMPLE_IMAGE)
        kpts = result.keypoints

        # .has_visible được suy ra từ cột keypoint thứ ba, và toàn True
        # khi checkpoint chỉ dự đoán (x, y).
        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: Dùng top-down
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # HRNet dùng top-down: trước hết crop từng người. Khi không cung cấp

        # nguồn người, mô hình tự ghép với detector LibreYOLO9t và ghi lựa chọn
        vào log.

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # coco8-pose.yaml chứa script tải nhúng, vì vậy cần quyền tường minh
        # trừ khi dữ liệu đã có cục bộ.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="coco8-pose.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            allow_download_scripts=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs-pose.pt data=coco8-pose.yaml \
          epochs=50 imgsz=640 batch=4 allow_download_scripts=True
    - label: Dataset riêng
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml phải khai báo kpt_shape, còn các dòng nhãn phải chứa
        # đúng 5 + K * D trường.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreECs-pose.pt")


        # val() trả về dict thuần, không phải đối tượng.

        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True)


        print(metrics["metrics/keypoints_mAP50-95"])

        print(metrics["metrics/keypoints_mAP50"],
        metrics["metrics/keypoints_mAP75"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs-pose.pt data=coco8-pose.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, vì vậy artifact đã xuất được nạp
        # như checkpoint và trả về cùng một đối tượng Results.
        model = LibreYOLO("LibreECs-pose.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
source_hash: 9de01d1f615bdf33
---

## Định nghĩa

Ước lượng tư thế trả về cấu trúc, không chỉ phạm vi. Mỗi thực thể vẫn nhận một
hộp, lớp đối tượng và điểm số, đồng thời nhận `K` keypoint theo thứ tự cố định,
vì vậy index 5 biểu thị cùng bộ phận cơ thể trên mọi thực thể và trong mọi ảnh.
Tập nhãn định nghĩa thứ tự đó; không nội dung nào trong đầu ra xác định keypoint
bằng tên.

`pose` là key tác vụ chuẩn, còn hậu tố `-pose` trong tên tệp checkpoint chọn tác
vụ, vì vậy không cần `task=` khi nạp trọng số đã công bố.

`predict()` điền `result.keypoints` bên cạnh `result.boxes`. `.data` có dạng
`(N, K, 2)` hoặc `(N, K, 3)`, các dòng được căn theo hộp, vì vậy thực thể `i`
trong một đối tượng là thực thể `i` trong đối tượng kia. `.xy` cắt tọa độ pixel,
còn `.xyn` chuẩn hóa chúng theo kích thước ảnh gốc. `.conf` là cột thứ ba khi
checkpoint dự đoán cột đó và là `None` khi không có, còn `.has_visible` là mặt
nạ boolean được suy ra, toàn True khi không có cột thứ ba.

Hai kiến trúc đi tới đầu ra này. Mô hình một giai đoạn dự đoán hộp và keypoint
trong một pass. Mô hình top-down chạy detector trước, crop từng thực thể rồi hồi
quy keypoint bên trong vùng crop, vì vậy độ chính xác phụ thuộc vào detector
phía trước.

## Mô hình

Ba family vừa huấn luyện vừa dự đoán: [RF-DETR](/docs/models/rf-detr),
[EdgeCrafter](/docs/models/edgecrafter) và
[YOLO-NAS](/docs/models/yolo-nas), đều là mô hình một giai đoạn. RF-DETR cần
thành phần bổ sung riêng `pip install "libreyolo[rfdetr]"`. RF-DETR và
EdgeCrafter cung cấp checkpoint pose đã công bố, cả hai đều tinh chỉnh trên
dataset một lớp chỉ có người; keypoint head của EdgeCrafter được cố định khi
dựng và từ chối dataset khai báo số lượng khác, còn RF-DETR khởi tạo lại head
cho một số lượng mới. YOLO-NAS lấy trọng số từ CDN riêng của Deci.AI theo giấy
phép phi thương mại, LibreYOLO không công bố tệp nào; pose head của nó cũng dựng
lại cho số keypoint mới, và đây là family duy nhất trong ba loại có số lớp đối
tượng không cố định ở một, vì vậy phù hợp cho skeleton nhiều lớp hoặc không
phải con người, như tư thế động vật.

[HRNet](/docs/models/hrnet) là lựa chọn top-down. Nó dự đoán, xác thực và xuất,
còn `train()` phát sinh `NotImplementedError`. Khi không có nguồn người, nó tự
ghép với detector LibreYOLO9t; `cropped=True` xử lý toàn ảnh như một thực thể,
`person_boxes=` nhận các hộp đã có, còn `person_detector=` đặt tên detector khác.

[SenseNova-Vision](/docs/models/sensenova-vision) cũng phát ra keypoint. Đây là
mô hình sinh theo prompt với factory riêng `LibreVLM` và thành phần bổ sung
riêng; khi không đặt từ vựng, `set_task("pose")` quay về category người. Trọng số
chỉ dùng cho mục đích phi thương mại, còn độ trễ trên mỗi ảnh cao hơn nhiều so
với pose head chuyên dụng vì mỗi dự đoán là một lượt diffusion decode.

## Dự đoán

Trọng số được tải từ Hugging Face trong lần sử dụng đầu tiên và lưu vào cache
cục bộ.

<code-tabs name="predict" />

Số lượng và thứ tự keypoint là thuộc tính checkpoint, không phải thư viện, vì
vậy mô hình được huấn luyện trên skeleton khác sẽ trả về `K` khác và ý nghĩa
theo index khác. Nội dung cột keypoint thứ ba cũng là thuộc tính checkpoint:
EdgeCrafter ghi một hằng số thay vì điểm theo từng điểm, và hoàn toàn không có
box head, vì vậy mỗi pose box là phạm vi bao của các keypoint riêng thuộc thực
thể đó. Xem [dự đoán](/docs/predict) để biết về nguồn, stream và cách xử lý kết
quả.

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

Một dòng là dòng phát hiện có keypoint được nối thêm:

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Số trường chính xác là `5 + K * D`, trong đó `D` là giá trị thứ hai của
`kpt_shape`. Tọa độ hộp và keypoint là số float đã chuẩn hóa tương đối với chiều
rộng cùng chiều cao ảnh gốc. Độ hiển thị `v`, chỉ có khi `D` là 3, nhận `0`, `1`
hoặc `2`.

YAML bổ sung hai key vào hợp đồng dùng chung:

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

`kpt_shape` là bắt buộc và có dạng `[K, 2]` hoặc `[K, 3]`. `flip_idx` là
permutation tùy chọn của `0..K-1`, cho biết với mỗi keypoint, index của nó sau
phép lật ngang, đây là cách cổ tay trái vẫn là cổ tay trái. Nếu bỏ key này,
augmentation lật ngang sẽ bị tắt cho keypoint thay vì áp dụng sai thứ tự index.

## Huấn luyện

<code-tabs name="train" />

Quá trình huấn luyện tiếp tục từ checkpoint `-pose` đã công bố, vốn đã mang
keypoint head; tác vụ được đọc từ checkpoint bạn nạp, không phải flag truyền tại
thời điểm huấn luyện, vì vậy checkpoint phát hiện không trở thành lượt chạy pose
chỉ bằng cách yêu cầu. `kpt_shape` trong YAML phải khớp chính xác với head của
EdgeCrafter vì head được cố định khi dựng, còn RF-DETR và YOLO-NAS đổi kích thước
head cho số lượng khác. Xem [huấn luyện](/docs/train) để biết về dataset,
augmentation, multi-GPU và logger.

## Xác thực

`val()` trả về dictionary thuần gồm các key `metrics/`. Cơ chế tính điểm là đánh
giá keypoint COCO trên Object Keypoint Similarity, gán trọng số cho sai số khoảng
cách của từng keypoint theo tỷ lệ thực thể và dung sai theo từng keypoint, vì vậy
nó đóng vai trò mà IoU đảm nhiệm cho hộp. Cơ chế cần `pycocotools`, có trong bản
cài đặt cơ sở.

<code-tabs name="val" />

`metrics/keypoints_mAP50-95` là con số chính, mean average precision lấy trung
bình trên các ngưỡng OKS từ 0.50 đến 0.95, và là số liệu quá trình huấn luyện
dùng để chọn epoch tốt nhất. `metrics/keypoints_mAP50` và
`metrics/keypoints_mAP75` là các phiên bản một ngưỡng, còn
`metrics/keypoints_mAP_M` và `metrics/keypoints_mAP_L` chia giá trị trung bình
theo diện tích thực thể, trung bình và lớn; đánh giá keypoint COCO không định
nghĩa nhóm nhỏ. Các số liệu average recall tương ứng là
`metrics/keypoints_AR50-95`, `metrics/keypoints_AR50`,
`metrics/keypoints_AR75`, `metrics/keypoints_AR_M` và
`metrics/keypoints_AR_L`. Mọi key trên tác vụ này đều có tiền tố `keypoints_`,
vì vậy các key `mAP` hộp mà detector trả về không xuất hiện.

## Xuất

<code-tabs name="export" />

Artifact đã xuất được nạp lại qua `LibreYOLO()` theo hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`. Phạm
vi định dạng khác nhau theo family; ma trận trên từng trang mô hình được tạo từ
tập đã xác thực thay vì nhập thủ công. Xem [xuất và triển khai](/docs/export) để
biết các định dạng, thành phần bổ sung và ràng buộc.
