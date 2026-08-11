---
title: Định dạng tập dữ liệu
seo_title: Định dạng tập dữ liệu LibreYOLO cho mọi tác vụ
description: >-
  Quy ước tệp tập dữ liệu cho từng tác vụ chuẩn: khóa YAML, cấu trúc thư mục,
  hàng nhãn, quy ước mask và map, cùng loader đọc từng định dạng.
lead: >-
  Trang này phản ánh quy ước tệp tập dữ liệu trong chính docs/dataset_schema.md
  của thư viện. Nội dung trình bày các khóa YAML và cấu trúc trên ổ đĩa mà từng
  tác vụ chuẩn yêu cầu.
keywords:
  - định dạng tập dữ liệu libreyolo
  - định dạng nhãn yolo
  - cấu hình data.yaml
  - tập dữ liệu mask phân đoạn
  - định dạng panoptic coco
  - tập dữ liệu độ sâu
  - kpt_shape pose
last_verified: 1.5.0
verification: >-
  Phản ánh docs/dataset_schema.md trong repo libreyolo ở phiên bản v1.5.0, với
  tên loader được đối chiếu theo libreyolo/data/.
snippets:
  usage:
    - label: Phân tích một hàng nhãn detection
      language: python
      code: >
        from libreyolo.data import parse_yolo_label_line


        # class_id cx cy w h, được chuẩn hóa về [0, 1]

        row = parse_yolo_label_line("0 0.5 0.5 0.25 0.5", 640, 480,
        num_classes=80)


        # (class_id, x1, y1, x2, y2, area) theo pixel

        print(row)
source_hash: a8282c079624044d
---

## YAML dùng chung

Áp dụng cho `detect`, `segment`, `pose` và `obb`.

| Khóa | Bắt buộc | Ý nghĩa |
|---|---|---|
| `path` | | Thư mục gốc của tập dữ liệu |
| `train` | Khi huấn luyện | Ảnh huấn luyện |
| `val` | Khi xác thực | Ảnh xác thực |
| `test` | | Ảnh kiểm thử |
| `names` | Có | Danh sách lớp hoặc ánh xạ với khóa là số nguyên |
| `nc` | | Số lớp; phải khớp với `names` khi có |
| `download` | | Chỉ dẫn tải xuống; script Python cần được cho phép rõ ràng |
| `annotations` | | Ánh xạ split đến tệp COCO JSON gốc, dành cho detect, segment và obb |

`train`, `val` và `test` có thể là thư mục ảnh, tệp `.txt` liệt kê ảnh,
hoặc danh sách các mục đó. Đường dẫn nhãn tuân theo một phép thay thế:

```text
images/.../image.jpg -> labels/.../image.txt
```

Với tập dữ liệu COCO JSON gốc, `annotations` ánh xạ một split đến tệp JSON
của nó và đường dẫn split chỉ định thư mục gốc chứa ảnh:

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Khi có `names`, tên category trong COCO JSON gốc phải khớp với tên lớp trong
YAML, và các tên đó xác định ID nhãn của mô hình. Khi không có `names`, các
category ID của COCO được sắp xếp rồi ánh xạ liên tục vào `0..N-1`.

YAML của tập dữ liệu không chứa khóa `task`. Việc chọn rõ mô hình và tác vụ
được ưu tiên.

Các quy tắc dùng chung cho mọi tệp nhãn văn bản:

- mỗi ảnh có một tệp nhãn `.txt`;
- tệp nhãn bị thiếu hoặc rỗng nghĩa là không có đối tượng;
- `class_id` là số nguyên trong `0..nc-1`;
- tọa độ là số thực hữu hạn được chuẩn hóa trong `[0, 1]`;
- tọa độ được tính tương đối theo chiều rộng và chiều cao ảnh gốc;
- các hàng không chứa độ tin cậy hoặc track ID.

<code-tabs name="usage" />

## detect

Mỗi hàng có đúng năm trường:

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h` là bounding box thẳng trục đã chuẩn hóa, còn `w` và `h` phải
dương.

## segment

Một hàng đa giác:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N` ít nhất là 3, số tọa độ sau `class_id` phải chẵn và đa giác không được
suy biến. Hàng detection gồm năm trường cũng được chấp nhận và biểu diễn một
vùng phân đoạn hình chữ nhật.

## pose

YAML bổ sung `kpt_shape`, đây là trường bắt buộc và có dạng `[K, 2]` hoặc
`[K, 3]`, cùng `flip_idx` tùy chọn, một hoán vị số nguyên của `0..K-1`.

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Số trường chính xác là `5 + K * D`, trong đó `D` là giá trị thứ hai của
`kpt_shape`. Tọa độ keypoint được chuẩn hóa. Giá trị hiển thị `v`, khi có,
là `0`, `1` hoặc `2`.

## obb

Có đúng chín trường:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Bốn điểm là tọa độ ảnh đã chuẩn hóa trong `[0, 1]` và tạo thành một hình chữ
nhật xoay không suy biến. Tệp nhãn không lưu góc.

Theo mặc định, parser chuẩn áp dụng nghiêm ngặt và từ chối tọa độ nằm ngoài
phạm vi. Quá trình nạp tập dữ liệu và xác thực có thể cắt tọa độ về `[0, 1]`
đối với nhãn hợp lệ ở biên crop, sau đó vẫn từ chối các box suy biến. Việc
phân tích có nhận biết tác vụ: chín trường chỉ mang nghĩa `obb` trong chế độ
`obb`, còn ở chế độ `segment` chúng có thể là một đa giác bốn điểm.

Bên trong, các góc đã chuẩn hóa được chuyển thành `xywhr` chuẩn, với góc tính
bằng radian biểu diễn phép xoay cạnh chiều rộng quanh tâm box. Kết quả công
khai biểu diễn detection OBB dưới dạng các hàng `xywhr, conf, cls`.

Quá trình nạp OBB từ COCO JSON gốc chấp nhận annotation theo thứ tự ưu tiên
sau: `obb` gồm tám góc theo không gian pixel; `obb` dạng
`[cx, cy, w, h, angle]` với góc tính bằng radian; đa giác hoặc RLE
`segmentation` của COCO được khớp lại thành hình chữ nhật có diện tích nhỏ
nhất; và `bbox` COCO, được đọc dưới dạng thẳng trục rồi chuẩn hóa.

Mosaic và mixup bị tắt khi huấn luyện OBB cho đến khi có data augmentation
OBB nhận biết các góc.

Parser hàng chuẩn là `libreyolo.data.parse_yolo_obb_label_line`.

## semantic

Mỗi ảnh được ghép với một mask một kênh dày đặc ở định dạng không mất dữ liệu,
thường là PNG, thay vì tệp `.txt`:

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

Mask có một kênh, còn PNG ở chế độ bảng màu được đọc dưới dạng chỉ số bảng
màu. Mỗi giá trị pixel là một ID lớp trong `0..nc-1`, giá trị pixel `255`
nghĩa là bỏ qua nên không tham gia tính loss và metric, đồng thời độ phân giải
mask phải bằng độ phân giải ảnh.

Hai khóa YAML tùy chọn được bổ sung trên quy ước chung. `masks_dir` là tên thư
mục mask dùng để thay thế `images` trong từng đường dẫn ảnh, mặc định là
`masks`. `label_mapping` là ánh xạ lại `{source_id: train_id}` được áp dụng
cho giá trị pixel của mask khi nạp, trong đó giá trị nguồn không được ánh xạ
sẽ bị bỏ qua và train ID phải nằm trong `0..nc-1`.

Khi bỏ qua `masks_dir`, mask được raster hóa lúc nạp từ nhãn đa giác `segment`
được phân giải theo quy ước chuyển `images` thành `labels`, và một lớp
`background` được nối sau các lớp đối tượng nên `nc` tăng thêm một.

Loader chuẩn: `libreyolo.data.SemanticDataset`.

## panoptic

LibreYOLO sử dụng nguyên trạng định dạng COCO-panoptic (Kirillov và cộng sự,
CVPR 2019). Không có định dạng panoptic riêng của LibreYOLO.

Mỗi ảnh có một PNG RGB ở đúng độ phân giải của ảnh, mã hóa segment ID của từng
pixel trong màu sắc:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Mỗi pixel thuộc chính xác một segment và các segment không bao giờ chồng lấn.
Segment ID `0`, màu đen RGB, là vùng trống: các pixel không có nhãn bị loại
khỏi metric.

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1, "supercategory": "person"}]
}
```

`annotations[].file_name` đặt tên cho PNG segment-ID bên trong `panoptic_dir`,
và `segments_info[].id` khớp với một giá trị trong PNG đó. `iscrowd` đánh dấu
các vùng nhóm: chúng không bao giờ là false negative, còn một dự đoán phủ phần
lớn một vùng như vậy không bị tính là false positive.

Phân biệt thing với stuff là thuộc tính theo từng category. `isthing` nằm trong
`categories`, không bao giờ nằm trong `segments_info`.

Giá trị `category_id` của COCO-panoptic là ID thô của tập dữ liệu và thường
không liên tục. Mô hình dự đoán dải liên tục `0..nc-1`, vì vậy ID thô được ánh
xạ lại qua `names` trong YAML theo tên category, cùng quy tắc mà loader detect
COCO JSON gốc tuân theo. Một category JSON không có trong `names` là lỗi thay
vì bị loại bỏ âm thầm, bởi nếu không nó sẽ luôn bị chấm là false negative.

```yaml
path: coco
val: images/val2017
annotations:
  val: annotations/panoptic_val2017.json
panoptic_dir:
  val: annotations/panoptic_val2017
names: {0: person, 1: bicycle, 132: rug-merged}
```

`annotations` và `panoptic_dir` chấp nhận một đường dẫn đơn hoặc ánh xạ theo
từng split.

Quá trình xác thực báo cáo Panoptic Quality, được tính ở độ phân giải ground
truth rồi lấy trung bình trên các category xuất hiện, sau đó tách thành
`PQ_things` và `PQ_stuff`. Phép ghép là duy nhất: một segment dự đoán và một
segment ground truth thuộc cùng category được ghép khi IoU lớn hơn 0.5.

Loader chuẩn: `libreyolo.data.PanopticDataset`.

## depth

Mỗi ảnh được ghép với một depth map một kênh dày đặc:

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

Map là PNG hoặc TIF một kênh, hoặc tệp `.npy`, ở độ phân giải ảnh. Các giá trị
là độ sâu thuần túy theo một đơn vị nhất quán trong tập dữ liệu. Giá trị bằng
không, âm, NaN và vô hạn đánh dấu pixel không hợp lệ nên bị loại khỏi loss và
metric.

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `depths_dir` | `depths` | Thư mục độ sâu thay thế cho `images` |
| `depth_stem_suffix` | | Hậu tố nối vào stem của ảnh; khi bỏ trống, cả stem giống hệt và hậu tố `_depth` đều được thử |
| `depth_mask_suffix` | `_mask` | Hậu tố cho mask hợp lệ; giá trị mask không lớn hơn không, NaN và vô hạn làm pixel độ sâu mất hiệu lực |
| `depth_scale` | `256.0` | Số chia cho depth map kiểu số nguyên, theo quy ước PNG 16 bit phổ biến |

Map `.npy` kiểu số thực được dùng nguyên trạng và không áp dụng `depth_scale`.

Loader chuẩn: `libreyolo.data.DepthDataset`.

## edge

Mỗi ảnh RGB được ghép với một map không mất dữ liệu một kênh có cùng stem và
một mask hợp lệ tùy chọn:

```text
images/val/scene.jpg -> edges/val/scene.png
                     -> masks/val/scene.png
```

Map là PNG hoặc TIF một kênh, không phải hình trực quan hóa RGB, và có cùng độ
phân giải với ảnh. Map số nguyên được chia cho giá trị cực đại của dtype; map
số thực phải hữu hạn sẵn và nằm trong `[0, 1]`. `0` nghĩa là không phải cạnh,
còn `1` nghĩa là cạnh. Pixel của mask tùy chọn hợp lệ khi khác không. Quá trình
đổi kích thước dùng phép nội suy lân cận gần nhất cho target và mask, còn pixel
được đệm không hợp lệ và không tham gia xác thực.

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `edges_dir` | `edges` | Thư mục edge map thay thế cho `images` |
| `edge_stem_suffix` | | Hậu tố nối vào stem ảnh |
| `edge_extension` | `.png` | Phần mở rộng không mất dữ liệu của target |
| `edge_invert` | | Đặt true khi map nguồn lưu cạnh màu đen trên nền trắng |
| `masks_dir` | `masks` | Thư mục mask hợp lệ tùy chọn |

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Quá trình xác thực làm mảnh dự đoán liên tục bằng non-maximum suppression
gradient bốn hướng và báo cáo F-measure ODS và OIS trên một dải quét ngưỡng có
thể cấu hình. Pixel dự đoán và ground truth được ghép một-một trong phạm vi
`edge_max_dist * image_diagonal`, với dung sai chuẩn hóa mặc định là `0.0075`.

Loader chuẩn: `libreyolo.data.EdgeDataset`. Loader chỉ xử lý định dạng: nó
không tải xuống hoặc phân phối lại dữ liệu benchmark.

## normal

Mỗi ảnh được ghép với một PNG 16 bit ba kênh có cùng stem, cộng thêm mask hợp
lệ cùng stem nếu có:

```text
images/val/room.jpg -> normals/val/room.png
                    -> masks/val/room.png
```

PNG có đúng ba kênh `uint16` được lưu dưới dạng RGB ở độ phân giải ảnh. Giải mã
bằng `n = png / 65535 * 2 - 1`, sau đó chuẩn hóa lại từng vector. Vector đã
giải mã dùng hệ tọa độ camera OpenCV, `+x` sang phải, `+y` xuống dưới, `+z`
hướng vào cảnh và quay về phía camera. Mask tùy chọn là PNG một kênh, trong đó
khác không nghĩa là hợp lệ; khi không có mask, mọi vector hữu hạn và khác không
sau khi giải mã đều hợp lệ. Pixel target không hợp lệ và được đệm được biểu
diễn nội bộ bằng `(0, 0, 0)`. Việc đổi kích thước nội suy song tuyến tính ba
thành phần rồi chuẩn hóa lại, mask hợp lệ dùng phép nội suy lân cận gần nhất,
và phép lật ngang cũng đổi dấu thành phần x.

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `normals_dir` | `normals` | Thư mục normal map thay thế cho `images` |
| `masks_dir` | `masks` | Thư mục mask hợp lệ tùy chọn |

Quá trình xác thực báo cáo sai số góc trung bình và trung vị theo độ, cùng tỷ lệ
pixel hợp lệ nằm trong 11.25, 22.5 và 30 độ.

Loader chuẩn: `libreyolo.data.NormalDataset`.

## restore

Mỗi ảnh đầu vào bị suy giảm được ghép với một target RGB sạch:

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

Đầu vào và target là các tệp ảnh tương thích RGB, đồng thời độ phân giải của
chúng phải khớp chính xác. Quá trình xác thực giữ độ phân giải gốc và chỉ đệm
đủ để xếp thành batch, còn metric được tính trên canvas ảnh gốc. Quá trình
huấn luyện áp dụng đồng bộ phép crop và lật ngang cho cặp đầu vào và target.

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `input_dir` | `inputs` | Thư mục đầu vào bị suy giảm dùng trong đường dẫn split |
| `target_dir` | `targets` | Thư mục target sạch thay thế cho `input_dir` |
| `target_stem_suffix` | | Hậu tố nối vào stem đầu vào trước khi tìm target |
| `target_stem_suffixes` | | Dạng danh sách của `target_stem_suffix` |
| `degradation` | | Nhãn metadata như `deblur` hoặc `denoise` |
| `dataset` | | Nhãn tập dữ liệu hoặc nguồn gốc |

Các trường YAML giống lớp là phần giữ chỗ của schema: dùng `nc: 1` và
`names: {0: image}`. Mô hình restoration cung cấp `Results.restored`, không
phải detection.

Loader chuẩn: `libreyolo.data.RestoreDataset`.

## matte

Mỗi ảnh RGB được ghép với một matte ground truth một kênh có cùng stem, trong
đó 0 là nền và 255 là tiền cảnh:

```text
images/subject.jpg -> mattes/subject.png
```

Hai cấu trúc được chấp nhận. Cấu trúc thứ nhất là thư mục gốc chứa `images/`
và một thư mục matte được tự động phát hiện trong số `mattes/`, `matte/`,
`gt/`, `masks/`, `mask/` và `alpha/`, rồi được truyền dưới dạng `data=`. Cấu
trúc thứ hai là YAML có `path` cùng `val_images` và `val_mattes` cho mỗi split,
cũng như `train_images` và `train_mattes` tùy chọn, mỗi đường dẫn tương đối với
`path` hoặc là đường dẫn tuyệt đối.

Matte là ảnh thang độ xám được đọc dưới dạng độ mờ trong `[0, 1]`, và được đổi
kích thước theo canvas dự đoán bằng phép nội suy song tuyến tính khi hình dạng
khác nhau. Các metric là MAE và S-measure (Fan và cộng sự, ICCV 2017) trên
canvas ảnh gốc, trong đó S-measure là fitness để chọn checkpoint tốt nhất.

Các trường YAML giống lớp là phần giữ chỗ của schema: dùng `nc: 1` và
`names: {0: matte}`. Mô hình matte cung cấp `Results.matte`.

Trong phiên bản này, xác thực chỉ dành cho inference. Bộ phân giải cặp chuẩn:
`libreyolo.data.matte_dataset.resolve_matte_pairs`.

## ocr

Nhãn là một tệp JSONL cho mỗi split, với một đối tượng JSON cho mỗi ảnh:

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` là một tứ giác bốn điểm theo tọa độ pixel tuyệt đối, được sắp xếp
theo thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái. Vùng chứa văn bản không
đọc được dùng `"text": "###"`, theo quy ước do-not-care của ICDAR: chúng bị loại
khỏi điểm số nhận dạng, còn dự đoán chồng lên chúng được bỏ qua thay vì bị phạt
trong quá trình ghép detection.

Các metric gồm hmean detection với phép ghép đa giác một-một trên IoU 0.5, F1
đầu cuối yêu cầu cả IoU trên 0.5 và bản chép chính xác sau khi chuẩn hóa NFKC và
loại bỏ khoảng trắng, có phân biệt chữ hoa chữ thường, cùng 1-NED trên các cặp
được ghép. Fitness để chọn checkpoint tốt nhất là F1 đầu cuối.

Hai cấu trúc được chấp nhận: thư mục gốc chứa `images/<split>/` và
`labels/<split>.jsonl`, được truyền dưới dạng `data=`, hoặc YAML có `path` cùng
tên thư mục `images` và `labels` tùy chọn.

Các trường YAML giống lớp là phần giữ chỗ của schema: dùng `nc: 1` và
`names: {0: text}`. Mô hình OCR cung cấp `Results.ocr`.

Trong phiên bản này, xác thực chỉ dành cho inference. Bộ phân giải mẫu chuẩn:
`libreyolo.data.ocr_dataset.resolve_ocr_samples`.

## classify

Cây thư mục theo kiểu ImageFolder, không phải các tệp nhãn:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

`train/` là bắt buộc khi huấn luyện và xác định ánh xạ lớp đến chỉ số theo tên
thư mục đã sắp xếp. `val/` là bắt buộc khi xác thực. Có thể có `test/`, nhưng
các lệnh train và val mặc định không sử dụng nó. Các split không dùng để huấn
luyện phải chứa cùng tên thư mục lớp với tập lớp dự kiến từ train hoặc
checkpoint. Các phần mở rộng ảnh được hỗ trợ được định nghĩa trong
`libreyolo.data.classify_dataset.IMAGE_EXTENSIONS`.

## gaze và point

Chưa triển khai quy ước tệp tập dữ liệu dùng cho huấn luyện hoặc xác thực
`gaze`.

`point` là một tác vụ đầu ra của mô hình thay vì một schema nhãn tập dữ liệu.
Các họ point có thể điều chỉnh nhãn hiện có trong nội bộ, chẳng hạn bằng cách
suy ra tâm đối tượng từ các hàng box, nhưng chưa định nghĩa định dạng nhãn văn
bản chỉ chứa point.
