---
title: Định dạng tập dữ liệu
seo_title: Định dạng tập dữ liệu LibreYOLO cho mọi tác vụ
description: >-
  Hợp đồng tệp cho tập dữ liệu (dataset) theo từng tác vụ chuẩn: khóa YAML, bố
  cục thư mục, hàng nhãn, quy ước mặt nạ (mask) và map, cùng trình tải đọc từng
  định dạng.
lead: >-
  Trang này phản ánh hợp đồng tệp dataset trong docs/dataset_schema.md của chính
  thư viện. Nội dung bao quát các khóa YAML và bố cục trên ổ đĩa mà mỗi tác vụ
  chuẩn yêu cầu.
keywords:
  - định dạng dataset libreyolo
  - định dạng nhãn yolo
  - data.yaml
  - dataset segmentation mask
  - định dạng coco panoptic
  - dataset độ sâu
  - pose kpt_shape
last_verified: 1.5.0
verification: >-
  Phản ánh docs/dataset_schema.md trong repo libreyolo tại v1.5.0, với tên các
  trình tải được đối chiếu với libreyolo/data/.
snippets:
  usage:
    - label: Phân tích một hàng nhãn phát hiện
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

## YAML chung

Áp dụng cho `detect`, `segment`, `pose` và `obb`.

| Khóa | Bắt buộc | Ý nghĩa |
|---|---|---|
| `path` | | Thư mục gốc của dataset |
| `train` | Khi huấn luyện | Ảnh huấn luyện |
| `val` | Khi xác thực | Ảnh xác thực |
| `test` | | Ảnh kiểm thử |
| `names` | Có | Danh sách lớp đối tượng hoặc ánh xạ với khóa số nguyên |
| `nc` | | Số lớp đối tượng; phải khớp với `names` khi có |
| `download` | | Hướng dẫn tải xuống; script Python cần được cho phép tường minh |
| `annotations` | | Ánh xạ split đến tệp JSON COCO gốc, dành cho detect, segment và obb |

`train`, `val` và `test` có thể là thư mục ảnh, tệp `.txt` liệt kê ảnh,
hoặc danh sách gồm các mục đó. Đường dẫn nhãn tuân theo một phép thay thế:

```text
images/.../image.jpg -> labels/.../image.txt
```

Đối với dataset JSON COCO gốc, `annotations` ánh xạ một split đến tệp JSON
của split đó, còn đường dẫn split chỉ định thư mục gốc của ảnh:

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Khi có `names`, tên category trong JSON COCO gốc phải khớp với tên lớp đối tượng
trong YAML, và những tên đó xác định ID nhãn của mô hình. Nếu không có `names`,
ID category COCO được sắp xếp và ánh xạ dày đặc vào `0..N-1`.

YAML của dataset không chứa khóa `task`. Lựa chọn tường minh về mô hình và tác vụ
được ưu tiên.

Các quy tắc chung cho mọi tệp nhãn văn bản:

- mỗi ảnh có một tệp nhãn `.txt`;
- tệp nhãn bị thiếu hoặc rỗng có nghĩa là không có đối tượng;
- `class_id` là số nguyên trong `0..nc-1`;
- tọa độ là số thực hữu hạn được chuẩn hóa trong `[0, 1]`;
- tọa độ tương ứng với chiều rộng và chiều cao của ảnh gốc;
- các hàng không chứa độ tin cậy hay track ID.

<code-tabs name="usage" />

## detect

Chính xác năm trường trên mỗi hàng:

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h` là một box thẳng trục đã chuẩn hóa, và `w` cùng `h` phải
dương.

## segment

Một hàng polygon:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N` ít nhất là 3, số tọa độ sau `class_id` phải là số chẵn, và polygon
không được suy biến. Một hàng phát hiện gồm năm trường cũng được chấp nhận và
biểu diễn một segment hình chữ nhật.

## pose

YAML bổ sung `kpt_shape`, đây là trường bắt buộc và có giá trị `[K, 2]` hoặc
`[K, 3]`, cùng `flip_idx` tùy chọn, một hoán vị số nguyên của `0..K-1`.

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Số trường chính xác là `5 + K * D`, trong đó `D` là giá trị thứ hai của
`kpt_shape`. Tọa độ keypoint được chuẩn hóa. Giá trị visibility `v`, khi có,
là `0`, `1` hoặc `2`.

## obb

Chính xác chín trường:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Bốn điểm là tọa độ ảnh được chuẩn hóa trong `[0, 1]` và tạo thành một hình chữ
nhật có hướng, không suy biến. Tệp nhãn không lưu góc.

Theo mặc định, parser chuẩn hoạt động nghiêm ngặt và từ chối tọa độ nằm ngoài
phạm vi. Quá trình nạp dataset và dữ liệu xác thực có thể cắt tọa độ về `[0, 1]`
cho các nhãn hợp lệ khác nằm ở biên crop, sau đó vẫn từ chối các box suy biến.
Việc phân tích có nhận biết tác vụ: chín trường chỉ có nghĩa là `obb` trong chế độ
`obb`, còn trong chế độ `segment`, chúng có thể là polygon bốn điểm.

Bên trong, các góc được chuẩn hóa được chuyển thành `xywhr` chuẩn, với góc tính
bằng radian biểu diễn phép xoay của cạnh chiều rộng quanh tâm box. Kết quả công
khai cung cấp các phát hiện OBB dưới dạng hàng `xywhr, conf, cls`.

Quá trình tải OBB từ JSON COCO gốc chấp nhận nhãn theo thứ tự ưu tiên sau:
`obb` dưới dạng tám góc trong không gian pixel; `obb` dưới dạng
`[cx, cy, w, h, angle]` với góc tính bằng radian; polygon `segmentation` COCO
hoặc RLE, được khớp lại thành hình chữ nhật có diện tích nhỏ nhất; và `bbox`
COCO, được đọc dưới dạng thẳng trục rồi chuẩn hóa.

Mosaic và mixup bị tắt khi huấn luyện OBB cho đến khi có phép tăng cường dữ liệu (data augmentation) OBB
nhận biết góc.

Parser hàng chuẩn là `libreyolo.data.parse_yolo_obb_label_line`.

## semantic

Mỗi ảnh ghép với một mặt nạ dense một kênh ở định dạng không mất dữ liệu,
thường là PNG, thay cho tệp `.txt`:

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

Mặt nạ có một kênh, và PNG ở chế độ palette được đọc dưới dạng chỉ mục palette.
Mỗi giá trị pixel là ID lớp đối tượng trong `0..nc-1`, giá trị pixel `255` có
nghĩa là bỏ qua và bị loại khỏi loss cũng như metric, còn độ phân giải mặt nạ
phải bằng độ phân giải ảnh.

Hai khóa YAML tùy chọn được bổ sung vào hợp đồng chung. `masks_dir` là tên thư mục
mặt nạ thay thế `images` trong mỗi đường dẫn ảnh, mặc định là `masks`.
`label_mapping` là phép ánh xạ lại `{source_id: train_id}` áp dụng cho giá trị
pixel mặt nạ khi tải, trong đó giá trị nguồn không được ánh xạ sẽ thành giá trị
bỏ qua và train ID phải nằm trong `0..nc-1`.

Khi bỏ qua `masks_dir`, mặt nạ được raster hóa lúc tải từ nhãn polygon `segment`
được phân giải theo quy ước thay `images` bằng `labels`, và một lớp đối tượng
`background` được thêm sau các lớp đối tượng, vì vậy `nc` tăng thêm một.

Trình tải chuẩn: `libreyolo.data.SemanticDataset`.

## panoptic

LibreYOLO áp dụng nguyên trạng định dạng COCO-panoptic (Kirillov et al., CVPR
2019). Không có định dạng panoptic riêng của LibreYOLO.

Mỗi ảnh có một PNG RGB ở cùng độ phân giải với ảnh, mã hóa ID của mỗi segment
bằng màu của pixel:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Mỗi pixel thuộc chính xác một segment và các segment không bao giờ chồng lấp.
ID segment `0`, màu đen RGB, là void: các pixel không có nhãn bị loại khỏi metric.

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

`annotations[].file_name` chỉ định tên PNG segment-ID bên trong `panoptic_dir`,
và `segments_info[].id` khớp với một giá trị trong PNG đó. `iscrowd` đánh dấu
các vùng nhóm: chúng không bao giờ là false negative, và một dự đoán chủ yếu
bao phủ một vùng như vậy không phải là false positive.

Thing và stuff là thuộc tính theo từng category. `isthing` nằm trong
`categories`, không bao giờ nằm trong `segments_info`.

Các giá trị `category_id` COCO-panoptic là ID thô của dataset và thường không
liên tục. Mô hình dự đoán các ID liên tục `0..nc-1`, vì vậy ID thô được ánh xạ
lại qua `names` trong YAML theo tên category, cùng quy tắc mà trình tải detect
từ JSON COCO gốc tuân theo. Category JSON không có trong `names` là một lỗi
thay vì bị âm thầm loại bỏ, vì nếu không nó sẽ được tính như một false negative
vĩnh viễn.

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

Quá trình xác thực báo cáo Panoptic Quality, được tính ở độ phân giải ground truth
và lấy trung bình trên các category xuất hiện, sau đó chia thành `PQ_things` và
`PQ_stuff`. Phép khớp là duy nhất: segment dự đoán và segment ground truth cùng
category khớp nhau khi IoU lớn hơn 0.5.

Trình tải chuẩn: `libreyolo.data.PanopticDataset`.

## depth

Mỗi ảnh ghép với một depth map dense một kênh:

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

Map là PNG hoặc TIF một kênh, hoặc tệp `.npy`, ở độ phân giải của ảnh.
Các giá trị là độ sâu thuần túy theo một đơn vị nhất quán trong dataset. Giá trị
không, âm, NaN và vô hạn đánh dấu pixel không hợp lệ và bị loại khỏi loss cũng
như metric.

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `depths_dir` | `depths` | Thư mục độ sâu thay thế `images` |
| `depth_stem_suffix` | | Hậu tố nối vào stem của ảnh; khi bỏ qua, cả cùng stem và hậu tố `_depth` đều được thử |
| `depth_mask_suffix` | `_mask` | Hậu tố cho mặt nạ hợp lệ; giá trị mặt nạ bằng hoặc nhỏ hơn không, NaN và vô hạn làm pixel độ sâu mất hiệu lực |
| `depth_scale` | `256.0` | Số chia cho depth map kiểu số nguyên, theo quy ước PNG 16 bit phổ biến |

Map `.npy` kiểu float được dùng nguyên trạng và không áp dụng `depth_scale`.

Trình tải chuẩn: `libreyolo.data.DepthDataset`.

## edge

Mỗi ảnh RGB ghép với một map không mất dữ liệu một kênh có cùng stem và một
mặt nạ hợp lệ tùy chọn:

```text
images/val/scene.jpg -> edges/val/scene.png
                     -> masks/val/scene.png
```

Map là PNG hoặc TIF một kênh, không phải hình trực quan RGB, và có độ phân giải
của ảnh. Map số nguyên được chia cho giá trị lớn nhất của dtype; map float phải
sẵn hữu hạn và nằm trong `[0, 1]`. `0` nghĩa là không phải edge và `1` nghĩa là
edge. Pixel mặt nạ tùy chọn hợp lệ khi khác không. Việc đổi kích thước dùng nội
suy nearest-neighbor cho target và mặt nạ, còn pixel được đệm không hợp lệ và
không đóng góp vào quá trình xác thực.

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `edges_dir` | `edges` | Thư mục edge map thay thế `images` |
| `edge_stem_suffix` | | Hậu tố nối vào stem của ảnh |
| `edge_extension` | `.png` | Phần mở rộng target không mất dữ liệu |
| `edge_invert` | | Đặt thành true khi map nguồn lưu edge đen trên nền trắng |
| `masks_dir` | `masks` | Thư mục mặt nạ hợp lệ tùy chọn |

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Quá trình xác thực làm mảnh dự đoán liên tục bằng non-maximum suppression gradient
bốn hướng và báo cáo F-measure ODS cùng OIS qua một dải ngưỡng có thể cấu hình.
Pixel dự đoán và pixel ground truth được khớp một-một trong phạm vi
`edge_max_dist * image_diagonal`, với dung sai chuẩn hóa mặc định là `0.0075`.

Trình tải chuẩn: `libreyolo.data.EdgeDataset`. Trình tải chỉ xử lý định dạng:
nó không tải xuống hay phân phối lại dữ liệu benchmark.

## normal

Mỗi ảnh ghép với một PNG 16 bit ba kênh có cùng stem, cùng một mặt nạ hợp lệ
cùng stem tùy chọn:

```text
images/val/room.jpg -> normals/val/room.png
                    -> masks/val/room.png
```

PNG có chính xác ba kênh `uint16` được lưu dưới dạng RGB, ở độ phân giải của ảnh.
Giải mã bằng `n = png / 65535 * 2 - 1`, sau đó chuẩn hóa lại từng vector.
Các vector đã giải mã dùng hệ tọa độ camera OpenCV, `+x` sang phải, `+y` xuống dưới,
`+z` hướng vào cảnh, và quay về phía camera. Mặt nạ tùy chọn là PNG một kênh,
trong đó khác không nghĩa là hợp lệ; khi không có mặt nạ, mọi vector đã giải mã
hữu hạn và khác không đều hợp lệ. Pixel target không hợp lệ và pixel được đệm
được biểu diễn nội bộ bằng `(0, 0, 0)`. Khi đổi kích thước, ba thành phần được nội
suy bilinear rồi chuẩn hóa lại, mặt nạ hợp lệ dùng nội suy nearest-neighbor, và
phép lật ngang cũng đảo dấu thành phần x.

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `normals_dir` | `normals` | Thư mục normal map thay thế `images` |
| `masks_dir` | `masks` | Thư mục mặt nạ hợp lệ tùy chọn |

Quá trình xác thực báo cáo sai số góc trung bình và trung vị theo độ, cùng tỷ lệ
pixel hợp lệ nằm trong các mức 11.25, 22.5 và 30 độ.

Trình tải chuẩn: `libreyolo.data.NormalDataset`.

## restore

Mỗi ảnh đầu vào bị suy giảm ghép với một target RGB sạch:

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

Đầu vào và target là các tệp ảnh tương thích RGB, và độ phân giải của chúng phải
khớp hoàn toàn. Quá trình xác thực giữ nguyên độ phân giải gốc và chỉ đệm vừa đủ
để xếp chồng thành một batch, còn các metric được tính trên canvas ảnh gốc. Quá
trình huấn luyện áp dụng crop kết hợp và lật ngang cho cặp đầu vào và target.

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `input_dir` | `inputs` | Thư mục đầu vào bị suy giảm được dùng trong đường dẫn split |
| `target_dir` | `targets` | Thư mục target sạch thay thế `input_dir` |
| `target_stem_suffix` | | Hậu tố nối vào stem đầu vào trước khi tra cứu target |
| `target_stem_suffixes` | | Dạng danh sách của `target_stem_suffix` |
| `degradation` | | Nhãn metadata như `deblur` hoặc `denoise` |
| `dataset` | | Nhãn dataset hoặc nguồn gốc |

Các trường YAML giống lớp đối tượng là giá trị giữ chỗ trong lược đồ: dùng `nc: 1`
và `names: {0: image}`. Mô hình restore cung cấp `Results.restored`, không phải
phát hiện.

Trình tải chuẩn: `libreyolo.data.RestoreDataset`.

## matte

Mỗi ảnh RGB ghép với một matte ground truth một kênh có cùng stem,
trong đó 0 là nền và 255 là tiền cảnh:

```text
images/subject.jpg -> mattes/subject.png
```

Hai bố cục được chấp nhận. Một thư mục gốc chứa `images/` và một thư mục matte,
được tự động phát hiện trong số `mattes/`, `matte/`, `gt/`, `masks/`, `mask/`
và `alpha/`, rồi truyền dưới dạng `data=`. Hoặc một YAML có `path` cùng
`val_images` và `val_mattes` theo từng split, và tùy chọn `train_images` cùng
`train_mattes`, mỗi giá trị là tương đối với `path` hoặc là đường dẫn tuyệt đối.

Matte ở thang xám và được đọc dưới dạng độ mờ trong `[0, 1]`, rồi được đổi kích
thước về canvas dự đoán bằng nội suy bilinear khi shape khác nhau. Metric là MAE
và S-measure (Fan et al., ICCV 2017) trên canvas ảnh gốc, với S-measure là độ
thích nghi để chọn checkpoint tốt nhất.

Các trường YAML giống lớp đối tượng là giá trị giữ chỗ trong lược đồ: dùng `nc: 1`
và `names: {0: matte}`. Mô hình matte cung cấp `Results.matte`.

Trong phiên bản này, quá trình xác thực chỉ dành cho suy luận (inference). Trình phân giải
cặp chuẩn: `libreyolo.data.matte_dataset.resolve_matte_pairs`.

## ocr

Nhãn là một tệp JSONL cho mỗi split, với một đối tượng JSON cho mỗi ảnh:

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` là tứ giác bốn điểm theo tọa độ pixel tuyệt đối, được sắp theo thứ tự
trên-trái, trên-phải, dưới-phải, dưới-trái. Các vùng có văn bản không đọc được
dùng `"text": "###"`, quy ước do-not-care của ICDAR: chúng bị loại khỏi việc
chấm điểm nhận dạng, và các dự đoán chồng lấp lên chúng được bỏ qua thay vì bị
phạt khi khớp phát hiện.

Các metric gồm hmean phát hiện với phép khớp polygon một-một trên IoU 0.5,
F1 end-to-end yêu cầu cả IoU trên 0.5 lẫn transcript khớp chính xác sau khi
chuẩn hóa NFKC và loại bỏ khoảng trắng, có phân biệt chữ hoa chữ thường, cùng
1-NED trên các cặp đã khớp. Độ thích nghi để chọn checkpoint tốt nhất là F1
end-to-end.

Hai bố cục được chấp nhận: thư mục gốc chứa `images/<split>/` và
`labels/<split>.jsonl`, được truyền dưới dạng `data=`, hoặc YAML có `path` cùng
tên thư mục `images` và `labels` tùy chọn.

Các trường YAML giống lớp đối tượng là giá trị giữ chỗ trong lược đồ: dùng `nc: 1`
và `names: {0: text}`. Mô hình OCR cung cấp `Results.ocr`.

Trong phiên bản này, quá trình xác thực chỉ dành cho inference. Trình phân giải
mẫu chuẩn: `libreyolo.data.ocr_dataset.resolve_ocr_samples`.

## classify

Cây thư mục kiểu ImageFolder, không phải các tệp nhãn:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

`train/` là bắt buộc khi huấn luyện và xác định ánh xạ từ lớp đối tượng đến chỉ
mục bằng tên thư mục đã sắp xếp. `val/` là bắt buộc để xác thực. Có thể có
`test/`, nhưng các lệnh train và val mặc định không dùng nó. Các split không
dùng để huấn luyện phải chứa cùng tên thư mục lớp đối tượng như tập lớp đối tượng
train hoặc checkpoint dự kiến. Các phần mở rộng ảnh được hỗ trợ được định nghĩa
trong `libreyolo.data.classify_dataset.IMAGE_EXTENSIONS`.

## gaze và point

Chưa triển khai hợp đồng tệp dataset để huấn luyện hoặc xác thực cho `gaze`.

`point` là tác vụ đầu ra mô hình chứ không phải lược đồ nhãn dataset. Các họ point
có thể điều chỉnh nhãn hiện có ở bên trong, ví dụ bằng cách suy ra tâm đối tượng
từ các hàng box, nhưng định dạng nhãn văn bản chỉ dành cho point chưa được định nghĩa.
