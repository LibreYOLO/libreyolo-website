---
title: NVIDIA DeepStream
seo_title: Chạy mô hình YOLO trên NVIDIA DeepStream
description: >-
  Xuất một mô hình LibreYOLO cho NVIDIA DeepStream: một đồ thị ONNX cùng một tệp
  config nvinfer được sinh ra. Các lệnh chính xác để build parser và chạy
  pipeline.
lead: >-
  NVIDIA DeepStream chạy suy luận (inference) qua phần tử nvinfer của nó, thứ
  cần một đồ thị ONNX, một tệp cấu hình tương ứng và một parser bounding box.
  Đặt deepstream=True khi xuất sang ONNX sẽ ghi ra hai thành phần đầu và nối
  chúng với thành phần thứ ba.
keywords:
  - deepstream yolo
  - xuất yolo sang deepstream
  - config nvinfer cho yolo
  - parser bounding box deepstream
  - config_infer_primary
  - NvDsInferParseYolo
  - deepstream-app
  - engine tensorrt deepstream
  - deepstream trên jetson
meta:
  - label: Tham số
    value: 'export(format="onnx", deepstream=True)'
    mono: true
  - label: Kết quả ghi ra
    value: 'Một đồ thị ONNX, config_infer_primary_<stem>.txt và <stem>_labels.txt'
  - label: Phạm vi
    value: 43 tổ hợp họ mô hình và tác vụ trải trên chín tác vụ
  - label: Parser
    value: >-
      NvDsInferParseYolo, từ dự án DeepStream-Yolo của Marcos Luciano, giấy phép
      MIT. Build một lần cho mỗi thiết bị.
    links:
      - label: github.com/marcoslucianops/DeepStream-Yolo
        href: 'https://github.com/marcoslucianops/DeepStream-Yolo'
  - label: Khả dụng
    value: Có trong v1.5.0. Đã merge vào dev ngày 2026-08-08 trong pull request 728.
    links:
      - label: pull request 728
        href: 'https://github.com/LibreYOLO/libreyolo/pull/728'
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
  - label: Đã kiểm chứng khi chạy
    value: 'DeepStream 8.0.0 trên một RTX 5070 Ti, chỉ phát hiện đối tượng, 2026-08-08'
verification: >-
  Viết ra từ lần kiểm chứng chạy thực tế ngày 2026-08-08. Danh sách họ mô hình,
  các khóa config và giá trị mặc định được đọc từ libreyolo/export/deepstream.py
  và libreyolo/export/exporter.py tại commit 5f81e11e, bản đã merge vào dev cùng
  ngày trong pull request 728.
snippets:
  install:
    - label: Cài đặt
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO9, LibreDFINE


        # Ghi libreyolo9s.onnx, config_infer_primary_libreyolo9s.txt

        # và libreyolo9s_labels.txt vào thư mục làm việc

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="onnx",
        deepstream=True)


        # Giữ mỗi mô hình phát hiện đối tượng trong thư mục riêng: mọi config

        # phát hiện đều đặt cùng một tên tệp cache engine. Xem "Các bẫy đã biết"

        LibreDFINE("LibreDFINEs.pt", size="s").export(format="onnx",
        deepstream=True)
    - label: Đối số
      language: python
      code: >
        model.export(
            format="onnx",     # deepstream=True bị từ chối với mọi format khác
            deepstream=True,
            conf=0.25,         # đặt giá trị cho pre-cluster-threshold (và classifier-threshold,
                               # segmentation-threshold ở những tác vụ đó)
            iou=0.45,          # đặt giá trị cho nms-iou-threshold, bỏ qua khi cluster-mode=4
            batch=1,           # đặt giá trị cho batch-size và tên tệp cache engine
            half=False,        # True đánh dấu config network-mode=2 (build fp16)
            int8=False,        # True đánh dấu config network-mode=1
            dynamic=True,      # trục batch động trong đồ thị ONNX
            imgsz=640,         # đặt giá trị cho infer-dims=3;H;W
        )


        # deepstream=True và nms=True loại trừ lẫn nhau: DeepStream chạy bước

        # suppression trong giai đoạn clustering, nên không có gì được nhúng vào
        đồ thị
    - label: Tải trọng số D-FINE trước
      language: bash
      code: |
        curl -L -o LibreDFINEs.pt \
          https://huggingface.co/LibreYOLO/LibreDFINEs/resolve/main/LibreDFINEs.pt
  gpu:
    - label: Xác nhận GPU passthrough trước mọi thứ khác
      language: bash
      code: |
        docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.04-py3 \
          nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
      expect: |
        name, driver_version, compute_cap
        NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0
  parser:
    - label: 'build_parser.sh, chạy bên trong container DeepStream'
      language: bash
      code: >
        set -e

        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo.git


        # /usr/local/cuda-12 trên image này chỉ là bản stub và bước build chết
        vì nó với

        # "fatal error: crt/host_defines.h: No such file or directory". Hãy tìm
        một

        # toolkit thực sự có header đó; trên image 8.0 thì đó là cuda-12.5

        CUDA_DIR=$(readlink -f /usr/local/cuda)

        [ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
          CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
                     while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

        # Image có sẵn libcublas.so.12 và libcublas.so.12.8.4.1 nhưng không có

        # libcublas.so không phiên bản mà -lcublas cần, nên bước link hỏng với

        # "/usr/bin/ld: cannot find -lcublas". Hãy đưa cho linker đúng các tên
        nó muốn

        mkdir -p /tmp/cudalibs

        for lib in cublas cublasLt cudart; do
          real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
          ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
        done

        export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"


        make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
        CUDA_VER="${CUDA_DIR##*/cuda-}"
    - label: Phân đoạn thực thể dùng một parser khác
      language: bash
      code: >
        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git

        make -C DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg \
          CUDA_VER="${CUDA_DIR##*/cuda-}"
  run:
    - label: deepstream_app_config.txt
      language: text
      code: >
        [application]

        enable-perf-measurement=1

        perf-measurement-interval-sec=5

        gie-kitti-output-dir=kitti


        [tiled-display]

        enable=0


        [source0]

        enable=1

        type=3

        uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

        num-sources=1

        gpu-id=0


        [streammux]

        gpu-id=0

        batch-size=1

        batched-push-timeout=40000

        width=1920

        height=1080

        live-source=0


        [primary-gie]

        enable=1

        gpu-id=0

        gie-unique-id=1

        config-file=config_infer_primary_libreyolo9s.txt


        [osd]

        enable=1

        border-width=2

        text-size=15


        [sink0]

        enable=1

        type=1

        sync=0


        [tests]

        file-loop=0
    - label: Chạy thử
      language: bash
      code: |
        deepstream-app -c deepstream_app_config.txt
      expect: |
        App run successful
    - label: Cả hai bước trong một container
      language: bash
      code: |
        docker run --rm --gpus all -v "$PWD:/work" -w /work \
          nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
          bash -c "bash build_parser.sh && deepstream-app -c deepstream_app_config.txt"
source_hash: 1ee91c265753dd9a
---

## Khả dụng

Xuất sang DeepStream có trong v1.5.0. Nó đã merge vào `dev` ngày 2026-08-08
trong pull request 728, nên một bản cài đặt hiện tại đã có sẵn và không cần ghim
vào branch nào.

<code-tabs name="install" />

Nếu bạn đã clone branch `deepstream-export` trước ngày 2026-08-08, hãy thay nó
bằng bản mới. Branch đó đã bị rebase và force-push, còn phần lịch sử cũ thì
thiếu bản sửa lỗi giúp các lần xuất này chạy được trên máy CUDA.

## Những gì lệnh xuất ghi ra

`model.export(format="onnx", deepstream=True)` ghi ra ba tệp cạnh nhau. Với
`libreyolo9s.pt`:

- `libreyolo9s.onnx`, đồ thị phát hiện đối tượng, một tensor đầu ra có shape
  `(batch, num_detections, 6)`, mỗi hàng là `[x1, y1, x2, y2, score, class_id]`
  theo tọa độ pixel của đầu vào mạng.
- `config_infer_primary_libreyolo9s.txt`, một cấu hình `nvinfer` mang theo các
  hằng số tiền xử lý của họ mô hình, số lớp đối tượng, các ngưỡng và phần nối
  tới parser.
- `libreyolo9s_labels.txt`, mỗi dòng một tên lớp đối tượng.

Tệp nhãn xuất hiện mỗi khi checkpoint mang theo tên các lớp đối tượng. Mô hình
độ sâu thì không có, nên chúng không nhận được tệp đó lẫn khóa `labelfile-path`.

LibreYOLO không sinh ra tệp `.so` nào. Tệp `.so` mà DeepStream nạp là parser
bounding box từ `marcoslucianops/DeepStream-Yolo`, được build một lần cho mỗi
thiết bị, và nó vẫn là cùng một binary dù bạn trỏ nó vào bộ phát hiện LibreYOLO
nào. Mô hình chính là tệp ONNX. Phân loại và phân đoạn ngữ nghĩa không cần
parser nào cả, vì `nvinfer` tự hậu xử lý những tác vụ đó.

## Xuất mô hình

<code-tabs name="export" />

`LibreDFINE._load_weights` ném `FileNotFoundError` khi tệp chưa có sẵn trên đĩa,
mà không thử tải về, nên hãy tự tải `LibreDFINEs.pt` trước. Khoảng trống đó đang
được theo dõi ở
[issue #727](https://github.com/LibreYOLO/libreyolo/issues/727). Trọng số YOLO9
được tải về ở lần dùng đầu tiên.

Tham số này chỉ có ở Python. `libreyolo export` trên branch này không có tùy
chọn `deepstream`, và CLI dựng các đối số xuất từ một danh sách cố định thay vì
chuyển tiếp những khóa lạ.

## Build parser bounding box

Phát hiện đối tượng cần thư viện parser, phân đoạn thực thể cần một thư viện
khác, còn các tác vụ còn lại thì không cần. Có hai thứ trên image DeepStream 8.0
làm hỏng lệnh build được ghi trong tài liệu, và cả hai đều là vấn đề môi trường
chứ không phải vấn đề của LibreYOLO.

Image có sẵn `cuda`, `cuda-12`, `cuda-12.5`, `cuda-12.8` và `cuda-12.9` dưới
`/usr/local`. Chỉ `cuda-12.5` có một toolkit đầy đủ. Nó cũng có
`libcublas.so.12` và `libcublas.so.12.8.4.1` nhưng không có `libcublas.so` không
phiên bản mà `-lcublas` dùng để resolve. Script bên dưới đi vòng qua cả hai.

<code-tabs name="parser" />

Sau đó hãy trỏ `custom-lib-path` trong config được sinh ra tới tệp
`libnvdsinfer_custom_impl_Yolo.so` vừa build. Giá trị được sinh ra là đường dẫn
tương đối `nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`, đường
dẫn này đúng khi `deepstream-app` chạy từ thư mục checkout của `DeepStream-Yolo`
và cần sửa lại trong các trường hợp còn lại.

## Chạy pipeline

Hãy kiểm tra xem container có thấy GPU không trước khi mất thời gian vào bất cứ
thứ gì khác. Đây là bước kiểm tra đầu tiên mà lần chạy kiểm chứng đã làm, trên
một card Blackwell dưới WSL2.

<code-tabs name="gpu" />

Lần chạy kiểm chứng điều khiển `deepstream-app` với một nguồn là tệp, không có
display sink, bật on-screen display, và đặt `gie-kitti-output-dir` để kết quả
phát hiện của từng frame rơi xuống đĩa dưới dạng văn bản KITTI. Một config với
những thiết lập đó:

<code-tabs name="run" />

`nvinfer` build engine TensorRT từ tệp ONNX ở lần chạy đầu tiên rồi cache nó
ngay cạnh mô hình, nên lần chạy đầu phải trả giá cho việc build engine còn những
lần sau chỉ nạp cache.

## Config được sinh ra

Cả hai config bên dưới đều do bộ xuất ghi ra cho lần chạy kiểm chứng, không hề
sửa lại sau đó.

| Khóa | YOLO9-s | D-FINE-s |
|---|---|---|
| `net-scale-factor` | 0.003921568627 | 0.003921568627 |
| `model-color-format` | 0 | 0 |
| `infer-dims` | 3;640;640 | 3;640;640 |
| `maintain-aspect-ratio` | 1 | 0 |
| `symmetric-padding` | 0 | 0 |
| `network-type` | 0 | 0 |
| `num-detected-classes` | 80 | 80 |
| `cluster-mode` | 2 | 4 |
| `parse-bbox-func-name` | NvDsInferParseYolo | NvDsInferParseYolo |
| `pre-cluster-threshold` | 0.25 | 0.25 |
| `nms-iou-threshold` | 0.45 | |
| `topk` | 300 | 300 |

Hai config khác nhau ở ba chỗ: `maintain-aspect-ratio`, `cluster-mode`, và việc
`nms-iou-threshold` có xuất hiện hay không. Config của D-FINE bỏ hẳn khóa đó,
đúng như `cluster-mode=4` đòi hỏi.

Những head sinh ra nhiều nhất một dự đoán cho mỗi vật thể sẽ nhận
`cluster-mode=4`, nên DeepStream không chạy clustering trên chúng; clustering sẽ
gộp nhầm những kết quả phát hiện thực sự khác nhau. Nhóm đó gồm `rfdetr`,
`dfine`, `deim`, `deimv2`, `ec`, `rtdetr`, `rtdetrv2`, `rtdetrv4` và
`yolo9_e2e`. Các head dạng grid và anchor nhận `cluster-mode=2` cùng
`nms-iou-threshold`.

Config phát hiện đối tượng còn mang theo
`engine-create-func-name=NvDsInferYoloCudaEngineGet`, khóa này giao việc build
engine cho thư viện parser. Đó chính là thứ cố định tên tệp cache engine, và
cũng là nguồn gốc của xung đột được mô tả ở phần các bẫy đã biết.

## Tác vụ và họ mô hình được hỗ trợ

Bốn mươi ba tổ hợp họ mô hình và tác vụ xuất được. `deepstream_supported_tasks()`
và `deepstream_supported_families(task)` trong `libreyolo/export/deepstream.py`
trả về đúng những danh sách đó lúc runtime.

| Tác vụ | `network-type` | Thư viện parser | Họ mô hình |
|---|---|---|---|
| Phát hiện đối tượng | 0 | DeepStream-Yolo | yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox, yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr, rtdetrv2, rtdetrv4 |
| Phân loại | 1 | Không cần | mobilenetv4, convnext, efficientnetv2, resnet, dinov2 |
| Phân đoạn ngữ nghĩa | 2 | Không cần | pidnet, eomt, dinov2, lingbotvision |
| Phân đoạn thực thể | 3 | DeepStream-Yolo-Seg | rfdetr, dfine, ec |
| Tư thế | 100 | Không cần | yolo9, yolonas, rfdetr, ec |
| Độ sâu | 100 | Không cần | depth_anything, zipdepth |
| Phục hồi ảnh | 100 | Không cần | nafnet, realesrgan, swinir |
| Matting | 100 | Không cần | birefnet |
| Gaze | 100 | Không cần | l2cs |

`network-type=100` nghĩa là DeepStream không có bộ hậu xử lý cho tác vụ đó.
Những config này đặt `output-tensor-meta=1`, các đầu ra gốc của đồ thị đi qua
nguyên vẹn, và ứng dụng tự giải mã chúng từ metadata của tensor. Đồ thị nhiều
đầu ra vẫn ổn ở đó: mọi lớp đầu ra đều tới được metadata với đúng tên đầu ra và
các trục động như một lần xuất ONNX thông thường.

Mỗi hàng của phân đoạn thực thể là hàng phát hiện đối tượng, theo sau là mặt nạ
(mask) của thực thể đó, được duỗi phẳng ở `(netH / 4, netW / 4)`, độ phân giải
mà parser phân đoạn hardcode sẵn, dưới dạng xác suất cho
`segmentation-threshold`.

Phân loại và gaze chạy như inference thứ cấp. Đặt `process-mode=2` và
`operate-on-gie-id` trong config được sinh ra để đưa một bộ phân loại ra sau một
bộ phát hiện. Gaze là hợp đồng chỉ có phần head, mỗi đầu vào là một ảnh cắt
khuôn mặt, nên nó cần một bộ phát hiện khuôn mặt đứng phía trước.

Ba họ mô hình vắng mặt là có chủ đích. `segformer` chưa được nối vào hợp đồng
xuất ngữ nghĩa dùng chung và không xuất được sang ONNX ở bất kỳ dạng nào.
RTMDet-Ins và YOLO9 bị chặn phần xuất phân đoạn thực thể ngay trong chính
LibreYOLO. `depth_anything3` chưa có phần hiện thực xuất.

Hai hàng trong bảng có khoảng trống về checkpoint phía sau. Chỉ có checkpoint
ngữ nghĩa EoMT cỡ `l` được công bố, còn phân loại DINOv2 thì chưa có checkpoint
công bố nào, nên tổ hợp đó cần trọng số do bạn tự tinh chỉnh (fine-tuning).

## Khác biệt ở khâu tiền xử lý

`nvinfer` tính `net-scale-factor * (x - offsets)` trên từng kênh với một hệ số vô
hướng, thứ không diễn đạt được độ lệch chuẩn riêng cho từng kênh. Những họ mô
hình cần điều đó (`rfdetr`, `ec`, các cỡ `deimv2` dùng backbone DINO, `rtmdet`,
`picodet`, và mọi họ phân loại) được nhúng sẵn phép chuẩn hóa vào đồ thị đã
xuất, và config được sinh ra sẽ đưa vào đồ thị đúng không gian đầu vào thô tương
ứng.

Phần hình học mới là chỗ mà pipeline Python của chính LibreYOLO và `nvinfer` vẫn
còn khác nhau:

- Các họ dùng letterbox (`yolo9`, `yolox`, `yolonas`, `rtmdet`, `yolo2`,
  `yolo3`, `yolo4`, `yolo7`) vốn pad bằng màu xám. `nvinfer` pad bằng màu đen.
- `yolonas` khi phát hiện đối tượng vốn resize cạnh dài nhất về 636 trong khung
  640 của nó. `maintain-aspect-ratio` của `nvinfer` dùng trọn 640.
- Phân loại vốn resize cạnh ngắn nhất rồi cắt giữa. `nvinfer` kéo giãn khung
  hình hoặc vùng ROI của vật thể về đúng đầu vào mạng, nên những chủ thể được
  cắt sát sẽ khác đi.
- EoMT vốn chạy các ô cửa sổ trượt cho phân đoạn ngữ nghĩa. Đồ thị đã xuất là
  một khung bị kéo giãn duy nhất, nhanh hơn và kém chính xác hơn.
- `pidnet` sinh ra bản đồ lớp đối tượng ở 1/8 độ phân giải đầu vào còn
  `lingbotvision` ở 1/16. DeepStream upsample bản đồ lớp đối tượng để hiển thị.

Cổng kiểm tra parity của ONNX nạp vào các tensor đã được tiền xử lý sẵn, nên nó
kiểm tra đầu ra của đồ thị và không bắt được lỗi sai thứ tự kênh màu hay sai
chính sách padding trong config. Hãy kiểm chứng trên dữ liệu của bạn trước khi
triển khai một tải công việc đòi hỏi parity chính xác.

## Các bẫy đã biết

### Hai mô hình phát hiện đối tượng trong cùng một thư mục sẽ nạp engine của nhau

Mọi config phát hiện đối tượng đều mang cùng một dòng:

```ini
model-engine-file=model_b1_gpu0_fp32.engine
```

Bộ build engine của parser đòi đúng tên tệp đó và nó không thay đổi theo mô
hình. Xuất một mô hình phát hiện thứ hai vào cùng thư mục thì lần chạy thứ hai
sẽ nạp engine đã cache của mô hình thứ nhất. Không có gì crash; chỉ có các box
là sai. Hãy cho mỗi mô hình phát hiện một thư mục riêng. Lần chạy kiểm chứng đã
phải tách D-FINE ra một thư mục riêng thì mới thử nghiệm được.

### Một box chỉ mang được một lớp đối tượng

Định dạng hàng của `nvinfer` là `[x1, y1, x2, y2, score, class_id]`, mỗi box một
lớp đối tượng, nên lần xuất sẽ ép điểm số của các lớp đối tượng về argmax của
chúng. Một box mà `predict` báo dưới hai lớp đối tượng thì chỉ sống sót dưới
một lớp. Trường hợp đo được: LibreYOLO báo `vase 0.773` và `bottle 0.383` trên
cùng một box, còn đồ thị DeepStream giữ lại `vase`. Điều này là hệ quả của định
dạng hàng mà parser dùng và không đổi được nếu không rời khỏi hợp đồng đó, nên
đây là hành vi được trông đợi chứ không phải một regression.

## Đã kiểm chứng

`deepstream-app` chạy tới EOS với `App run successful` trên cả hai kiểu head của
bộ phát hiện, trên tệp `sample_1080p_h264.mp4` (1443 frame) mà NVIDIA đóng gói
kèm, có bật ghi KITTI cho từng frame.

| | YOLO9-s | D-FINE-s |
|---|---|---|
| Kiểu head | grid | one-to-one |
| `cluster-mode` | 2 | 4 |
| `maintain-aspect-ratio` | 1 | 0 |
| Số frame có kết quả phát hiện | 1443 | 1443 |
| Tổng số kết quả phát hiện | 18031 | 71105 |

Biểu đồ tần suất lớp đối tượng trên toàn bộ 1443 frame xếp xe hơi ở vị trí đầu
và người ở vị trí thứ hai với cả hai mô hình, đúng như một cảnh đường phố. Chênh
lệch gấp bốn lần về số lượng kết quả phát hiện chính là khác biệt `cluster-mode`
đang làm đúng việc của nó: D-FINE ở `cluster-mode=4` không chạy clustering, nên
mọi query vượt ngưỡng đều sống sót, kể cả những kết quả gần trùng nhau.

Hai mô hình được huấn luyện độc lập đặt vật thể chiếm ưu thế vào cùng một chỗ:

```text
YOLO9  bus  [706.72,  0.82, 1916.34, 1062.97]  conf 0.965
D-FINE bus  [702.73,  2.93, 1916.24, 1069.32]  conf 0.965
```

Lần chạy đó khẳng định năm điều: TensorRT build được engine từ tệp ONNX đã xuất
trên sm_120, `nvinfer` chấp nhận mọi khóa trong config được sinh ra,
`NvDsInferParseYolo` đọc đúng bố cục tensor, các box rơi vào hệ tọa độ 1920x1080
theo độ phân giải nguồn, và nhãn được resolve đúng theo tệp nhãn được sinh ra.

Môi trường mà nó đã chạy:

| Thành phần | Giá trị |
|---|---|
| Hệ điều hành máy chủ | Windows 11 Pro 26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| Driver | 591.86 |
| Compute capability | 12.0 (Blackwell, sm_120) |
| Runtime container | Docker Desktop 29.4.3, backend WSL2 |
| Image DeepStream | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` |
| Phiên bản DeepStream | 8.0.0 |
| CUDA trong container | 12.8.1 |
| Parser | `marcoslucianops/DeepStream-Yolo` tại HEAD |

Song song với lần chạy pipeline, `tests/unit/test_deepstream_export.py` phủ các
adapter của đồ thị và các khóa config được sinh ra, và 35 test của nó đều pass
trên commit này.

## Chưa được kiểm chứng

Nêu ra để phạm vi ở trên không bị hiểu rộng hơn thực tế.

- Jetson và aarch64. Hợp đồng xuất không phụ thuộc vào kiến trúc, nhưng pipeline
  mới chỉ được chạy trên một GPU rời x86.
- Bốn mươi mốt trong số 43 tổ hợp. Chỉ có phát hiện đối tượng với `yolo9` và
  phát hiện đối tượng với `dfine` là đã đi qua DeepStream. Phân loại, phân đoạn
  ngữ nghĩa, phân đoạn thực thể và các tác vụ trả tensor thô được phủ bởi unit
  test và các bài kiểm tra parity ONNX, chứ không phải bởi một lần chạy pipeline.
- FP16 và INT8. Chỉ `network-mode=0` được thử.
- Đa luồng (multi-stream) và batching. Một nguồn, `batch-size=1`.
- Độ chính xác so với một tập dữ liệu (dataset) có ground truth. Các kết quả
  phát hiện chỉ được kiểm tra về mức hợp lý ngữ nghĩa và mức đồng thuận giữa hai
  mô hình, chứ không được chấm mAP qua DeepStream.
