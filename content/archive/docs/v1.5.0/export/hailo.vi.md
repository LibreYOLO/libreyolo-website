---
title: Hailo
seo_title: Chạy mô hình LibreYOLO trên bộ tăng tốc Hailo
description: >-
  Triển khai một mô hình LibreYOLO lên Hailo-8 hoặc Hailo-8L: bản xuất ONNX
  tĩnh, bước Dataflow Compiler bạn tự chạy, và những kiến trúc nào biên dịch
  được.
lead: >-
  Bộ tăng tốc Hailo được biên dịch bằng Hailo Dataflow Compiler, một SDK độc
  quyền phân phối qua Developer Zone của Hailo. Phần việc của LibreYOLO trong
  luồng này chỉ là một bản xuất ONNX tĩnh thuần túy; việc parse, lượng tử hóa
  (quantization) và biên dịch thành HEF diễn ra sau đó bên trong DFC.
keywords:
  - libreyolo hailo
  - chạy yolo trên hailo-8
  - hailo-8l
  - raspberry pi ai kit
  - ai hat+
  - hailo dataflow compiler
  - biên dịch hef
  - hailortcli
last_verified: 1.5.0
meta:
  - label: Bước của LibreYOLO
    value: 'export(format="onnx", imgsz=640, dynamic=False)'
    mono: true
  - label: Không phải một định dạng
    value: Không có format="hef". DFC không thể là một dependency của pip.
  - label: Gói bổ sung
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Máy biên dịch
    value: >-
      Linux x86_64, kể cả WSL2 Ubuntu 22.04. Việc biên dịch không thể chạy trên
      ARM.
  - label: Biên dịch được
    value: >-
      Graph thuần CNN, hình dạng cố định. Attention, hình dạng động và các thiết
      kế thiên về LayerNorm thì không.
  - label: Trạng thái
    value: >-
      Chưa có họ mô hình LibreYOLO nào được đưa trọn vẹn qua DFC tới một HEF
      chạy được.
verification: >-
  Đọc từ skills/libreyolo-export-hailo/SKILL.md, libreyolo/export/onnx.py và
  libreyolo/cli/commands/export.py trên nhánh dev. Các ràng buộc của DFC là
  những ràng buộc được ghi lại trong skill đó; chưa có HEF LibreYOLO nào được
  biên dịch và đo đạc.
snippets:
  install:
    - label: Phía LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'Phía Hailo, bạn tự cài'
      language: text
      code: >
        Prerequisites, none of them installable from PyPI:


        - A Linux x86_64 machine. WSL2 Ubuntu 22.04 works. The Raspberry Pi is a
          runtime target, never the compile host.
        - The Dataflow Compiler wheel (hailo_sdk_client) from the Hailo
        Developer
          Zone, which is free to register for.
        - For Hailo-8 and Hailo-8L, the Hailo Model Zoo v2.x line, for its
          recipes and NMS configurations.
        - A GPU on the compile host is strongly recommended: the quantization
          step takes hours without one.
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Hailo cần batch 1, một độ phân giải cố định và không có trục động
        # API Python mặc định dùng dynamic=True, nên phải tắt nó tường minh
        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    - label: CLI
      language: bash
      code: |
        # CLI vốn đã mặc định dùng hình dạng tĩnh
        libreyolo export --model LibreYOLOXs.pt --format onnx --imgsz 640
    - label: Xác nhận graph là tĩnh trước khi biên dịch
      language: python
      code: |
        import onnx

        graph = onnx.load("weights/LibreYOLOXs.onnx").graph
        shape = graph.input[0].type.tensor_type.shape
        print([d.dim_value or d.dim_param for d in shape.dim])
  compile:
    - label: 'Parse, lượng tử hóa và biên dịch'
      language: python
      code: >
        from pathlib import Path


        import numpy as np

        from hailo_sdk_client import ClientRunner

        from PIL import Image


        ONNX = "weights/LibreYOLOXs.onnx"

        HW_ARCH = "hailo8"     # hailo8 | hailo8l | hailo10h

        IMGSZ = 640


        runner = ClientRunner(hw_arch=HW_ARCH)


        # Với YOLOX, chạy translate một lần mà không có end_node_names: log của
        DFC

        # sẽ in ra các end node mà nó đề xuất. Chạy lại với những tên đó

        runner.translate_onnx_model(ONNX)


        # Normalization phải khớp với tiền xử lý của LibreYOLO. YOLOX và YOLO9

        # không cần mean hay độ lệch chuẩn, chỉ cần tỷ lệ 0-255 sang 0-1

        script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0,
        255.0])\n"


        # Tùy chọn: để Hailo đảm nhiệm NMS. Cấu hình phụ thuộc vào cả số lớp đối

        # tượng lẫn kích thước đầu vào, nên một cấu hình COCO-80 là sai với mô
        hình

        # ba lớp đối tượng đã tinh chỉnh. Không có dòng này thì HEF xuất ra
        tensor

        # head thô và ứng dụng tự decode

        # script += 'nms_postprocess("yolox_nms_config.json", meta_arch=yolox,
        engine=cpu)\n'


        runner.load_model_script(script)


        # Ảnh hiệu chuẩn phải đại diện cho dữ liệu lúc triển khai.

        # Ảnh ngẫu nhiên vẫn biên dịch được và âm thầm phá hủy độ chính xác

        calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]

        calib = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)),
                dtype=np.float32,
            )
            for p in calib_paths
        ])


        runner.optimize(calib)

        Path("libreyoloxs.hef").write_bytes(runner.compile())
    - label: Các end node của YOLO9
      language: python
      code: >
        # Graph của LibreYOLO dùng tiền tố "/head/...", không phải tiền tố
        "model.N"

        # thấy trong các cấu hình viết cho những bản xuất khác. Một cấu hình sao
        chép

        # sẽ không khớp. Hãy xác nhận tên trong graph của chính bạn nếu parse
        thất bại

        END_NODES = [
            "/head/cv2.0/cv2.0.2/Conv", "/head/cv3.0/cv3.0.2/Conv",
            "/head/cv2.1/cv2.1.2/Conv", "/head/cv3.1/cv3.1.2/Conv",
            "/head/cv2.2/cv2.2.2/Conv", "/head/cv3.2/cv3.2.2/Conv",
        ]

        runner.translate_onnx_model(ONNX, end_node_names=END_NODES)
  device:
    - label: Raspberry Pi 5 với AI Kit hoặc AI HAT+
      language: bash
      code: >
        sudo apt install dkms hailo-all

        hailortcli fw-control identify       # kiểm tra thiết bị, và nó cho biết
        tên arch

        hailortcli run libreyoloxs.hef       # smoke test và throughput
source_hash: 33b077f1c23d5535
---

## Cài đặt

LibreYOLO không có `format="hef"` và sẽ không bao giờ có. Hailo Dataflow Compiler
là một SDK độc quyền, được phân phối dưới dạng wheel riêng tư sau khi đăng ký
Developer Zone, nên nó không thể là một dependency hay một gói bổ sung. Việc triển
khai gồm hai giai đoạn: LibreYOLO ghi ra một tệp ONNX tĩnh, còn bạn chạy DFC trên
tệp đó.

```text
Libre<Model>.pt  ->  ONNX  ->  HAR (parse)  ->  HAR (quantize INT8)  ->  HEF
                 [libreyolo]           [Hailo DFC, installed by you]
```

<code-tabs name="install" />

## Xuất mô hình

<code-tabs name="export" />

Đừng truyền `half=True`. DFC nhận ONNX ở FP32 và tự thực hiện quantization INT8
của riêng nó. Cũng đừng truyền `nms=True`: NMS hoặc do Hailo đảm nhiệm qua
`nms_postprocess`, hoặc do ứng dụng đảm nhiệm, và một subgraph NMS nằm sau các end
node chỉ là gánh nặng thừa. Opset mặc định dùng được; nếu parser của DFC phản đối,
hãy xuất lại với `opset=11`.

DFC cắt graph tại các end node bạn cung cấp, tức là các lớp convolution của
detection head, rồi bỏ đi mọi thứ nằm sau đó. Vì vậy bản ONNX đã decode thông
thường của LibreYOLO vẫn là đầu vào chấp nhận được: phần đuôi decode đơn giản là bị
parser bỏ qua.

## Biên dịch

<code-tabs name="compile" />

Chọn `hw_arch` theo thiết bị đích: `hailo8` cho Hailo-8, bản AI HAT+ 26 TOPS cùng
các module M.2 và PCIe; `hailo8l` cho Hailo-8L, Raspberry Pi AI Kit và bản AI HAT+
13 TOPS; `hailo10h` cho Hailo-10H, vốn cần một bản DFC và Model Zoo mới hơn tương
ứng. Khi không chắc, `hailortcli fw-control identify` chạy trên thiết bị sẽ trả lời
câu hỏi đó.

Có hai họ mô hình ánh xạ được vào một meta-architecture NMS của HailoRT, nên Hailo
có thể đảm nhiệm phần suppression ngay bên trong pipeline đã biên dịch: YOLOX qua
`meta_arch=yolox`, và YOLO9 qua meta-architecture decoupled-head của Hailo, vốn có
bố cục head giống hệt. Hãy lấy cấu hình `nms_postprocess` tương ứng từ Hailo Model
Zoo rồi điều chỉnh nó theo số lớp đối tượng và kích thước đầu vào của bạn. Mọi
detector convolution khác đều biên dịch thành một graph không có
meta-architecture tương ứng: HEF xuất ra các tensor head thô và ứng dụng chạy
decode cùng NMS trên CPU.

Hãy giữ lại log biên dịch khi có gì đó thất bại. Mọi cách khắc phục đều xoay quanh
tên chính xác của lớp hoặc operator bị lỗi.

## Chạy tệp kết quả

<code-tabs name="device" />

Suy luận (inference) ở phía ứng dụng dùng API Python `hailo_platform`. Khi
`nms_postprocess` đã được biên dịch vào bên trong, đầu ra là
`(batch, num_classes, max_dets, 5)` mang `[y1, x1, y2, x2, score]` theo tọa độ của
mô hình, và bạn phải tự quy đổi ngược về ảnh nguồn. Pipeline `Results` của LibreYOLO
không tham gia lúc chạy; HEF là một artifact độc lập, còn tiền xử lý và hậu xử lý
là việc của ứng dụng.

## Ràng buộc

Việc một mô hình có nhắm được tới Hailo-8 hay Hailo-8L hay không là thuộc tính của
kiến trúc, không phải của cái tên, nên quy tắc dưới đây áp dụng cho cả những họ mô
hình được thêm vào sau khi trang này được viết.

Một mô hình sẽ không biên dịch được nếu nó chứa bất kỳ thứ nào sau đây:

- Attention thuộc bất kỳ dạng nào: self, cross, deformable hay windowed. Điều đó
  loại bỏ mọi detector kiểu DETR, mọi detector phát hiện với từ vựng mở
  (open-vocabulary) hoặc điều kiện theo văn bản, mọi backbone ViT, và mọi tower
  ngôn ngữ hay ngôn ngữ-thị giác. Zoo của chính Hailo có vài HEF transformer được
  chỉnh tay; đó là công việc riêng của nhà cung cấp, và không phải bằng chứng rằng
  một graph attention bất kỳ sẽ biên dịch được.
- Hình dạng động hoặc luồng điều khiển phụ thuộc dữ liệu. DFC biên dịch một hình
  dạng đầu vào cố định và một graph tĩnh, nên số lượng query thay đổi, prompt văn
  bản, top-k động, `NonZero`, `Gather` hoặc `TopK` với chỉ số động, và
  `grid_sample` đều bị loại.
- Thiết kế thiên về LayerNorm hoặc thiên về GELU. BatchNorm gộp gọn gàng vào các
  lớp convolution; hỗ trợ cho LayerNorm thì kém và GELU không phải hàm kích hoạt
  native, nên một stack kiểu ConvNeXt là lựa chọn không phù hợp dù trên danh nghĩa
  nó là mạng convolution.
- Các tác vụ ảnh sang ảnh ở độ phân giải gốc. Các mô hình phục hồi ảnh chạy ở toàn
  bộ độ phân giải đầu vào và vượt quá ngân sách SRAM thực tế của Hailo.

Một họ mô hình là ứng viên khi nó chỉ dùng convolution, dùng BatchNorm với ReLU
hoặc SiLU, và có kích thước đầu vào cố định. Trong thư viện này, điều đó nghĩa là
các detector một giai đoạn dạng CNN, với YOLOX và YOLO9 là mục tiêu chính; các
detector convolution khác như PicoDet, YOLO-NAS và RTMDet, kèm phần decode phía
ứng dụng; các mô hình phân loại CNN gồm ResNet, MobileNetV4-conv và EfficientNetV2,
trong đó ResNet được hỗ trợ tốt nhất vì Model Zoo của Hailo có sẵn công thức cho
nó; và các head tác vụ convolution nhỏ như phát hiện điểm FOMO và ước lượng hướng
nhìn L2CS trên backbone ResNet, về nguyên tắc là biên dịch được nhưng không có công
thức nào của Hailo.

Một lưu ý về trạng thái, cũng là lý do không có gì trên trang này được trình bày
như đã được hỗ trợ: chưa có họ mô hình LibreYOLO nào được đưa trọn vẹn qua DFC tới
một HEF chạy được. Các quy tắc ở trên dự đoán khả năng biên dịch từ kiến trúc. Hành
vi của parser, quantization và độ chính xác vẫn chưa được kiểm chứng cho tới khi
một HEF được biên dịch và đo đạc, nên hãy coi mọi ứng viên là cần có bằng chứng ghi
nhận của riêng nó: một HEF biên dịch từ đúng checkpoint đó, kèm phiên bản DFC, Model
Zoo và HailoRT được ghi lại, quá trình hiệu chuẩn có tài liệu, và một phép so sánh
độ chính xác trên thiết bị với baseline FP32 thay vì một con số throughput.

Nếu mô hình bị loại, các lựa chọn thay thế là những runtime đã có parity được ghi
nhận: [ONNX](/docs/export/onnx), [TensorRT](/docs/export/tensorrt) và
[OpenVINO](/docs/export/openvino).
