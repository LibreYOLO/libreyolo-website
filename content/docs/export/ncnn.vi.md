---
title: ncnn
seo_title: Xuất sang ncnn từ LibreYOLO
description: >-
  Xuất một mô hình LibreYOLO sang ncnn qua PNNX: cặp tệp param và bin, khung ảnh
  xuất cố định, phần viết lại Focus của YOLOX, và những họ mô hình chuyển đổi
  được.
lead: >-
  ncnn là thư viện suy luận (inference) trên CPU của Tencent, hướng tới các đích
  di động. LibreYOLO chuyển đổi qua PNNX, ghi ra graph model.ncnn.param bên cạnh
  tệp trọng số model.ncnn.bin và một metadata.yaml mang theo họ mô hình, tác vụ
  và tên các lớp đối tượng.
keywords:
  - xuất yolo sang ncnn
  - pnnx
  - model.ncnn.param
  - chạy yolo trên cpu di động
  - ncnn extractor
  - focus pixel_unshuffle
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="ncnn")
    mono: true
  - label: Kết quả ghi ra
    value: 'Một thư mục chứa model.ncnn.param, model.ncnn.bin và metadata.yaml'
  - label: Gói bổ sung
    value: 'pip install "libreyolo[ncnn]"'
    mono: true
  - label: Tải lại bằng
    value: LibreYOLO("weights/LibreYOLO9t_ncnn")
    mono: true
  - label: Hình dạng
    value: Cố định. Metadata ghi dynamic=False bất kể cờ được đặt thế nào.
  - label: Độ chính xác
    value: Chỉ FP32. half=True và int8=True bị từ chối.
verification: >-
  Đọc từ libreyolo/export/ncnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/ncnn.py và pyproject.toml trên
  nhánh dev.
snippets:
  install:
    - label: Cài đặt
      language: bash
      code: |
        # pnnx chuyển đổi, ncnn chạy kết quả
        pip install "libreyolo[ncnn]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Ghi ra thư mục weights/LibreYOLO9t_ncnn
        path = model.export(format="ncnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640
    - label: Tham số
      language: python
      code: |
        model.export(
            format="ncnn",
            imgsz=640,        # int, hoặc (chiều cao, chiều rộng)
            batch=1,
            simplify=True,    # chỉ áp dụng cho đường ONNX dự phòng
            opset=None,       # tự động; chỉ áp dụng cho đường ONNX dự phòng
            output_path=None, # None sẽ ghi ra weights/<stem>_ncnn
        )

        # half=True và int8=True bị từ chối trong quá trình kiểm tra
  run:
    - label: Qua LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_ncnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ncnn thuần
      language: python
      code: |
        import ncnn
        import numpy as np
        import yaml

        directory = "weights/LibreYOLO9t_ncnn"
        net = ncnn.Net()
        net.load_param(f"{directory}/model.ncnn.param")
        net.load_model(f"{directory}/model.ncnn.bin")

        # ncnn nhận một ảnh CHW đơn lẻ, không phải một batch
        mat_in = ncnn.Mat(np.zeros((3, 640, 640), dtype=np.float32))
        extractor = net.create_extractor()
        extractor.input("in0", mat_in)
        ret, mat_out = extractor.extract("out0")
        print(ret, np.array(mat_out).shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # Tiền xử lý và hậu xử lý là phần việc của bạn trên đường này
  support:
    - label: Kiểm tra một họ mô hình và tác vụ trước khi xuất
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 9a849a16a3b32334
---

## Cài đặt

<code-tabs name="install" />

Gói bổ sung kéo về cả hai nửa của toolchain: `pnnx` thực hiện việc chuyển đổi còn
`ncnn` chạy kết quả. Trên đường đi chính, không bên nào đi qua ONNX.

## Xuất mô hình

<code-tabs name="export" />

Kết quả xuất ra là một thư mục. `weights/LibreYOLO9t_ncnn` chứa
`model.ncnn.param`, `model.ncnn.bin` và `metadata.yaml`; cả ba là một khối duy
nhất và luôn đi cùng nhau.

Quá trình chuyển đổi thử PNNX trực tiếp từ PyTorch trước. Nếu cách đó thất bại,
nó xuất một graph ONNX tĩnh ra một thư mục tạm rồi gọi công cụ dòng lệnh `pnnx`
lên graph đó, và việc xuất chỉ báo lỗi khi cả hai đường đều thất bại, kèm theo cả
hai lỗi. Vì vậy `opset` và `simplify` chỉ ảnh hưởng tới đường dự phòng.

YOLOX cần một lần viết lại thì mới chuyển đổi được. Lớp Focus của nó dùng phép
cắt lát có bước nhảy (strided slicing) mà PNNX không hạ cấp được, nên khi xuất,
lớp đó được thay bằng `pixel_unshuffle` và các kênh đầu vào của convolution kế
tiếp được hoán vị để bù cho thứ tự kênh khác đi. Đầu ra giống hệt về mặt số học,
và trọng số gốc được khôi phục sau khi xuất xong.

## Chạy kết quả xuất ra

<code-tabs name="run" />

`LibreYOLO()` nhận ra mọi thư mục có chứa `model.ncnn.param` và
`model.ncnn.bin`, đọc `metadata.yaml`, và trả về cùng đối tượng `Results` như khi
chạy từ checkpoint.

Đoạn mã thứ hai là đường chạy thẳng trên runtime, và có hai chi tiết khác với mọi
định dạng khác ở đây. ncnn làm việc trên một ảnh CHW đơn lẻ chứ không phải một
batch, nên không có trục batch ở đầu. Tên các blob lấy từ tệp `.param`; PNNX ghi
ra `in0` và `out0` theo quy ước, còn backend thì đọc tệp thay vì giả định sẵn hai
tên đó. Tiền xử lý, giải mã, NMS và việc quy đổi lại tọa độ là phần việc của bạn
trên đường đó.

## Ràng buộc

FP32 trên một khung ảnh cố định. `half=True` và `int8=True` đều bị từ chối trong
quá trình kiểm tra, và metadata xuất ra ghi `dynamic=False` bất kể cờ được đặt
thế nào, nên không backend nào giả định một trục mà graph không có.

Mọi họ mô hình kiểu DETR đều bị từ chối ngay ở bước kiểm tra trước khi xuất:
`detr`, `deformable_detr`, `dinodetr`, `dfine`, `lwdetr`, `deim`, `deimv2`,
`rtdetr`, `rtdetrv2`, `rtdetrv4`, `rfdetr` và `ec`. Thông báo cho tất cả đều
giống nhau, rằng mô hình cần các phép decoder hoặc sampling mà ncnn không có, và
nó chỉ sang ONNX, OpenVINO, TorchScript hoặc TensorRT.

Phần chuyển đổi được thì rất rộng ở phía convolutional: YOLO9 và YOLO9-E2E,
YOLOX, PicoDet, YOLO-NAS bản detection và pose, các bộ phát hiện cũ hơn YOLO1,
YOLO3, YOLO4 và YOLO7, bốn họ phân loại CNN, PIDNet cho phân đoạn ngữ nghĩa, FOMO
phát hiện điểm ở kích thước cố định 96 nhân 96, ZipDepth, NAFNet và Real-ESRGAN.

Các mục bị chặn đều nêu rõ lỗi cụ thể. Graph transformer thường để lại các node
`pnnx.Expression` không được hỗ trợ, tạo ra một mạng không có blob đầu vào chạy
được, và đó là thứ chặn DINOv2, CLIP, SigLIP2 và SegFormer. BiRefNet cần
deformable convolution của torchvision, thứ mà PNNX không hạ cấp được. Graph đã
chuyển đổi của YOLO2 làm runtime ncnn trên Windows chết hẳn vì lỗi chia số nguyên
cho 0 ở tầng native khi trích xuất đầu ra.

Về bảng đầy đủ theo họ mô hình và tác vụ, xem
[ma trận xuất mô hình](/docs/reference/export-matrix). Với một tổ hợp cụ thể:

<code-tabs name="support" />
