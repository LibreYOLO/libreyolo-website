---
title: Triton Inference Server
seo_title: Phục vụ mô hình LibreYOLO trên NVIDIA Triton
description: >-
  Phục vụ bản xuất ONNX của LibreYOLO qua NVIDIA Triton: bố cục kho mô hình
  (model repository), tệp config.pbtxt được sinh ra, và cách dự đoán với một
  model URL HTTP.
lead: >-
  Triton Inference Server lưu trữ một model repository và trả lời các yêu cầu
  suy luận (inference) qua HTTP. LibreYOLO xuất đồ thị ONNX, sinh một
  config.pbtxt mang toàn bộ metadata của lần xuất dưới dạng một parameter
  Triton, và coi một model URL như một đường dẫn mô hình có thể tải được.
keywords:
  - libreyolo triton
  - triton inference server
  - config.pbtxt
  - tritonclient http
  - model repository triton
  - chạy yolo từ xa qua http
last_verified: 1.5.0
meta:
  - label: Lệnh gọi
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: Hàm hỗ trợ
    value: >-
      create_triton_config(onnx_path, config_path, model_name=...,
      max_batch_size=8)
    mono: true
  - label: Extra
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: Giao thức
    value: >-
      Chỉ inference V2 qua HTTP và HTTPS. Không có gRPC, xác thực, shared
      memory, hay load và unload mô hình.
  - label: Thời gian chờ
    value: Timeout kết nối và timeout mạng mặc định là 30 giây
verification: >-
  Đọc từ libreyolo/backends/triton.py, libreyolo/models/__init__.py,
  docs/triton.md và pyproject.toml trên nhánh dev. Các lệnh container là những
  lệnh đã ghim phiên bản lấy từ docs/triton.md.
snippets:
  install:
    - label: Cài đặt
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: Xuất theo bố cục thư mục của repository
      language: python
      code: |
        from pathlib import Path

        from libreyolo import LibreYOLO

        model_dir = Path("triton_repo/yolo9/1")
        model_dir.mkdir(parents=True, exist_ok=True)

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            output_path=str(model_dir / "model.onnx"),
            dynamic=True,
            simplify=False,
        )
    - label: Sinh config.pbtxt
      language: python
      code: |
        from libreyolo import create_triton_config

        create_triton_config(
            "triton_repo/yolo9/1/model.onnx",
            "triton_repo/yolo9/config.pbtxt",
            model_name="yolo9",
            max_batch_size=8,
        )
    - label: Bố cục thu được
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: Khởi động server
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: Chờ server sẵn sàng
      language: bash
      code: >
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do
        sleep 1; done
    - label: Dừng server
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: Dự đoán với mô hình đang được phục vụ
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
        result = remote.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: So sánh với mô hình cục bộ
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9").predict(SAMPLE_IMAGE)
        native = LibreYOLO("LibreYOLO9t.pt").predict(SAMPLE_IMAGE)

        print(len(remote.boxes), len(native.boxes))
        print(remote.boxes.xyxy[:3])
        print(native.boxes.xyxy[:3])
    - label: 'Ghim một phiên bản, hoặc đổi timeout'
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # Đoạn path thứ hai chọn phiên bản mô hình. Nếu không có nó,
        # version policy đã cấu hình của Triton sẽ quyết định
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # Timeout kết nối và timeout mạng mặc định là 30 giây
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
source_hash: 0652e4faf0224df3
---

## Cài đặt

<code-tabs name="install" />

Extra `triton` cài `tritonclient[http]`. Các extra cho gRPC và shared memory bị
loại bỏ có chủ đích: tích hợp này chỉ có inference V2 qua HTTP và HTTPS. Cần
`onnx` vì cả artifact được phục vụ lẫn bộ sinh config đều làm việc từ một đồ thị
ONNX.

## Xây dựng model repository

Xuất với trục batch động, theo đúng bố cục thư mục mà Triton mong đợi.

<code-tabs name="repo" />

Triton không giữ lại metadata tùy chỉnh của ONNX trong phản hồi model-config, nên
toàn bộ metadata đã xuất phải đi theo một đường khác. `create_triton_config` mã hóa
nó thành một parameter kiểu chuỗi JSON tên `libreyolo_metadata` trong
`config.pbtxt`, phát ra các khai báo đầu vào và đầu ra theo thứ tự trong đồ thị, xử
lý phần escape JSON, và ghim mô hình vào `KIND_CPU`.

Hàm hỗ trợ kiểm tra hợp lệ trước khi ghi. Nó đòi hỏi đúng một đầu vào cho đồ thị
ONNX, ít nhất một đầu ra, các shape tensor phân giải được, và metadata có map
`names` định nghĩa mọi chỉ số lớp đối tượng từ 0 đến `nc - 1`. Mô hình không qua
được bất kỳ kiểm tra nào trong số đó sẽ bị từ chối ngay lúc sinh config, chứ không
phải ở request đầu tiên.

`max_batch_size: 8` khớp với một lần xuất động và cho phép server gom tới tám ảnh
mỗi request. Với đồ thị ONNX có batch cố định bằng 1, hãy dùng `max_batch_size=0`;
khi đó LibreYOLO gửi ảnh tuần tự.

## Khởi động server

<code-tabs name="serve" />

Các lệnh này ghim Triton Server 26.04 và cố ý bỏ qua các cờ GPU của Docker, vì dù
sao `KIND_CPU` trong config được sinh ra cũng đã chặn việc đặt mô hình lên GPU.

## Chạy artifact

Một model URL của Triton chính là một đường dẫn mô hình. `LibreYOLO()` kiểm tra
scheme `http` hoặc `https` trước mọi xử lý đường dẫn cục bộ và trả về một backend
nói chuyện với server, nên chỗ gọi hàm giống hệt như với một checkpoint cục bộ, và
đối tượng `Results` trả về cũng vậy.

<code-tabs name="run" />

Dạng URL là `http(s)://host:port/model` với một đoạn phiên bản tùy chọn. Port phải
được ghi rõ. Thông tin đăng nhập nhúng trong URL, query string và fragment đều bị
từ chối, đường dẫn có hơn hai đoạn cũng vậy.

`device` được chấp nhận rồi bỏ qua kèm một dòng log, vì việc đặt mô hình ở đâu là
quyết định của server.

## Giới hạn

Backend báo lỗi thẳng thay vì trả về kết quả kém chất lượng khi hợp đồng không được
đáp ứng: thiếu metadata LibreYOLO trong model config, mô hình có nhiều hơn một đầu
vào, các đầu ra được cấu hình không khớp với metadata của mô hình, kiểu dữ liệu đầu
vào mà nó không hỗ trợ, hoặc server hay mô hình chưa sẵn sàng.

Nằm ngoài hợp đồng ở phiên bản này: gRPC, xác thực, shared memory, và việc load hay
unload mô hình qua API.

Bất kỳ format nào bản thân Triton hỗ trợ đều có thể được phục vụ, nhưng ở đây
parameter metadata và config được sinh ra đều mang hình dạng ONNX, nên con đường
của LibreYOLO là [ONNX](/docs/export/onnx) đưa vào repository. Nếu cần một pipeline
video đầy đủ thay vì một server kiểu request-response, xem
[DeepStream](/docs/export/deepstream).
