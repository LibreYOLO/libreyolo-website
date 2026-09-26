---
title: Huấn luyện trên GPU thuê
seo_title: Huấn luyện LibreYOLO trên GPU đám mây thuê
description: >-
  Chạy job huấn luyện LibreYOLO trên GPU thuê hoặc serverless: chuẩn bị dữ liệu,
  cài đặt, khởi chạy, theo dõi trực tiếp, lấy trọng số về và ngừng trả phí.
lead: >-
  GPU thuê biến một lượt huấn luyện thành job có thời điểm bắt đầu, kết thúc và
  hóa đơn. Công việc giống như huấn luyện cục bộ; khác biệt nằm ở cách đưa dữ
  liệu vào, theo dõi từ bên ngoài, lấy trọng số ra và tắt máy.
keywords:
  - huấn luyện gpu đám mây
  - thuê gpu
  - huấn luyện vast.ai
  - modal serverless gpu
  - beam gpu
  - huấn luyện từ xa
  - đưa dataset lên hugging face
  - chi phí gpu mỗi epoch
last_verified: 1.5.0
snippets:
  install:
    - label: Trên máy thuê
      language: bash
      code: >
        pip install libreyolo


        # Chỉ thêm những thành phần bổ sung mà lượt chạy cần. rfdetr để huấn
        luyện

        # RF-DETR, lora để tinh chỉnh ít tham số, onnx để xuất sau đó.

        pip install "libreyolo[rfdetr,lora]"
    - label: Kiểm tra GPU trước mọi thao tác khác
      language: python
      code: |
        import torch

        print(torch.__version__, torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))

        # Wheel được build cho kiến trúc khác sẽ báo True rồi thất bại ở
        # kernel thực đầu tiên, vì vậy hãy chạy thử một kernel.
        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  stage:
    - label: Đóng gói và tải lên một lần từ máy của bạn
      language: bash
      code: >
        tar cf my-dataset.tar my-dataset/

        huggingface-cli upload my-org/my-dataset my-dataset.tar --repo-type
        dataset
    - label: Chuẩn bị trên máy thuê
      language: python
      code: |
        import tarfile

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "my-org/my-dataset", "my-dataset.tar", repo_type="dataset"
        )
        with tarfile.open(path) as archive:
            archive.extractall("/root/data")
  launch:
    - label: Chạy tách rời để job không dừng khi mất kết nối
      language: bash
      code: |
        nohup libreyolo train \
          model=LibreYOLO9s.pt \
          data=/root/data/my-dataset/data.yaml \
          epochs=100 batch=-1 imgsz=640 \
          project=/root/runs name=run1 \
          > /root/train.log 2>&1 &
    - label: Multi-GPU từ tệp Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="/root/data/my-dataset/data.yaml",
                epochs=100,
                batch=64,          # batch toàn cục trên mọi GPU
                device="0,1,2,3",
                project="/root/runs",
                name="run1",
            )
  watch:
    - label: Một lượt đọc nhẹ
      language: bash
      code: |
        cat /root/runs/run1/status.json
    - label: Từ script
      language: python
      code: |
        import json

        with open("/root/runs/run1/status.json") as handle:
            status = json.load(handle)

        print(status["state"], status["current_epoch"], status["eta_seconds"])
        print(status.get("metrics"))
    - label: Trong trình duyệt qua tunnel SSH
      language: bash
      code: |
        # Trên máy thuê (mặc định bind 127.0.0.1:8420):
        libreyolo monitor /root/runs/run1 --no-browser

        # Từ máy của bạn, sau đó mở http://localhost:8420 cục bộ:
        #   ssh -L 8420:localhost:8420 <user>@<host>
  push:
    - label: Đẩy trọng số tới nơi lưu trữ lâu dài
      language: bash
      code: |
        huggingface-cli upload my-org/my-run \
          /root/runs/run1/weights/best.pt best.pt
source_hash: 75d314de06aca3b6
---

## Trước khi thuê bất cứ thứ gì

Hai quyết định sẽ tốn kém hơn nhiều nếu để tới sau.

Trước hết, hãy đưa dataset lên CDN. Đóng gói thành một tệp tar trong repo
dataset Hugging Face hoạt động giống nhau trên mọi nhà cung cấp, phân phối nhanh
cho tất cả và chỉ cần `HF_TOKEN` trong môi trường job khi repo là riêng tư. Sao
chép dataset lên từ kết nối gia đình hoặc tải từ nguồn chậm trên máy thuê đều là
thời gian GPU phải trả phí chỉ để chờ.

<code-tabs name="stage" />

Sau đó chọn dung lượng ổ đĩa. Nhà cung cấp tính phí lưu trữ theo dung lượng được
cấp phát, không phải dung lượng đã dùng, và không thể thu nhỏ ổ đĩa sau khi tạo.
Cộng dung lượng dữ liệu đã chuẩn bị, checkpoint và khoảng 30 phần trăm khoảng
trống dự phòng, rồi dừng ở đó.

## Cài đặt trên máy thuê

<code-tabs name="install" />

Cài PyTorch trước nếu image chưa có bản build CUDA khớp với card, sau đó mới cài
LibreYOLO để pip không tự phân giải bản torch chỉ dùng CPU. Snippet thứ hai
không phải thủ tục tùy ý: wheel được build cho kiến trúc GPU sai sẽ báo
`torch.cuda.is_available() == True` rồi thất bại ở phép toán thực đầu tiên với
`CUDA error: no kernel image is available for execution on the device`. Một
phép nhân ma trận sẽ phát hiện vấn đề trước khi bạn mất một giờ thiết lập.

Trỏ `HF_HOME` tới bộ nhớ lâu dài nếu nhà cung cấp có volume, để checkpoint và
dataset đã tải tồn tại giữa các lượt chạy.

## Khởi chạy

Chạy job ở chế độ tách rời. Phiên tương tác chết theo kết nối mạng sẽ kéo cả quá
trình huấn luyện dừng theo.

<code-tabs name="launch" />

`batch=-1` đặc biệt đáng dùng ở đây vì thường bạn đang dùng card chưa từng huấn
luyện trước đó. Nó thăm dò mô hình ở training mode bằng backward pass thực và
chọn lũy thừa lớn nhất của hai có thể chạy, nhanh hơn việc tìm giới hạn qua lỗi
hết bộ nhớ sau hai mươi phút. Xem [Siêu tham
số](/docs/train/hyperparameters).

Trên máy multi-GPU, `device="0,1,2,3"` tự tạo một worker cho mỗi GPU và `batch`
vẫn là batch toàn cục trên tất cả GPU. Bắt buộc phải có guard `__main__` vì mỗi
worker import lại script. Nội dung này và các hành vi phân tán khác nằm trong
[Huấn luyện multi-GPU](/docs/train/multi-gpu).

## Theo dõi từ bên ngoài

Mỗi lượt chạy ghi `status.json` vào thư mục lượt chạy và thay thế tệp theo cách
atomic sau mỗi epoch. Đây là lượt đọc nhẹ: vài trăm byte mang trạng thái, epoch
hiện tại, ETA và metric mới nhất mà không cần parse log.

<code-tabs name="watch" />

Tệp `metrics.jsonl` bên cạnh có toàn bộ lịch sử theo epoch, còn `train.log` có
đầu ra console. `libreyolo monitor` cung cấp dashboard trình duyệt trên cả ba
bằng thư viện chuẩn, vì vậy máy thuê không cần cài gì ngoài chính LibreYOLO. Kết
nối tới dashboard qua chuyển tiếp cổng SSH.

Không nội dung nào trong số này tác động tới process huấn luyện, vì vậy chúng có
thể gắn vào lượt chạy đang hoạt động, mở lại lượt chạy đã xong hoặc kiểm tra lượt
chạy bị lỗi.

## Lấy trọng số ra trước khi ngừng trả phí

Máy thuê là tài nguyên dùng rồi bỏ. Hãy đẩy checkpoint ở các mốc, không chỉ ở
cuối, vì nếu không, crash, preemption hoặc hết tiền sẽ làm mất toàn bộ lượt chạy.

<code-tabs name="push" />

`weights/best.pt` và `weights/last.pt` được ghi sau mỗi epoch và mỗi lần cải
thiện. `save_period=N` bổ sung snapshot `weights/epoch_<N>.pt`, giúp việc đẩy dữ
liệu giữa lượt chạy trở nên nhẹ. `summary.json` và `results.csv`, ở những family
có ghi chúng, cũng nhỏ và đáng lấy về.

Callback trên `on_train_epoch_end` là cách rõ ràng để tự động hóa thao tác đẩy.
Xem [Logger thí nghiệm](/docs/train/loggers), nơi các backend được host còn cung
cấp metric mà không cần chạm vào máy thuê.

## Ngừng trả phí

Đây là phần thực sự tốn tiền khi xảy ra sai sót, và quy tắc khác nhau theo mô
hình của nhà cung cấp.

Trên marketplace cho thuê máy thô, hóa đơn tính theo thời gian thực cho đến khi
instance bị hủy. GPU nhàn rỗi có giá đúng bằng GPU bận, nên chỉ dừng process huấn
luyện không tiết kiệm gì. Instance đã dừng vẫn tính phí ổ đĩa.

Trên nền tảng serverless nơi job là một hàm được trang trí, container scale về 0
khi hàm trả về, nên nguy cơ quên máy thấp hơn nhiều. Job bị treo không có timeout
vẫn tính phí, vì vậy luôn đặt timeout.

Dừng thay vì hủy vừa là đòn bẩy thực, vừa là cạm bẫy thực. Kết quả đo trên máy
thuê 8x RTX 4090 với ổ đĩa 250 GB vào ngày 2026-07-31: trạng thái chạy tính phí
$3.4828 mỗi giờ, trạng thái dừng tính $0.0694 mỗi giờ chỉ cho ổ đĩa, còn trạng
thái đã hủy không tính phí. Đó là mức tiết kiệm 98 phần trăm trong khi vẫn giữ
môi trường, dữ liệu đã chuẩn bị và checkpoint.

Bạn có thể tính mức phí khi dừng trước khi thuê:

```text
stopped $/hr = allocated_GB * storage_cost_per_GB_per_month / 730
             = 250 * 0.20 / 730 = $0.0694/hr
```

So sánh nó với chi phí dựng lại: thuê lại, tải image, cài đặt và chuẩn bị lại dữ
liệu. Trên cùng máy, việc dựng lại mất khoảng 15 phút thiết lập cộng 43 GB dữ
liệu vào, tổng cộng xấp xỉ $1.00. So với $0.0694 mỗi giờ, quay lại trong khoảng
14 giờ thì nên dừng, còn khoảng nghỉ dài hơn thì nên hủy và dựng lại từ bản đã
chuẩn bị.

Có một rủi ro khiến việc dừng không an toàn với phần cứng khan hiếm: dừng sẽ
giải phóng GPU. Không có gì giữ trước chúng, vì vậy chỉ có thể khởi động lại nếu
host vẫn còn GPU trống. Ổ đĩa an toàn; GPU thì không.

## Serverless dưới dạng hàm

Nếu không muốn quản lý máy, cả Modal và Beam đều chạy một hàm Python được trang
trí trên GPU và scale về 0 khi hàm trả về. Bộ test hằng đêm riêng của LibreYOLO
chạy trên Modal, còn `tools/ci/modal_nightly.py` trong repo thư viện là ví dụ
hoạt động ngay trong repo để tham khảo.

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")   # các thư viện hệ thống OpenCV
    .pip_install("libreyolo[rfdetr]")
)
app = modal.App("libreyolo-train")
cache = modal.Volume.from_name("libreyolo-cache", create_if_missing=True)


@app.function(gpu="A100", timeout=6 * 60 * 60, volumes={"/cache": cache})
def train():
    import os

    os.environ["HF_HOME"] = "/cache/hf"          # cache trọng số giữa các lượt chạy

    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLO9s.pt")
    model.train(data="coco8.yaml", epochs=100, project="/cache/runs")
    cache.commit()                                # lưu lâu dài volume


@app.local_entrypoint()
def main():
    train.remote()
```

Chạy bằng `modal run modal_train.py`. Filesystem container là tạm thời, vì vậy
mọi thứ đáng giữ phải nằm trong volume hoặc được đẩy ra ngoài. Đặt `timeout=`
một cách tường minh; đó là điều duy nhất ngăn cách job bị treo với hóa đơn không
có điểm dừng.

Beam có cùng cấu trúc với decorator `@function`, một `Volume` và
`train.remote()` được gọi từ `__main__`.

## Chọn kích thước theo chi phí mỗi job

$/giờ là con số không phù hợp để tối ưu. Mô hình nhỏ chỉ dùng một phần card lớn,
vì vậy GPU rẻ hơn và chậm hơn thường rẻ hơn trên mỗi epoch. Chạy profiler trong
vài bước trên card thuê trước khi cam kết lượt chạy dài: nếu kết luận là
`dataloader` hoặc `host / launch`, GPU nhanh hơn không giúp gì, còn tăng worker
hoặc batch sẽ giúp nhiều. Xem [Hiệu năng huấn
luyện](/docs/train/performance).

## Nội dung liên quan

- Xem [Dataset](/docs/train/datasets) để biết bố cục mà archive đã chuẩn bị cần
  có và lệnh doctor bắt lỗi trước khi GPU bắt đầu tính phí.
- Xem [Huấn luyện multi-GPU](/docs/train/multi-gpu) để biết về máy nhiều card.
