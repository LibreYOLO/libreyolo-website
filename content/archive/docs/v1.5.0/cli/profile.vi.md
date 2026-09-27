---
title: libreyolo profile
seo_title: tham chiếu lệnh libreyolo profile
description: >-
  Đo tốc độ huấn luyện và suy luận (inference) rồi đọc kết quả: mọi subcommand
  của profile, tham số và giá trị mặc định của chúng, và những gì từng góc đọc
  báo cáo.
lead: >-
  Một nhóm lệnh đo xem thời gian đi vào đâu trong một bước huấn luyện hoặc một
  lần gọi inference, ghi ra một profile độc lập, rồi đọc lại profile đó qua
  nhiều góc nhìn.
keywords:
  - libreyolo profile cli
  - profiling huấn luyện yolo
  - đo độ trễ inference yolo
  - profiling kernel gpu pytorch
  - so sánh hiệu năng libreyolo
last_verified: 1.5.0
meta:
  - label: Lệnh
    value: libreyolo profile
    mono: true
  - label: Đầu ra
    value: profile.json và profile_trace.json trong runs/profile
    mono: true
snippets:
  examples:
    - label: Đo inference
      language: bash
      code: |
        # Không truyền tham số source thì dùng ảnh mẫu đi kèm
        libreyolo profile infer --device cpu --warmup 5 --runs 20
    - label: Đọc kết luận
      language: bash
      code: |
        libreyolo profile summary runs/profile/infer/profile.json
    - label: So sánh hai lần đo
      language: bash
      code: >
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --project
        runs/profile/a

        libreyolo profile infer --device cpu --warmup 5 --runs 20 --batch 4
        --project runs/profile/b


        libreyolo profile compare runs/profile/a/infer/profile.json \
          runs/profile/b/infer/profile.json
source_hash: b967e869fd9ba418
---

## Cú pháp

```bash
libreyolo profile <subcommand> [<positional>] [--flag value ...]
```

Nhóm lệnh này không nhận tham số dạng `key=value`. Các subcommand của nó dùng
tham số vị trí và cờ POSIX, nên phải viết `--weights LibreYOLO9t.pt`, không
phải `weights=LibreYOLO9t.pt`. Chạy `libreyolo profile` mà không kèm subcommand
sẽ in ra danh sách.

Hai subcommand thực hiện đo và ghi ra một profile; phần còn lại đọc profile. Cả
`run` và `infer` đều xuất ra cùng một tệp `profile.json` độc lập, nên mọi
subcommand đọc đều dùng được với cả hai.

## profile run

Chạy một phiên huấn luyện ngắn có đo đạc và ghi ra một profile.

```bash
libreyolo profile run <data> [--flag value ...]
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `data` | | Tham số vị trí. Tệp YAML của tập dữ liệu (dataset) hoặc tên dataset, ví dụ `coco128`. Bắt buộc |
| `--weights` | `LibreYOLO9t.pt` | Tệp trọng số hoặc tên mô hình |
| `--size` | `t` | Biến thể kích thước mô hình |
| `--batch` | `16` | Micro-batch. `-1` tự động chiếm khoảng 70% VRAM |
| `--imgsz` | `640` | Kích thước ảnh huấn luyện |
| `--workers` | `8` | Số worker của dataloader |
| `--amp` | `true` | Dùng nhánh AMP của họ mô hình. `--no-amp` tắt nó đi |
| `--steps` | `20` | Số bước được profile, tức là được đo |
| `--warmup` | `5` | Số bước warmup trước khi đo |
| `--repeat` | `1` | Lặp lại N lần để lấy giá trị trung bình và độ lệch chuẩn |
| `--device` | `0` | Thiết bị |
| `--project` | `runs/profile` | Thư mục gốc chứa đầu ra |
| `--json` | `false` | In đầu ra JSON ra stdout |

Cửa sổ đo là `--warmup` cộng `--steps` vòng lặp. Một dataset quá nhỏ để lấp đầy
cửa sổ đó sẽ không tạo ra profile nào và lệnh thoát với mã `3`, kèm theo ba
hướng xử lý: dùng dataset lớn hơn, giảm số bước, hoặc giảm kích thước batch.

`--repeat` lớn hơn 1 sẽ ghi ra tệp tổng hợp `runs/profile/profile_repeat.json`,
trong đó các chỉ số vô hướng được lấy trung bình qua các lần thử, còn danh sách
kernel lấy từ lần thử cuối cùng. Đây cũng là điều kiện cần để `compare` đưa ra
kết luận về ý nghĩa thống kê: một lần chạy đơn lẻ không đủ để kết luận.

## profile infer

Đo đường inference và ghi ra một profile.

```bash
libreyolo profile infer [<source>] [--flag value ...]
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `source` | | Tham số vị trí. Ảnh hoặc thư mục. Nếu bỏ trống thì dùng ảnh mẫu đi kèm |
| `--weights` | `LibreYOLO9t.pt` | Tệp trọng số hoặc tên mô hình |
| `--size` | `t` | Biến thể kích thước mô hình |
| `--batch` | `1` | Số ảnh mỗi lượt forward |
| `--imgsz` | `640` | Kích thước ảnh đầu vào |
| `--half` | `false` | Autocast khi forward, chỉ với CUDA. `--no-half` tắt nó đi |
| `--amp-dtype` | `float16` | Kiểu dữ liệu autocast của CUDA: `float16` hoặc `bfloat16` |
| `--warmup` | `20` | Số vòng warmup trước khi đo |
| `--runs` | `100` | Số vòng được đo |
| `--repeat` | `1` | Lặp lại N lần để lấy giá trị trung bình và độ lệch chuẩn |
| `--conf` | `0.25` | Ngưỡng độ tin cậy, thứ làm thay đổi khối lượng công việc của NMS |
| `--iou` | `0.45` | Ngưỡng IoU của NMS |
| `--max-det` | `300` | Số detection tối đa mỗi ảnh, thứ làm thay đổi khối lượng công việc của NMS |
| `--device` | `0` | Thiết bị |
| `--trace` | `true` | Xuất một Chrome trace để đi sâu vào kernel và op. `--no-trace` bỏ qua bước này |
| `--project` | `runs/profile` | Thư mục gốc chứa đầu ra |
| `--json` | `false` | In đầu ra JSON ra stdout |

Báo cáo độ trễ ở p50, p90 và p99, thông lượng (throughput) tính bằng số ảnh mỗi
giây, và phân tách theo từng giai đoạn: preprocess, forward và postprocess. Ba
tham số ngưỡng có mặt ở đây vì chúng làm thay đổi con số của postprocess.

## profile summary

```bash
libreyolo profile summary <trace> [--json]
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `trace` | | Tham số vị trí. Đường dẫn tới một tệp `profile.json` hoặc `profile_trace.json`. Bắt buộc |
| `--json` | `false` | In đầu ra JSON ra stdout |

Góc đọc tổng quan: thời gian mỗi bước, throughput, mức sử dụng GPU, tỷ lệ Tensor
Core, đỉnh VRAM, overhead phía host, số lần khởi chạy kernel mỗi bước, kết luận
về điểm nghẽn kèm lý do, tỷ lệ kernel theo từng nhóm, và các kernel nặng nhất
mỗi bước. Với một profile inference, lệnh còn in thêm các phân vị độ trễ và
phân tách theo giai đoạn.

Một profile được đo trong lúc VRAM bị thrash sẽ được đánh dấu, vì mức sử dụng và
throughput đo được khi đó không đáng tin.

## profile get

```bash
libreyolo profile get <trace> [<field>] [--json]
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `trace` | | Tham số vị trí. Đường dẫn tới một profile. Bắt buộc |
| `field` | | Tham số vị trí. Tên chỉ số. Bỏ trống để liệt kê các chỉ số có sẵn |
| `--json` | `false` | In đầu ra JSON ra stdout |

Chỉ in ra đúng một chỉ số và không gì khác, phục vụ các vòng lặp trong script.
Một field không hợp lệ sẽ thoát với mã `2` và chỉ về dạng liệt kê.

## profile phases

```bash
libreyolo profile phases <trace> [--json]
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `trace` | | Tham số vị trí. Đường dẫn tới một profile. Bắt buộc |
| `--json` | `false` | In đầu ra JSON ra stdout |

Số mili giây GPU, số mili giây thực tế, số kernel và số op cho từng phase:
forward, backward, dataload, to_device, optimizer.

## profile kernels

```bash
libreyolo profile kernels <trace> [--flag value ...]
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `trace` | | Tham số vị trí. Đường dẫn tới một profile. Bắt buộc |
| `--top` | `20` | Hiển thị N kernel dẫn đầu theo thời gian GPU |
| `--category` | | Lọc theo chuỗi con của tên nhóm: `gemm`, `layout`, `norm`, `elementwise` |
| `--grep` | | Lọc theo biểu thức chính quy trên tên kernel |
| `--tensorcore` | `false` | Chỉ các kernel dùng Tensor Core |
| `--sort` | `time` | `time`, `count` hoặc `name` |
| `--phase` | | Giới hạn trong một phase: `forward`, `backward`, `dataload`, `to_device`, `optimizer` |
| `--json` | `false` | In đầu ra JSON ra stdout |

Tầng sâu nhất của phân tích: từng kernel GPU riêng lẻ kèm tỷ lệ thời gian GPU,
số mili giây mỗi bước, số lần gọi mỗi bước và nhóm. Một giá trị `--phase` không
hợp lệ sẽ thoát với mã `2` và liệt kê các phase mà profile đó có.

## profile ops

```bash
libreyolo profile ops <trace> [--flag value ...]
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `trace` | | Tham số vị trí. Đường dẫn tới một profile. Bắt buộc |
| `--top` | `20` | Hiển thị N op dẫn đầu theo thời gian CPU |
| `--phase` | | Giới hạn trong một phase |
| `--json` | `false` | In đầu ra JSON ra stdout |

Góc nhìn từ framework thay vì từ thiết bị: các op `aten` và autograd được xếp
hạng theo thời gian CPU, đó là nơi chi phí khởi chạy từ phía host lộ ra.

## profile compare

```bash
libreyolo profile compare <before> <after> [--json]
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `before` | | Tham số vị trí. Profile làm mốc. Bắt buộc |
| `after` | | Tham số vị trí. Profile mới. Bắt buộc |
| `--json` | `false` | In đầu ra JSON ra stdout |

So sánh chênh lệch về throughput, số mili giây mỗi ảnh, mức sử dụng GPU,
overhead phía host, số lần khởi chạy kernel mỗi bước và kết luận về điểm nghẽn.

Kết luận về ý nghĩa thống kê đòi hỏi cả hai phía đều được đo với `--repeat` ít
nhất bằng 2. Khi có điều đó, một chênh lệch được coi là có ý nghĩa khi nó vượt
quá hai lần sai số chuẩn gộp, và đầu ra sẽ in ra phép so sánh đã thực hiện. Nếu
không, dòng đó ghi rằng một lần chạy đơn lẻ không đủ cơ sở để kết luận.

## profile what-if

```bash
libreyolo profile what-if <trace> [--flag value ...]
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `trace` | | Tham số vị trí. Đường dẫn tới một profile. Bắt buộc |
| `--remove-category` | | Ước tính khi bỏ đi một nhóm kernel: `gemm`, `layout`, `norm`, `elementwise` |
| `--remove-launches` | | Ước tính khi bỏ đi N lần khởi chạy kernel mỗi bước, ví dụ một khoản lợi từ việc hợp nhất op |
| `--json` | `false` | In đầu ra JSON ra stdout |

Ước tính một thay đổi sẽ đem lại gì trước khi thay đổi đó được viết ra. Bắt buộc
phải có một trong hai tùy chọn; không có tùy chọn nào thì lệnh thoát với mã `2`.

Phép ước tính đi theo chính kết luận của profile. Dưới 80% mức sử dụng GPU, nó
mô hình hóa phần tiết kiệm được bằng số lần khởi chạy giảm đi nhân với chi phí
host mỗi lần khởi chạy đã đo; trên mức đó, bằng phần công việc GPU giảm đi. Kết
quả kèm theo một field cảnh báo, vì chi phí mỗi lần khởi chạy chỉ là con số xấp
xỉ và bằng chứng duy nhất là một lần đo thứ hai.

## Ví dụ

<code-tabs name="examples" />

## Ghi chú

Trình profiler chỉ đo và báo cáo. Nó không thay đổi gì cả: đọc kết luận, sửa
cấu hình hoặc mã nguồn, chạy lại, rồi so sánh, đó chính là vòng lặp mà nó được
xây dựng để phục vụ.

`--device` mặc định là `0`, tức thiết bị CUDA số 0. Truyền `--device cpu` sẽ đo
trên CPU và tạo ra một profile mà các subcommand đọc vẫn chấp nhận, chỉ thiếu
phần chi tiết kernel GPU.

Mọi subcommand đều hỗ trợ `--json`, và các subcommand đọc chỉ in ra stdout,
chính điều đó khiến nhóm lệnh này dùng được từ trong script.

Mã thoát ở đây là của riêng nhóm lệnh này: `2` cho một tệp không tồn tại hoặc
một tham số không phân giải được, `3` khi `run` không tạo ra profile nào, và `1`
khi không phân tích được một trace.

Liên quan: [`libreyolo train`](/docs/cli/train), các tham số của lệnh đó chính
là thứ mà một profile huấn luyện thường được đo để điều chỉnh.
