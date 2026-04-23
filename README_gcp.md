# Hướng dẫn Thực hành LAB 16: Cloud AI Environment Setup (2.5h) - Phiên bản Google Cloud Platform (GCP)

Chào mừng các bạn đến với Lab 16 phiên bản Google Cloud Platform (GCP). Trong bài thực hành này, chúng ta sẽ thiết lập một môi trường Cloud AI hoàn chỉnh trên GCP bằng cách sử dụng **Terraform** (Infrastructure as Code) và **Docker/vLLM**.

Mục tiêu của bài lab là triển khai mô hình ngôn ngữ lớn (`google/gemma-4-E2B-it`) lên một máy chủ GPU (NVIDIA T4) nằm an toàn trong mạng nội bộ (Private VPC), và cung cấp API truy cập ra bên ngoài thông qua Load Balancer.

---

## Phần 1: Chuẩn bị tài khoản GCP và thiết lập IAM (Least-Privilege)

Trên GCP, mọi tài nguyên đều thuộc về một **Project**. Bạn cần tạo một Project và cấp quyền vừa đủ (least-privilege) cho một Service Account hoặc tài khoản thực hành để Terraform có thể triển khai hạ tầng.

### Bước 1.1: Tạo GCP Project
1. Đăng nhập vào [Google Cloud Console](https://console.cloud.google.com/).
2. Nhấp vào menu chọn Project ở thanh trên cùng (cạnh logo Google Cloud) -> Chọn **New Project**.
3. Đặt tên Project (ví dụ: `ai-lab-16-gcp`) và nhấp **Create**.
4. **LƯU Ý:** Ghi lại **Project ID** (thường có dạng `ai-lab-16-gcp-123456`). Bạn sẽ cần nó cho Terraform.
5. Chắc chắn rằng bạn đã bật **Billing** (thanh toán) cho Project này để có thể tạo tài nguyên.

### Bước 1.2: Kích hoạt các API cần thiết
Để Terraform có thể tạo tài nguyên (máy ảo, network), bạn cần bật các API tương ứng trên Project. Mở **Cloud Shell** (biểu tượng `>_` trên góc phải) và chạy lệnh:
```bash
gcloud services enable compute.googleapis.com iam.googleapis.com
```

### Bước 1.3: Cấp quyền IAM (Least Privilege)
Nếu bạn tự làm lab trên máy cá nhân bằng tài khoản Google của mình (tài khoản đã tạo Project), bạn mặc định có quyền Owner và đã đủ quyền. Tuy nhiên, theo best practice (hoặc nếu phân quyền cho một user/Service Account khác để Terraform chạy), bạn cần vào **IAM & Admin** -> **IAM** và cấp các Roles sau:
- `Compute Admin` (`roles/compute.admin`): Để tạo Compute Engine (VM, GPU, Load Balancer, VPC, Firewall, Cloud NAT).
- `Service Account User` (`roles/iam.serviceAccountUser`): Để gán Service Account cho máy ảo Compute Engine.

### Bước 1.4: Tăng hạn mức (Quota) cho GPU (Rất quan trọng)
Giống như AWS, GCP mặc định khóa quota GPU (hạn mức = 0) cho các Project mới để phòng chống lạm dụng đào coin. Bạn cần xin tăng quota để chạy được máy ảo gắn GPU T4.
1. Trên thanh tìm kiếm của GCP Console, gõ **Quotas** và chọn trang **Quotas (IAM & Admin)**.
2. Tại bộ lọc (Filter), tìm kiếm thuộc tính:
   - `Quota: GPUs (all regions)`
   - `Quota: NVIDIA T4 GPUs`
3. Tích chọn vào quota **NVIDIA T4 GPUs** tại region bạn định triển khai (ví dụ: `us-central1`).
4. Nhấp **Edit Quotas** -> Điền số lượng mong muốn là **1** -> Gửi yêu cầu (Submit request).
*Lưu ý: Quá trình GCP xét duyệt tăng Quota có thể mất từ vài phút đến 24 giờ. Hãy thực hiện bước này càng sớm càng tốt.*

> **⚠️ Ghi chú quan trọng cho tài khoản mới / Free Tier:** Nếu yêu cầu tăng quota GPU bị từ chối hoặc chưa được duyệt trong thời gian làm lab, hãy chuyển sang **[Phần 7: Phương án Dự phòng — CPU Instance với LightGBM](#phần-7-phương-án-dự-phòng--cpu-instance-với-lightgbm-khi-không-xin-được-quota-gpu)**. Đây là phương án thay thế hợp lệ và sẽ được chấm điểm tương đương.

---

## Phần 2: Cài đặt và cấu hình môi trường Local

Trên máy tính cá nhân của bạn, mở Terminal/Command Prompt.

### Bước 2.1: Cài đặt và xác thực Google Cloud SDK (gcloud CLI)
Đảm bảo bạn đã cài đặt [Google Cloud CLI](https://cloud.google.com/sdk/docs/install). Gõ lệnh sau để xác thực tài khoản và chọn Project:
```bash
# Đăng nhập vào GCP
gcloud auth login

# Cấp quyền cho Terraform (Application Default Credentials)
gcloud auth application-default login

# Thiết lập Project ID mặc định
gcloud config set project <PROJECT_ID_CỦA_BẠN>
```

### Bước 2.2: Lấy Hugging Face Token
Mô hình `google/gemma-4-E2B-it` là một mô hình bị giới hạn (gated model).
1. Đăng nhập [Hugging Face](https://huggingface.co/).
2. Vào trang mô hình [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) và đồng ý với điều khoản (Accept license).
3. Vào **Settings** -> **Access Tokens** -> Tạo một token (quyền Read) và copy lại.

---

## Phần 3: Triển khai Hạ tầng với Terraform

Kiến trúc AI Server trên GCP bao gồm:
- **VPC & Subnets**: Một Private Subnet tại `us-central1`.
- **Cloud NAT & Cloud Router**: Hoạt động tương tự NAT Gateway trên AWS, cho phép VM ẩn trong Private Subnet truy cập internet để tải Docker và Model.
- **Bastion Host (Identity-Aware Proxy)**: Trong lab GCP, chúng ta sẽ sử dụng [IAP TCP forwarding](https://cloud.google.com/iap/docs/tcp-forwarding-overview) làm giải pháp truy cập SSH hiện đại và an toàn (không tốn chi phí chạy VM Bastion riêng như AWS).
- **GPU Node**: Máy ảo `n1-standard-4` gắn 1x `nvidia-tesla-t4`, sử dụng Deep Learning VM Image, nằm hoàn toàn trong Private Subnet.
- **Cloud Load Balancing**: External HTTP Load Balancer để nhận request từ internet và chuyển vào port 8000 của GPU Node.
- **VPC Firewall Rules**: Chỉ cho phép dải IP của IAP SSH (cổng 22) và dải IP của Load Balancer Healthcheck (cổng 8000) truy cập vào GPU Node.

### Bước 3.1: Khởi tạo Terraform
Di chuyển vào thư mục code Terraform GCP (giả sử là `terraform-gcp`):
```bash
cd terraform-gcp
terraform init
```

### Bước 3.2: Cấu hình biến môi trường
Thiết lập Token Hugging Face và Project ID của bạn để Terraform sử dụng:
```bash
export TF_VAR_project_id="<PROJECT_ID_CỦA_BẠN>"
export TF_VAR_hf_token="<DÁN_TOKEN_HUGGING_FACE_CỦA_BẠN_VÀO_ĐÂY>"
```

### Bước 3.3: Triển khai (Apply)
Chạy lệnh apply để Terraform tạo toàn bộ tài nguyên:
```bash
terraform apply
```
Gõ `yes` khi được hỏi. Quá trình triển khai hạ tầng mạng trên GCP thường rất nhanh (chưa tới 5 phút).

*Mẹo: Các bạn hãy bắt đầu bấm giờ (benchmark) từ lúc gõ `yes` ở bước này nhé!*

---

## Phần 4: Kiểm tra AI Endpoint (Inference)

Khi lệnh `terraform apply` chạy xong, bạn sẽ thấy Outputs cung cấp địa chỉ IP tĩnh của Load Balancer:
```text
Outputs:

alb_ip_address = "34.120.x.x"
endpoint_url = "http://34.120.x.x/v1/completions"
gpu_private_ip = "10.0.1.x"
```

**Quan trọng:** Mặc dù Terraform báo tạo xong hạ tầng, GPU Node bên trong vẫn đang chạy script tải Docker image (vLLM) và model weights (~vài GB) từ Hugging Face. **Bạn cần đợi thêm khoảng 5-10 phút** để model được nạp hoàn toàn vào VRAM của GPU.

### Bước 4.1: Gọi API bằng cURL
Sử dụng IP của Load Balancer để thực hiện truy vấn AI. Hãy thử đoạn lệnh sau:

```bash
curl -X POST http://<THAY_BẰNG_ALB_IP_ADDRESS_CỦA_BẠN>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "messages": [
      {"role": "system", "content": "Bạn là một chuyên gia về Google Cloud."},
      {"role": "user", "content": "Hãy giải thích ngắn gọn Cloud NAT trong Google Cloud là gì?"}
    ],
    "max_tokens": 150
  }'
```
Nếu nhận được câu trả lời từ AI, chúc mừng bạn đã triển khai thành công AI Endpoint trên GCP! Ghi lại tổng thời gian (Cold start time) từ lúc chạy `terraform apply` đến lúc lệnh `curl` thành công.

### Bước 4.2: SSH vào GPU Node qua IAP (Dùng để Debug - Tùy chọn)
Nếu gặp lỗi và cần kiểm tra log của vLLM, bạn không cần SSH key phức tạp hay Bastion Host. Trên GCP, bạn dùng lệnh `gcloud` kết hợp IAP tunnel:
```bash
# Truy cập vào máy ảo nằm trong Private Subnet
gcloud compute ssh <TÊN_GPU_NODE> --zone=us-central1-a --tunnel-through-iap

# Sau khi vào được bên trong VM, xem log của Docker:
sudo docker logs -f vllm
```

---

## Phần 5: Tiêu chí nộp bài (Deliverables)

Để hoàn thành Lab 16 trên môi trường GCP, bạn cần nộp các kết quả sau:
1. **Ảnh chụp màn hình (Screenshot) API gọi thành công:** Chụp lại Terminal chứa lệnh `curl` và kết quả trả về từ mô hình Gemma.
2. **Ảnh chụp màn hình Billing/Cost Dashboard:** 
   - Truy cập **Billing** -> **Reports** trên Google Cloud Console.
   - Chụp lại màn hình thể hiện các dịch vụ đang phát sinh chi phí (Compute Engine, Load Balancing, Cloud NAT).
3. **Report Cold Start Time:** Cung cấp thông số thời gian triển khai từ lúc khởi tạo đến lúc inference thành công (Mục tiêu: < 15 phút cho GPU T4).
4. **Mã nguồn:** Nén file cấu hình thư mục Terraform GCP của bạn và đính kèm.

---

## Phần 6: Dọn dẹp tài nguyên (CỰC KỲ QUAN TRỌNG)

Máy chủ chứa GPU (`nvidia-tesla-t4`), Cloud NAT và External IP trên GCP sẽ bị trừ tiền liên tục theo giây/phút. Ngay sau khi test thành công và chụp màn hình nộp bài, bạn **BẮT BUỘC** phải xóa toàn bộ tài nguyên:

Mở Terminal và chạy lệnh:
```bash
terraform destroy
```
Gõ `yes` để xác nhận việc xóa. Sau khi xóa xong, bạn có thể đăng nhập lại GCP Console để kiểm tra lần cuối, đảm bảo không còn máy ảo (VM instances) nào đang ở trạng thái `Running`.

---

## Phần 7: Phương án Dự phòng — CPU Instance với LightGBM (Khi không xin được Quota GPU)

> **Ghi chú (tiếng Việt):** Đây là phương án dành cho các bạn dùng tài khoản GCP mới hoặc Free Tier ($300 credit). GCP mặc định khóa quota GPU ở mức 0 cho mọi Project mới và quá trình xét duyệt tăng quota đôi khi bị từ chối do tài khoản chưa đủ lịch sử thanh toán. Thay vì bỏ qua bài lab, bạn sẽ chuyển sang triển khai một **bài toán Machine Learning thực tế** (LightGBM — gradient boosting) trên một **instance CPU cao cấp**. Quy trình này vẫn đầy đủ: Terraform IaC → Cloud VM → Training → Inference → Billing check, chỉ khác là không cần GPU.

### 7.1: Thay đổi cấu hình Terraform sang CPU Instance

GCP đã hỗ trợ biến `machine_type`, `gpu_type` và `gpu_count` trong `terraform-gcp/variables.tf`. Để chuyển sang CPU, bạn cần:

**Bước 1 — Thiết lập biến môi trường để đổi machine type và tắt GPU:**

```bash
export TF_VAR_project_id="<PROJECT_ID_CỦA_BẠN>"
export TF_VAR_machine_type="n2-standard-8"
export TF_VAR_gpu_count=0
export TF_VAR_hf_token="dummy"   # Không cần HF token khi chạy LGBM
```

**Bước 2 — Tắt GPU accelerator block trong `terraform-gcp/main.tf`:**

Tìm và comment out block `guest_accelerator` (khoảng dòng 108–111) và `scheduling` block bắt buộc `on_host_maintenance = "TERMINATE"`. Thay bằng:

```hcl
# guest_accelerator {   # <-- Comment out toàn bộ block này
#   type  = var.gpu_type
#   count = var.gpu_count
# }

scheduling {
  on_host_maintenance = "MIGRATE"   # MIGRATE thay vì TERMINATE
  automatic_restart   = true
}
```

> **Tại sao `n2-standard-8`?** Instance này có 8 vCPU và 32 GB RAM, không yêu cầu quota đặc biệt, có sẵn ngay trên tài khoản mới. Chi phí ~$0.382/giờ tại us-central1 — rẻ hơn GPU T4 (~$0.35/giờ chỉ GPU, chưa kể VM).

### 7.2: Triển khai hạ tầng CPU

```bash
cd terraform-gcp
terraform init
terraform apply
```

Gõ `yes` khi được hỏi. Hạ tầng GCP (VPC, NAT, Load Balancer) tạo rất nhanh, thường **< 5 phút**.

### 7.3: Kết nối vào CPU Instance qua IAP

Sau khi `terraform apply` hoàn tất, kết nối vào VM thông qua IAP (không cần Bastion Host riêng):

```bash
# Lấy tên instance từ output hoặc gõ trực tiếp
gcloud compute ssh ai-gpu-node --zone=us-central1-a --tunnel-through-iap --project=<PROJECT_ID_CỦA_BẠN>
```

### 7.4: Cài đặt môi trường ML

Trên VM, chạy các lệnh sau:

```bash
# Cập nhật và cài Python packages
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv

python3 -m pip install --upgrade pip
pip3 install lightgbm scikit-learn pandas numpy kaggle

# Tạo thư mục làm việc
mkdir -p ~/ml-benchmark && cd ~/ml-benchmark
```

### 7.5: Tải Dataset từ Kaggle

Chúng ta sẽ dùng **Credit Card Fraud Detection** — bộ dữ liệu chuẩn cho benchmark ML với 284,807 giao dịch thực.

**Lấy Kaggle API Key:**
1. Đăng nhập [kaggle.com](https://www.kaggle.com) -> **Settings** -> **API** -> **Create New Token** -> tải về `kaggle.json`.
2. Copy nội dung vào VM:

```bash
mkdir -p ~/.kaggle
# Tạo file credentials (thay YOUR_USERNAME và YOUR_KEY):
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username": "YOUR_KAGGLE_USERNAME", "key": "YOUR_KAGGLE_API_KEY"}
EOF
chmod 600 ~/.kaggle/kaggle.json

# Tải dataset
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/
```

### 7.6: Kết quả Benchmark trên `n2-standard-8`

| Metric | Kết quả |
|---|---|
| Thời gian load data | |
| Thời gian training | |
| Best iteration | |
| AUC-ROC | |
| Accuracy | |
| F1-Score | |
| Precision | |
| Recall | |
| Inference latency (1 row) | |
| Inference throughput (1000 rows) | |

### 7.7: Kiểm tra Chi phí sau 1 giờ

Sau khi chạy benchmark xong, **đợi tổng cộng 1 giờ** kể từ lúc `terraform apply` hoàn tất rồi kiểm tra chi phí:

1. Vào [GCP Billing Console](https://console.cloud.google.com/billing) -> **Reports**.
2. Chọn khoảng thời gian hôm nay để xem chi phí hiện tại theo từng dịch vụ.
3. Chụp màn hình thể hiện các dịch vụ đang phát sinh chi phí.

**Ước tính chi phí 1 giờ (us-central1):**

| Dịch vụ | Loại tài nguyên | Chi phí/giờ |
|---|---|---|
| Compute Engine — CPU Node | `n2-standard-8` | ~$0.382 |
| Cloud NAT | (xử lý egress traffic) | ~$0.044 + data |
| Cloud Load Balancing | External HTTP LB | ~$0.008 |
| **Tổng ước tính** | | **~$0.43/giờ** |

> **Ghi chú (tiếng Việt):** So sánh với GPU: Instance `n1-standard-4` + 1x NVIDIA T4 trên GCP có giá ~$0.35/giờ (GPU) + ~$0.19/giờ (VM) = ~$0.54/giờ. Phương án CPU `n2-standard-8` (~$0.43/giờ) thực ra **rẻ hơn** và có thể dùng ngay mà không cần chờ quota. Đây là bài học thực tế về việc lựa chọn infrastructure phù hợp với workload.

### 7.8: Tiêu chí nộp bài (Phương án CPU thay thế)

Nếu sử dụng phương án CPU + LightGBM, nộp các mục sau (được chấm tương đương phương án GPU):

1. **Screenshot terminal** chạy `python3 benchmark.py` với toàn bộ output kết quả.
2. **File `benchmark_result.json`** chứa metrics đầy đủ (training time, AUC, inference latency).
3. **Screenshot GCP Billing Reports** sau 1 giờ triển khai, thể hiện Compute Engine và Cloud NAT.
4. **Mã nguồn** thư mục `terraform-gcp/` đã chỉnh sửa (comment GPU block, `n2-standard-8`).
5. **Báo cáo ngắn** (5–10 dòng): so sánh kết quả training time, AUC, inference speed; giải thích lý do phải dùng CPU thay GPU.

---

> **Lưu ý cuối (tiếng Việt):** Dù chạy GPU hay CPU, **bước dọn dẹp (Phần 6 — `terraform destroy`) là bắt buộc** ngay sau khi nộp bài. VM `n2-standard-8`, Cloud NAT và External IP vẫn tính phí liên tục theo giây dù không có tác vụ nào đang chạy. Đừng bỏ qua bước này!