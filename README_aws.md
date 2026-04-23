# Hướng dẫn Thực hành LAB 16: Cloud AI Environment Setup (2.5h)

Chào mừng các bạn đến với Lab 16. Trong bài thực hành này, chúng ta sẽ thiết lập một môi trường Cloud AI hoàn chỉnh trên AWS bằng cách sử dụng **Terraform** (Infrastructure as Code) và **Docker/vLLM**. 

Mục tiêu của bài lab là triển khai một mô hình ngôn ngữ lớn (LLM - cụ thể là `google/gemma-4-E2B-it`) lên một máy chủ GPU (NVIDIA T4) nằm an toàn trong mạng nội bộ (Private VPC), và cung cấp API truy cập ra bên ngoài thông qua Load Balancer.

---

## Phần 1: Chuẩn bị tài khoản AWS và thiết lập IAM (Least-Privilege)

Để làm việc với AWS an toàn, chúng ta không bao giờ sử dụng tài khoản Root. Thay vào đó, bạn sẽ tạo một IAM User thuộc một IAM Group với các quyền vừa đủ (least-privilege) để Terraform có thể triển khai hạ tầng.

### Bước 1.1: Truy cập AWS Console
1. Đăng nhập vào [AWS Management Console](https://console.aws.amazon.com/) bằng tài khoản Root hoặc tài khoản Admin của bạn.
2. Trên thanh tìm kiếm, gõ **IAM** và chọn dịch vụ **IAM (Identity and Access Management)**.

### Bước 1.2: Tạo IAM Group và gắn quyền (Policies)
1. Trong menu bên trái của IAM, chọn **User groups** -> click **Create group**.
2. Đặt tên nhóm: `AI-Lab-Group`.
3. Trong phần **Attach permissions policies**, bạn cần tìm và tick chọn các quyền (roles) sau. **Giải thích tại sao cần:**
   - `AmazonEC2FullAccess`: Cần thiết để Terraform tạo máy chủ ảo (Bastion Host, GPU Node), Key Pairs, và Security Groups.
   - `AmazonVPCFullAccess`: Cần thiết để Terraform tạo môi trường mạng (VPC, Subnets, Internet Gateway, NAT Gateway, Route Tables).
   - `ElasticLoadBalancingFullAccess`: Cần thiết để tạo Application Load Balancer (ALB) giúp phân phối traffic từ internet vào private GPU Node.
   - `IAMFullAccess`: Bắt buộc vì Terraform script của chúng ta sẽ tạo một IAM Role và Instance Profile (gắn vào GPU node để cấp quyền cho node nếu cần tương tác với AWS services sau này).
4. Click **Create user group**.

### Bước 1.3: Tạo IAM User và lấy Access Keys
1. Trong menu bên trái, chọn **Users** -> click **Create user**.
2. Đặt tên user: `ai-lab-user`. Click Next.
3. Chọn **Add user to group**, tick chọn nhóm `AI-Lab-Group` vừa tạo. Click Next -> **Create user**.
4. Bấm vào tên user `ai-lab-user` vừa tạo. Chuyển sang tab **Security credentials**.
5. Kéo xuống phần **Access keys**, click **Create access key**.
6. Chọn **Command Line Interface (CLI)** -> Check đồng ý -> Next -> **Create access key**.
7. **LƯU Ý:** Copy `Access key ID` và `Secret access key` lưu vào nơi an toàn. Bạn sẽ không thể xem lại Secret key sau khi đóng cửa sổ này.

### Bước 1.4: Tăng hạn mức vCPU cho GPU (Rất quan trọng)
Theo mặc định, AWS khóa hạn mức sử dụng máy chủ GPU của các tài khoản mới ở mức 0 vCPU để bảo mật. Bạn cần mở khóa để chạy được instance `g4dn.xlarge` (cần 4 vCPU).
1. Trên thanh tìm kiếm của AWS Console, gõ **Service Quotas** và chọn nó.
2. Menu trái chọn **AWS services** -> tìm và chọn **Amazon Elastic Compute Cloud (Amazon EC2)**.
3. Ở ô tìm kiếm của Quotas, gõ `Running On-Demand G and VT instances`.
4. Chọn nó và click **Request quota increase**.
5. Nhập số **4** (tương đương 4 vCPU cho 1 máy `g4dn.xlarge`). 
*Lưu ý: AWS có thể mất từ vài phút đến vài giờ để duyệt yêu cầu này.*

> **⚠️ Ghi chú quan trọng cho tài khoản mới / Free Tier:** Nếu yêu cầu tăng quota GPU bị từ chối hoặc chưa được duyệt trong thời gian làm lab, hãy chuyển sang **[Phần 7: Phương án Dự phòng — CPU Instance với LightGBM](#phần-7-phương-án-dự-phòng--cpu-instance-với-lightgbm-khi-không-xin-được-quota-gpu)**. Đây là phương án thay thế hợp lệ và sẽ được chấm điểm tương đương.

---

## Phần 2: Cài đặt và cấu hình môi trường Local

Trên máy tính cá nhân của bạn, mở Terminal/Command Prompt.

### Bước 2.1: Cấu hình AWS CLI
Đảm bảo bạn đã cài đặt [AWS CLI](https://aws.amazon.com/cli/). Gõ lệnh sau để cấu hình tài khoản vừa tạo:
```bash
aws configure
```
Nhập các thông tin:
- **AWS Access Key ID**: (Dán Access key ID của bạn)
- **AWS Secret Access Key**: (Dán Secret access key của bạn)
- **Default region name**: `us-east-1` (Bắt buộc dùng us-east-1 cho lab này)
- **Default output format**: `json`

### Bước 2.2: Lấy Hugging Face Token
Mô hình `google/gemma-4-E2B-it` là một mô hình bị giới hạn (gated model). Bạn cần cấp quyền truy cập cho Terraform.
1. Đăng nhập [Hugging Face](https://huggingface.co/).
2. Vào trang của model [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) và đồng ý với điều khoản (Accept license).
3. Vào **Settings** -> **Access Tokens** -> Tạo một token (quyền Read) và copy lại.

---

## Phần 3: Triển khai Hạ tầng với Terraform

Terraform là công cụ giúp chúng ta khởi tạo hạ tầng AWS hoàn toàn tự động bằng code. Kiến trúc bao gồm:
- Mạng **Private VPC** cách ly hoàn toàn với bên ngoài.
- **Bastion Host** (t3.micro) ở Public Subnet: Dùng làm trạm trung chuyển an toàn nếu cần SSH vào GPU Node.
- **GPU Node** (g4dn.xlarge - T4 GPU) ở Private Subnet: Chạy Docker chứa vLLM để load model AI.
- **NAT Gateway**: Cho phép Private Subnet kéo image Docker và tải Model từ internet.
- **Application Load Balancer (ALB)**: Mở cổng 80 (HTTP) để nhận API request và đẩy vào GPU node ở cổng 8000.

### Bước 3.1: Khởi tạo Terraform
Di chuyển vào thư mục code Terraform:
```bash
cd terraform
terraform init
```

### Bước 3.2: Cấu hình biến môi trường
Thiết lập Token Hugging Face của bạn để Terraform truyền vào máy chủ EC2 khi khởi động:
```bash
export TF_VAR_hf_token="<DÁN_TOKEN_HUGGING_FACE_CỦA_BẠN_VÀO_ĐÂY>"
```

### Bước 3.3: Triển khai (Apply)
Chạy lệnh apply để Terraform bắt đầu tạo tài nguyên trên AWS:
```bash
terraform apply
```
Gõ `yes` khi được hỏi. Quá trình này sẽ mất khoảng **10 đến 15 phút** (phần lớn thời gian là để khởi tạo NAT Gateway).

*Mẹo: Các bạn hãy bắt đầu bấm giờ (benchmark) từ lúc gõ `yes` ở bước này nhé!*

---

## Phần 4: Kiểm tra AI Endpoint (Inference)

Khi `terraform apply` chạy xong, màn hình terminal sẽ in ra các thông số quan trọng (Outputs). Trông sẽ giống thế này:
```text
Outputs:

alb_dns_name = "ai-inference-alb-xxxxxx.us-east-1.elb.amazonaws.com"
bastion_public_ip = "100.x.x.x"
endpoint_url = "http://ai-inference-alb-xxxxxx.us-east-1.elb.amazonaws.com/v1/completions"
gpu_private_ip = "10.0.1x.x"
```

**Quan trọng:** Mặc dù Terraform đã báo thành công, GPU Node vẫn đang ngầm tải Docker image (vLLM) và model weights (~vài GB) từ Hugging Face. **Bạn cần đợi thêm 5-10 phút** để model sẵn sàng.

### Bước 4.1: Gọi API bằng cURL
Thay thế URL của ALB bạn nhận được vào lệnh dưới đây và chạy thử:

```bash
curl -X POST http://<THAY_BẰNG_ALB_DNS_NAME_CỦA_BẠN>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "messages": [
      {"role": "system", "content": "Bạn là một trợ lý AI hữu ích."},
      {"role": "user", "content": "Hãy giải thích Bastion Host trong AWS là gì?"}
    ],
    "max_tokens": 150
  }'
```
Nếu nhận được câu trả lời từ AI, chúc mừng bạn đã triển khai thành công! Hãy ghi lại tổng thời gian (Cold start time) từ lúc chạy `terraform apply` đến lúc nhận được API response đầu tiên.

---

## Phần 5: Tiêu chí nộp bài (Deliverables)

Để hoàn thành Lab 16, sinh viên cần thu thập và nộp các kết quả sau:
1. **Ảnh chụp màn hình (Screenshot) API gọi thành công:** Chụp lại lệnh curl và câu trả lời của AI.
2. **Ảnh chụp màn hình AWS Billing/Cost Dashboard:** 
   - Vào AWS Console -> Gõ **Billing** trên thanh tìm kiếm.
   - Chụp lại màn hình thể hiện các dịch vụ đang chạy phát sinh chi phí (EC2, NAT Gateway).
3. **Report Cold Start Time:** Ghi lại tổng thời gian triển khai (Mục tiêu: < 15 phút cho instance T4).
4. **Mã nguồn:** Nén thư mục chứa file Terraform đã chạy thành công.

---

## Phần 6: Dọn dẹp tài nguyên (CỰC KỲ QUAN TRỌNG)

GPU EC2 (`g4dn.xlarge`) và NAT Gateway tính phí theo giờ và **rất đắt**. Ngay sau khi test thành công và chụp ảnh nộp bài, bạn **BẮT BUỘC** phải xóa toàn bộ tài nguyên để tránh mất tiền.

Chạy lệnh sau trong thư mục `terraform`:
```bash
terraform destroy
```
Gõ `yes` khi được hỏi. Quá trình xóa sẽ mất khoảng 5 phút. Hãy đợi đến khi terminal báo `Destroy complete!` để chắc chắn mọi thứ đã bị xóa.

---

## Phần 7: Phương án Dự phòng — CPU Instance với LightGBM (Khi không xin được Quota GPU)

> **Ghi chú (tiếng Việt):** Đây là phương án dành cho các bạn dùng tài khoản AWS mới hoặc Free Tier. Do các tài khoản mới bị hạn chế quota GPU nghiêm ngặt (mặc định = 0 vCPU cho dòng G/VT), yêu cầu tăng quota thường bị trì hoãn hoặc từ chối. Thay vì bỏ qua bài lab, bạn sẽ chuyển sang triển khai một **bài toán Machine Learning thực tế** (LightGBM — gradient boosting) trên một **instance CPU cao cấp**. Quy trình này vẫn đầy đủ: Terraform IaC → Cloud instance → Training → Inference → Billing check, chỉ khác là không cần GPU.

### 7.1: Thay đổi instance type trong Terraform

Mở file `terraform/main.tf` và tìm dòng khai báo GPU Node (khoảng dòng 209):

```hcl
# Trước (GPU):
instance_type = "g4dn.xlarge"

# Sau (CPU cao cấp — 8 vCPU, 32 GB RAM):
instance_type = "r5.2xlarge"
```

> **Tại sao `r5.2xlarge`?** Instance này có 8 vCPU và 32 GB RAM, không yêu cầu quota đặc biệt, đủ mạnh để chạy gradient boosting với dataset hàng trăm nghìn dòng và không cần Deep Learning AMI. Chi phí ~$0.504/giờ tại us-east-1 — tương đương `g4dn.xlarge` (~$0.526/giờ).

Ngoài ra, cập nhật AMI trong cùng resource sang Amazon Linux 2023 thông thường (không cần Deep Learning AMI):

```bash
# Lấy AMI ID của Amazon Linux 2023 tại us-east-1
aws ec2 describe-images \
  --region us-east-1 \
  --owners amazon \
  --filters "Name=name,Values=al2023-ami-*-x86_64" \
            "Name=state,Values=available" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" \
  --output text
```

Sau đó thay giá trị AMI trong `main.tf` GPU Node block bằng AMI ID vừa lấy được.

### 7.2: Triển khai hạ tầng

```bash
cd terraform
export TF_VAR_hf_token="dummy"   # Không cần HF token khi chạy LGBM
terraform apply
```

Gõ `yes` khi được hỏi. Quá trình tạo hạ tầng (NAT Gateway, ALB) mất khoảng **10–15 phút** như bình thường.

### 7.3: Kết nối vào CPU Instance

Từ Terraform outputs, lấy `bastion_public_ip` và `gpu_private_ip` (bây giờ là CPU node):

```bash
# SSH vào Bastion Host
ssh -i <KEY_FILE>.pem ec2-user@<BASTION_PUBLIC_IP>

# Từ Bastion, SSH vào CPU Node
ssh ec2-user@<CPU_PRIVATE_IP>
```

### 7.4: Cài đặt môi trường ML

Trên CPU Node, chạy các lệnh sau:

```bash
# Cập nhật hệ thống và cài Python packages
sudo dnf update -y
sudo dnf install -y python3 python3-pip

pip3 install --upgrade pip
pip3 install lightgbm scikit-learn pandas numpy kaggle

# Tạo thư mục làm việc
mkdir -p ~/ml-benchmark && cd ~/ml-benchmark
```

### 7.5: Tải Dataset từ Kaggle

Chúng ta sẽ dùng **Credit Card Fraud Detection** — bộ dữ liệu chuẩn cho benchmark ML với 284,807 giao dịch thực.

**Lấy Kaggle API Key:**
1. Đăng nhập [kaggle.com](https://www.kaggle.com) -> **Settings** -> **API** -> **Create New Token** -> tải về `kaggle.json`.
2. Copy nội dung file vào máy EC2:

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

### 7.6: Kết quả Benchmark trên `r5.2xlarge`

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

1. Vào [AWS Billing Console](https://console.aws.amazon.com/billing/) -> **Bills** hoặc **Cost Explorer**.
2. Chọn ngày hôm nay để xem chi phí hiện tại.
3. Chụp màn hình thể hiện các dịch vụ phát sinh chi phí.

**Ước tính chi phí 1 giờ (us-east-1):**

| Dịch vụ | Instance/Loại | Chi phí/giờ |
|---|---|---|
| EC2 — CPU Node | `r5.2xlarge` | ~$0.504 |
| EC2 — Bastion | `t3.micro` | ~$0.010 |
| NAT Gateway | (mỗi AZ) | ~$0.045 + data |
| ALB | Application Load Balancer | ~$0.008 |
| **Tổng ước tính** | | **~$0.57/giờ** |

> **Ghi chú (tiếng Việt):** Chi phí thực tế có thể dao động nhẹ tùy vào lượng data transfer. Hãy chụp màn hình billing ngay sau 1 giờ rồi chạy `terraform destroy` để tránh phát sinh thêm chi phí. Instance CPU `r5.2xlarge` có chi phí tương đương GPU `g4dn.xlarge` (~$0.526/giờ) nhưng không cần xin quota đặc biệt — đây là điểm khác biệt quan trọng khi làm việc với tài khoản mới.

### 7.8: Tiêu chí nộp bài (Phương án CPU thay thế)

Nếu sử dụng phương án CPU + LightGBM, nộp các mục sau (được chấm tương đương phương án GPU):

1. **Screenshot terminal** chạy `python3 benchmark.py` với toàn bộ output kết quả.
2. **File `benchmark_result.json`** chứa metrics đầy đủ (training time, AUC, inference latency).
3. **Screenshot AWS Billing** sau 1 giờ triển khai, thể hiện EC2 và NAT Gateway.
4. **Mã nguồn** thư mục `terraform/` đã chỉnh sửa (với `r5.2xlarge`).
5. **Báo cáo ngắn** (5–10 dòng): so sánh kết quả training time, AUC, inference speed; giải thích lý do phải dùng CPU thay GPU.

---

> **Lưu ý cuối (tiếng Việt):** Dù chạy GPU hay CPU, **bước dọn dẹp (Phần 6 — `terraform destroy`) là bắt buộc** ngay sau khi nộp bài. Instance `r5.2xlarge` và NAT Gateway vẫn tính phí liên tục theo giờ dù không có tác vụ nào đang chạy.