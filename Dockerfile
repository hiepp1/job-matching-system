# ==========================================
# GIAI ĐOẠN 1: BUILDER (Cài đặt thư viện)
# ==========================================
FROM python:3.10-slim as builder

WORKDIR /app

# 1. Cài đặt các gói hệ thống cần thiết để build (git, gcc...)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Tạo môi trường ảo (Virtual Environment) tại /opt/venv
RUN python -m venv /opt/venv
# Kích hoạt môi trường ảo cho các lệnh tiếp theo
ENV PATH="/opt/venv/bin:$PATH"

# 3. Copy requirements
COPY requirements.txt .

# 4. Cài đặt PyTorch CPU 
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. Cài đặt các thư viện còn lại
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# GIAI ĐOẠN 2: RUNNER (Chạy ứng dụng)
# ==========================================
FROM python:3.10-slim

WORKDIR /app

# 1. Cài đặt thư viện hệ thống tối thiểu cho OpenCV (nếu dùng)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. COPY MÔI TRƯỜNG ẢO TỪ BUILDER SANG
COPY --from=builder /opt/venv /opt/venv

# 3. Thiết lập biến môi trường để sử dụng venv ngay lập tức
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# 4. Copy mã nguồn dự án
COPY . .

# 5. Mở cổng
EXPOSE 7860

# 6. Chạy ứng dụng
CMD ["python", "app.py"]