FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN echo 'Acquire::AllowInsecureRepositories "true";' > /etc/apt/apt.conf.d/90ignore-check && \
    echo 'Acquire::AllowUnauthenticated "true";' >> /etc/apt/apt.conf.d/90ignore-check && \
    sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/cuda* && \
    rm -f /etc/apt/sources.list.d/nvidia* && \
    apt-get update && \
    apt-get install -y --no-install-recommends wget bzip2 git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN echo "channels:\n\
  - defaults\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch\n\
show_channel_urls: true" > ~/.condarc

WORKDIR /workspace/MeInTime
COPY . /workspace/MeInTime

# 4️⃣ 创建 Conda 虚拟环境 + 安装 pip 依赖（强制使用国内源）
RUN conda create -n meintime -y python=3.10 --override-channels \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch && \
    conda run -n meintime pip install --upgrade pip && \
    conda run -n meintime pip install --timeout 300 --no-cache-dir -r requirements.txt && \
    conda clean -afy

COPY xformers-0.0.24+cu118-cp310-cp310-manylinux2014_x86_64.whl /tmp/
RUN conda run -n meintime pip install /tmp/xformers-0.0.24+cu118-cp310-cp310-manylinux2014_x86_64.whl



# 5️⃣ 激活环境 shell
SHELL ["conda", "run", "-n", "diffbir", "/bin/bash", "-c"]

# 6️⃣ 默认启动命令
CMD ["bash"]
