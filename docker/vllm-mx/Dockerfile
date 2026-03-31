# vLLM + ModelExpress P2P weight transfer with GPUDirect RDMA
#
# Stack:
#   1. NVIDIA rdma-core 54 from nccl-tests (Data Direct DMA-BUF support)
#   2. UCX 1.20.0 from source (built against NVIDIA rdma-core)
#   3. NIXL v1.0.0 vanilla from source (built against our UCX)
#   4. modelexpress Python package
#
# NO PATCHES — vanilla NIXL, proper rdma-core + UCX should make it work.
#
# Build (from repo root):
#   bob build --platform linux/arm64 \
#     -f docker/vllm-mx/Dockerfile \
#     -t accel-registry.cwcr.cloud/cw4637/vllm-mx:v2.4.0-mx0.3.0 \
#     --push .

# --- Stage 1: NVIDIA rdma-core (has Data Direct DMA-BUF) ---
FROM ghcr.io/coreweave/nccl-tests:12.9.1-devel-ubuntu22.04-nccl2.29.2-1-0bf62c2 AS rdma-source

# --- Stage 2: Build UCX + NIXL ---
FROM docker.cloudsmith.io/coreweave/infr/vllm:v2.4.0 AS builder

COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/libibverbs.so* /usr/lib/aarch64-linux-gnu/
COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/libmlx5.so* /usr/lib/aarch64-linux-gnu/
COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/librdmacm.so* /usr/lib/aarch64-linux-gnu/
COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/libibverbs/ /usr/lib/aarch64-linux-gnu/libibverbs/
COPY --from=rdma-source /usr/include/infiniband/ /usr/include/infiniband/

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential autoconf automake libtool pkg-config libnuma-dev \
        cmake git patchelf \
    && rm -rf /var/lib/apt/lists/*
RUN ldconfig

RUN grep -q "mlx5dv_get_data_direct_sysfs_path" /usr/include/infiniband/mlx5dv.h \
    && echo "Data Direct header: OK" \
    && nm -D /usr/lib/aarch64-linux-gnu/libmlx5.so | grep -q data_direct \
    && echo "Data Direct library: OK"

# UCX 1.20.0 from source
RUN git clone --depth 1 --branch v1.20.0 https://github.com/openucx/ucx.git /tmp/ucx && \
    cd /tmp/ucx && ./autogen.sh && ./configure \
      --disable-logging --disable-debug --disable-assertions --disable-params-check \
      --prefix=/usr/local/ucx --enable-shared --disable-static --disable-doxygen-doc \
      --enable-optimizations --enable-cma --enable-devel-headers \
      --with-cuda=/usr/local/cuda --with-verbs --with-dm --enable-mt \
    && grep "HAVE_DECL_MLX5DV_GET_DATA_DIRECT_SYSFS_PATH.*1" config.h \
    && echo "UCX: Data Direct DETECTED" \
    && make -j$(nproc) && make install \
    && cd / && rm -rf /tmp/ucx

# NIXL v1.0.0 with explicit CUDA memory type fix
COPY docker/vllm-mx/patch_nixl_vram.py /tmp/patch_nixl_vram.py
RUN pip install --no-cache-dir meson ninja pybind11 && \
    mkdir /tmp/nixl && cd /tmp/nixl && \
    wget "https://github.com/ai-dynamo/nixl/archive/refs/tags/v1.0.0.tar.gz" -qO- \
        | tar --strip-components=1 -xzf - && \
    python3 /tmp/patch_nixl_vram.py src/plugins/ucx/ucx_utils.cpp && \
    CUDA_TARGET="$(find /usr/local/cuda/targets -mindepth 1 -maxdepth 1 -type d | head -1)" && \
    meson setup build --prefix=/opt/nixl \
        -Ducx_path=/usr/local/ucx \
        -Dgds_path="${CUDA_TARGET}" \
        -Dcudapath_inc="${CUDA_TARGET}/include" \
        -Dcudapath_lib="${CUDA_TARGET}/lib" \
        -Dcudapath_stub="${CUDA_TARGET}/lib/stubs" && \
    cd build && ninja -j$(nproc) && ninja install && \
    cd / && rm -rf /tmp/nixl

# --- Stage 3: Final runtime ---
FROM docker.cloudsmith.io/coreweave/infr/vllm:v2.4.0

COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/libibverbs.so* /usr/lib/aarch64-linux-gnu/
COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/libmlx5.so* /usr/lib/aarch64-linux-gnu/
COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/librdmacm.so* /usr/lib/aarch64-linux-gnu/
COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/libibverbs/ /usr/lib/aarch64-linux-gnu/libibverbs/
COPY --from=builder /usr/local/ucx /usr/local/ucx
COPY --from=builder /opt/nixl /opt/nixl
RUN ldconfig 2>/dev/null || true

RUN pip uninstall -y nixl nixl-cu12 2>/dev/null || true
COPY modelexpress_client/python/ /tmp/mx/
RUN pip install --no-cache-dir /tmp/mx/ && rm -rf /tmp/mx && \
    pip uninstall -y nixl nixl-cu12 && \
    pip install --no-cache-dir --no-deps nixl

ENV PYTHONPATH=/opt/nixl/lib/python3/dist-packages:${PYTHONPATH} \
    LD_LIBRARY_PATH=/opt/nixl/lib/aarch64-linux-gnu:/usr/local/ucx/lib:${LD_LIBRARY_PATH} \
    UCX_MODULE_DIR=/usr/local/ucx/lib/ucx

RUN python3 -c "from modelexpress import register_modelexpress_loaders; print('OK')"
