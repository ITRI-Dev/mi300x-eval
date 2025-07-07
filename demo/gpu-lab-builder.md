# GPU Lab Builder Demo Guide

## What is GPU Lab Builder?

GPU Lab Builder is a tool that automatically detects your GPU setup (NVIDIA or AMD) and builds the Docker container for machine learning.

**What it does:**
- **Auto-detects** your GPU type (NVIDIA CUDA or AMD ROCm)
- **Finds GPU tools** even in non-standard installation paths
- **Builds ready-to-use** machine learning containers
- **Validates everything** before starting

---

## Quick Start

### Step 1: Get the Files Ready

Make sure you have these files in your directory:
```
gpu-lab-builder/
├── create_image.sh          # The magic script
├── Dockerfile.nvidia        # For NVIDIA GPUs
├── Dockerfile.rocm          # For AMD GPUs
└── src/
    ├── entrypoint.sh
    ├── smart_gpu_test.py
    ├── gpu_keepalive.py
    └── rocm-smi_wrapper.py   # AMD only
```

### Step 2: Make it Executable

```bash
chmod +x create_image.sh
```

### Step 3: Run the Builder

```bash
./create_image.sh
```

The script will do everything else automatically.

---

## AMD GPU Demo (MI300X)

### Check Your MI300X Setup

```bash
rocm-smi
```

**You should see:**
```
============================================ ROCm System Management Interface ============================================
====================================================== Concise Info ======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)
==========================================================================================================================
0       2     0x74a1,   23018  47.0°C      171.0W    NPS1, SPX, 0        137Mhz  900Mhz  0%   auto  750.0W  0%     0%
1       3     0x74a1,   53458  53.0°C      177.0W    NPS1, SPX, 0        142Mhz  900Mhz  0%   auto  750.0W  0%     0%
==========================================================================================================================
```

Two MI300X beasts with 192GB each.

### Run the Builder

```bash
./create_image.sh
```

**Here's Builder flow:**
```
==========================================
        GPU Docker Image Builder
==========================================
Docker environment check passed
Checking NVIDIA environment...
nvidia-smi command not found
Checking ROCm environment...
ROCm environment check passed
ROCm environment detected, building ROCm image

ROCm image requires:
  - Dockerfile.rocm
  - src/entrypoint.sh
  - src/gpu_keepalive.py
  - src/rocm-smi_wrapper.py
  - src/smart_gpu_test.py

==========================================
The following Docker images will be built:
==========================================
• rocm-lab image (using Dockerfile.rocm)
  Based on: rocm/pytorch:latest
  Includes: PyTorch, SSH service, GPU testing tools

==========================================
Confirm building the above images? (y/N): 
```

**Type `y` and hit Enter:**

```
==========================================
Checking required files for all images...
==========================================
Checking required files for rocm image...
■ ROCm image required files complete

==========================================
■ All required files check passed!
==========================================

Starting image build...
==========================================
Building ROCm image...
[+] Building 0.7s (15/15) FINISHED
■ ROCm image built successfully: rocm-lab
Launch command: docker run --device=/dev/kfd --device=/dev/dri --group-add video -it --name rocm-lab_container rocm-lab bash

■ All images built successfully!
Script execution completed!
```

Created a `rocm-lab` image!

### Test Your MI300X Container

```bash
docker run --device=/dev/kfd --device=/dev/dri --group-add video -it --name rocm-lab_container rocm-lab bash
```

**Inside the container:**
```bash
python3 /app/smart_gpu_test.py
```

**Expected results:**
```
=== GPU Stress Testing Tool ===
Selected GPU 0: AMD Instinct MI300X
  Total memory: 191.98 GB
  Multi-processor count: 304
  Compute capability: 9.4

=== Test Completed ===
Total time: 1.03 seconds
Successful iterations: 5000/5000
Success rate: 100.0%
Average performance: 83,090.86 GFLOPS

✓ All tests completed!
```

83K GFLOPS !

---

## NVIDIA GPU Demo (A6000)

### Check Your NVIDIA Setup

```bash
nvidia-smi
```

**You should see:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA RTXA6000-24Q            On  |   00000000:06:10.0 Off |                    0 |
| N/A   N/A    P8             N/A /  N/A  |     814MiB /  24576MiB |      0%      Default |
+-----------------------------------------------------------------------------------------+
```

24GB of VRAM ready to go.

### Run the Builder

```bash
./create_image.sh
```

**Here's Builder flow:**
```
==========================================
        GPU Docker Image Builder
==========================================
Docker environment check passed
Checking NVIDIA environment...
NVIDIA environment check passed
Checking ROCm environment...
rocm-smi command not found
NVIDIA environment detected, building NVIDIA image

NVIDIA image requires:
  - Dockerfile.nvidia
  - src/entrypoint.sh
  - src/gpu_keepalive.py
  - src/smart_gpu_test.py

==========================================
The following Docker images will be built:
==========================================
• nvidia-lab image (using Dockerfile.nvidia)
  Based on: nvidia/cuda:12.4.0-runtime-ubuntu22.04
  Includes: TensorFlow, SSH service, GPU testing tools

==========================================
Confirm building the above images? (y/N): 
```

**Type `y` and press Enter:**

```
==========================================
Checking required files for all images...
==========================================
Checking required files for nvidia image...
■ NVIDIA image required files complete

==========================================
■ All required files check passed!
==========================================

Starting image build...
==========================================
Building NVIDIA image...
[+] Building 1.4s (13/13) FINISHED
■ NVIDIA image built successfully: nvidia-lab
Launch command: docker run --gpus all -it --name nvidia-lab_container nvidia-lab bash

■ All images built successfully!
Script execution completed!
```

Created a `nvidia-lab` image!

### Test Your NVIDIA Container

```bash
docker run --gpus all -it --name nvidia-lab_container nvidia-lab bash
```

**Inside the container:**
```bash
python3 /app/smart_gpu_test.py
```

**Expected results:**
```
=== GPU Stress Testing Tool ===
GPU 0: NVIDIA RTXA6000-24Q
  compute_capability: (8, 6)
  device_name: NVIDIA RTXA6000-24Q

=== Test Completed ===
Total time: 12.38 seconds
Successful iterations: 5000/5000
Success rate: 100.0%
Average performance: 6,936.11 GFLOPS

✓ All tests completed!
```

About 7K GFLOPS performance on the A6000.

---

## Comparison of Two Setups

| GPU | Memory | Performance | Test Time | Speed |
|-----|---------|-------------|-----------|--------|
| **AMD MI300X** | 192GB | 83,091 GFLOPS | 1.03s | **12x faster** |
| **NVIDIA A6000** | 24GB | 6,936 GFLOPS | 12.38s | Baseline |

---

## Features

### Auto-Detection Magic

The builder automatically finds GPU tools even in non-standard places:

**ROCm paths it searches:**
- `/usr/bin/rocm-smi` (package install)
- `/usr/local/bin/rocm-smi` (local install)
- `/opt/rocm/bin/rocm-smi` (official ROCm)
- `/opt/rocm/libexec/rocm_smi/rocm_smi.py` (Python version)

**NVIDIA paths it searches:**
- `/usr/bin/nvidia-smi` (standard install)
- `/usr/local/bin/nvidia-smi` (manual install)
- `/usr/lib/wsl/lib/nvidia-smi` (WSL environment)
- `/opt/nvidia/bin/nvidia-smi` (NVIDIA installer)

### File Validation

Before building, it checks for all required files:
```
Checking required files for nvidia image...
■ NVIDIA image required files complete
```

---

## Dual GPU Environments

If you have both NVIDIA and ROCm on the same machine, you get this menu:

```
Please select the image type to build:
1) NVIDIA image
2) ROCm image  
3) Build both NVIDIA and ROCm images
```

Choose option 3 to build both for testing and comparison.

---

## You Will Get

### ROCm Lab Container (~60.3GB)  
- **PyTorch** with ROCm support
- **SSH server** (password: `root`)
- **rocm-smi** monitoring
- **Conda environment** (Python 3.12)
- **Testing scripts** ready to run

### NVIDIA Lab Container (~10.2GB)
- **TensorFlow** with GPU support
- **SSH server** (password: `root`)
- **nvidia-smi** and **nvtop** monitoring
- **Python 3** environment
- **Testing scripts** ready to run

---

## Next Steps

After building your images:

1. **For ROCm:**
   ```bash
   docker run --device=/dev/kfd --device=/dev/dri --group-add video -it --name my-rocm-lab rocm-lab bash
   ```

2. **For NVIDIA:**
   ```bash
   docker run --gpus all -it --name my-nvidia-lab nvidia-lab bash
   ```

3. **Install your frameworks:**
   - Jupyter Notebook
   - Additional ML libraries
   - Your custom code
   
---

## Common Issues & Fixes

### Docker Not Running
```
Error: Docker service is not running. Please start Docker service
```
**Fix:** `sudo systemctl start docker`

### No GPU Drivers
```
Error: Neither NVIDIA nor ROCm environment detected
```
**Fix:** Install proper GPU drivers first

### Permission Denied
```
bash: ./create_image.sh: Permission denied
```
**Fix:** `chmod +x create_image.sh`

### Missing Files
```
Error: Missing required files:
  Dockerfile.nvidia
  smart_gpu_test.py
```
**Fix:** Make sure all files are in the same directory