# ITRI AMD MI300X GPU Evaluation Project

## Overview

This repository contains the results of a one-month collaborative evaluation between **ITRI ICL** and **AMD**, testing the AMD MI300X GPU (x2) for machine learning and distributed training workloads during June-July 2025.

During this evaluation period, we developed and tested three projects that demonstrate the capabilities of AMD ROCm on MI300X hardware. Our goal is to share these findings with the community to promote both ITRI's software solutions and AMD's powerful MI300X GPU platform.

## System Requirements

**For AMD ROCm (Recommended):**
- AMD ROCm drivers and runtime
- ROCm-compatible GPU
- Docker with GPU support

**For NVIDIA (Also supported):**
- NVIDIA drivers + CUDA
- nvidia-container-toolkit
- Docker with GPU support

**General Requirements:**
- Docker and Docker Compose
- Ubuntu 22.04+ or compatible Linux
- Sufficient disk space (60GB+ for full setup)

## Projects

Three independent projects that can be used separately or together:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  GPU Lab        │    │  GPU Docker      │    │  ROCm DDP       │
│  Builder        │    │  Service         │    │  Training       │
│                 │    │                  │    │                 │
│ • Auto-detect   │    │ • RESTful API    │    │ • Multi-GPU     │
│ • Build images  │    │ • Resource mgmt  │    │ • Multi-node    │
│ • ROCm/CUDA     │    │ • Snapshots      │    │ • Standalone    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Projects Developed

### 1. [GPU Lab Builder](https://github.com/ITRI-Dev/gpu-lab-builder)
**Automatic GPU-enabled Docker Image Creation Tool**

A automatic script that detects your GPU environment (NVIDIA CUDA or AMD ROCm) and builds ready-to-use machine learning Docker containers.

**Key Features:**
- **Auto-detection**: Automatically finds NVIDIA or ROCm environments
- **Path-smart**: Searches common GPU tool installation paths
- **Interactive setup**: Guides you through the build process
- **Pre-validation**: Checks all required files before building

**What You Get:**
- **NVIDIA Lab**: ~10.2GB container with TensorFlow GPU, SSH, monitoring tools
- **ROCm Lab**: ~60.3GB container with PyTorch ROCm, SSH, testing tools
- Ready-to-use GPU development environments

**Quick Start:**
```bash
chmod +x create_image.sh
./create_image.sh
```

### 2. [GPU Docker Service](https://github.com/ITRI-Dev/gpu-docker-service)
**RESTful API for GPU Container Management**

A FastAPI-based service that makes managing GPU containers as easy as a web API call. Perfect for VDI systems and multi-user GPU environments.

**Key Features:**
- **Auto GPU allocation**: Smart GPU resource management
- **SSH auto-assignment**: Automatic port allocation for remote access
- **Resource monitoring**: Real-time CPU, memory, and GPU usage
- **Snapshot & backup**: Save and restore container states
- **Cross-machine deployment**: Move containers between systems

**API Highlights:**
- Create GPU containers with automatic resource allocation
- Start/stop/restart container lifecycle management
- Take snapshots for quick state restoration
- Create backups for long-term storage and migration
- Monitor resource usage in real-time

**Quick Start:**
```bash
sudo mkdir -p /backups && sudo chmod -R 777 /backups
sudo ./setup_env.sh
```

### 3. [ROCm DDP Training Framework](https://github.com/ITRI-Dev/rocm-ddp)
**Standalone Interactive Multi-Model Distributed Training (No dependencies on other projects)**

A standalone Docker-based solution for distributed PyTorch training on AMD ROCm GPUs, supporting both single-node and multi-node setups.
(**Note:** Multi-node support is implemented in this framework, but has not yet been validated in a physical multi-node environment.)

**Key Features:**
- **Multi-GPU training**: Support for 1-8 GPUs per node
- **Multi-node support**: Scale across multiple machines
- **Interactive & script modes**: Choose your workflow
- **Auto resource management**: Smart GPU allocation and port handling
- **Model variety**: ResNet, EfficientNet, Vision Transformer, and more
- **Standalone design**: Complete self-contained training environment

**Supported Models:**
- ResNet series (18, 50, 101, 152)
- EfficientNet (B0-B7)
- Vision Transformer
- DenseNet, VGG, MobileNet, and more

**Quick Start:**
```bash
./build.sh  # Build the Docker image first
./run-ddp.sh 2  # Interactive mode with 2 GPUs
./run-ddp.sh 1 script_single_node "--model resnet50 --learning_rate 0.001 --optimizer adam"
```

## Demo Guides
- [GPU Lab Builder](./demo/gpu-lab-builder.md)
- [GPU Docker Service for ROCM](./demo/gpu-docker-service_rocm.md)
- [GPU Docker Service for NVIDIA](./demo/gpu-docker-service_nvidia.md)
- [ROCm DDP Training Framework](./demo/rocm-ddp.md)

## Performance Results

### GPU Performance Comparison

We tested our GPU Lab containers on both AMD MI300X and NVIDIA A6000 to benchmark performance:

#### AMD MI300X Results
```
Selected GPU 0: AMD Instinct MI300X
Total memory: 191.98 GB
Multi-processor count: 304
Compute capability: 9.4

=== Test Results ===
Total time: 1.03 seconds
Successful iterations: 5000/5000
Success rate: 100.0%
Average performance: 83,090.86 GFLOPS
```

#### NVIDIA A6000 Results (Comparison)
```
GPU 0: NVIDIA RTXA6000-24Q
Compute capability: 8.6

=== Test Results ===
Total time: 12.38 seconds
Successful iterations: 5000/5000
Success rate: 100.0%
Average performance: 6,936.11 GFLOPS
```

### Key Performance Insights

**AMD MI300X Advantages:**
- 12x faster compared to NVIDIA top-tier consumer GPU
- ROCm ecosystem has Excellent PyTorch integration

### DDP Training Performance

Our distributed training framework demonstrated excellent scaling characteristics on AMD MI300X hardware:

#### Single GPU vs Dual GPU Comparison

**Test Configuration:**
- Model: ResNet101
- Dataset: CIFAR-100
- Optimizer: SGD
- Learning rate: 0.002
- Epochs: 100
- Normalization: neg_one_one

**Single GPU Setup (MI300X):**
```bash
./run-ddp.sh 1 script_single_node
```
- **GPU allocation**: 1 × MI300X
- **Batch size**: 512
- **Training time**: ~1.18 minutes per epoch
- **Total training**: ~1.97 hours (100 epochs, 7079 seconds)

**Dual GPU Setup (MI300X):**
```bash
./run-ddp.sh 2 script_single_node  
```
- **GPU allocation**: 2 × MI300X 
- **Batch size**: 256 per GPU (512 total, same as 1 GPU)
- **Training time**: ~0.63 minutes per epoch
- **Total training**: ~1.04 hours (100 epochs, 3754 seconds)
- **Speed improvement**: 1.89x faster than single GPU

#### Training Time Performance Table

| Configuration | GPU Count | Batch Size | Time/Epoch | Total Time (100 epochs) | Speed Improvement |
|---------------|-----------|------------|------------|-------------------------|-------------------|
| Single GPU    | 1 × MI300X | 512       | 1.18 min   | 1.97 hours              | Baseline          |
| Dual GPU      | 2 × MI300X | 512 total | 0.63 min   | 1.04 hours              | **1.89x faster**  |

#### Scaling Performance Insights

**Training Time Comparison:**
- **1 GPU**: 1.97 hours total (70.8 sec/epoch × 100 epochs)
- **2 GPU**: 1.04 hours total (37.5 sec/epoch × 100 epochs)  
- **Time saved**: 55.4 minutes (47% reduction)
- **Scaling efficiency**: 94.5% (near-perfect linear scaling)

**Benefits of Multi-GPU DDP:**
- **1.89x speed improvement**: Dual GPU reduces training time by 47%
- **Linear scaling**: 94.5% efficiency indicates excellent parallelization
- **Parallel efficiency**: ROCm DDP shows optimal GPU utilization
- **Cost per epoch**: Reduced from 70.8 to 37.5 seconds per epoch

#### Sample Training Commands

**1 GPU Demo:**
```bash
# Start interactive 1-GPU training
./run-ddp.sh 1

# Or with script mode parameters
./run-ddp.sh 1 script_single_node "--model resnet101 --learning_rate 0.002 --optimizer sgd --batch_size 1024 --num_epochs 100"
```

**2 GPU Demo:**
```bash
# Start interactive 2-GPU training  
./run-ddp.sh 2

# Or with script mode parameters
./run-ddp.sh 2 script_single_node "--model resnet101 --learning_rate 0.002 --optimizer sgd --batch_size 1024 --num_epochs 100"
```

**Multi-node Scaling (Future):**
```bash
# Scale to multiple machines
./run-ddp.sh 8 auto 192.168.1.100 script_multi_node "--model resnet152 --learning_rate 0.005 --optimizer sgd --batch_size 2048"
```

## Use Cases

**For Researchers:**
- Quick GPU environment setup
- Distributed model training
- Experiment state management

**For System Administrators:**
- Multi-user GPU resource management
- Container lifecycle automation
- Resource monitoring and allocation

**For Development Teams:**
- Consistent GPU development environments
- Easy scaling from single to multi-GPU
- Cross-system deployment capabilities

## Acknowledgments

**Special thanks to AMD** for providing access to the MI300X GPU hardware during our evaluation period. The performance and capabilities demonstrated have been exceptional, and we're excited to share these results with the community.

This collaboration between ITRI ICL and AMD showcases the potential of AMD's ROCm ecosystem for machine learning and distributed computing workloads.

---

*Built with ITRI ICL collaboration with AMD*
