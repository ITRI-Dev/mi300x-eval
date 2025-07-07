# ROCm DDP Training Demo Guide

## What is ROCm DDP?

ROCm DDP (Distributed Data Parallel) is a Docker-based framework for distributed deep learning training on AMD ROCm GPUs. It automatically manages resources, scales across multiple GPUs, and provides both interactive and automated training modes.

**What it does:**
- **Auto-scales** training across 1-8 GPUs per node
- **Manages resources** automatically (ports, GPU allocation)
- **Supports multi-node** distributed training
- **Interactive mode** with SSH access for development
- **Multiple models** (ResNet, EfficientNet, Vision Transformers)
- **Auto-downloads** CIFAR-100 dataset during build

---

## Quick Start

**Prerequisites:** The build script will automatically download CIFAR-100 dataset. For other datasets, manual preparation is required.

### Step 1: Check Your AMD GPUs

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
0       2     0x74a1,   23018  46.0°C      169.0W    NPS1, SPX, 0        134Mhz  900Mhz  0%   auto  750.0W  0%     0%
1       3     0x74a1,   53458  52.0°C      176.0W    NPS1, SPX, 0        141Mhz  900Mhz  0%   auto  750.0W  0%     0%
==========================================================================================================================
```

Two MI300X GPUs in NPS1+SPX mode (required).

### Step 2: Build the Training Environment

```bash
cd rocm-ddp/
./build.sh
```

**Expected output:**
```
Get DDP sources...
Build Docker image...
[+] Building 2.1s (13/13) FINISHED
■ Docker image built successfully: rocm-ddp
Prepare training data...
cifar-100-python/
cifar-100-python/file.txt~
cifar-100-python/train
cifar-100-python/test
cifar-100-python/meta
```

The build script automatically cleans any existing data directory and downloads fresh CIFAR-100 dataset.

### Step 3: Verify Data Setup

```bash
ls data/
# Should show: cifar-100-python.tar.gz  cifar-100-python/

ls data/cifar-100-python/
# Should show: file.txt~  meta  test  train
```

**If data download failed:** You can manually download it:
```bash
mkdir -p data
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -P ./data
cd data && tar -xvzf cifar-100-python.tar.gz && cd ..
```

### Step 4: Start Your First Training

```bash
./run-ddp.sh 2
```

This starts interactive mode with 2 GPUs.

---

## Single GPU Training Demo

### Quick Single GPU Test

The CIFAR-100 data should already be ready from the build step:

```bash
./run-ddp.sh 1 script_single_node "--model resnet101 --learning_rate 0.002 --optimizer sgd --batch_size 1024 --num_epochs 10"
```

**Performance results:**
```
Hello from local_rank 0, global_rank 0
Epoch - 0/10: time - 265.46s || loss_train - 4.3796 || accuracy_train - 0.1064
Accuracy on test dataset - 0.2912
Epoch - 1/10: time - 74.39s || loss_train - 3.2721 || accuracy_train - 0.4509
Epoch - 2/10: time - 73.37s || loss_train - 1.9979 || accuracy_train - 0.6320
...
Training took 823.35s for 10 epochs
```

**~74 seconds per epoch** after warmup.

---

## Multi-GPU Training Demo

### Dual GPU Training

Start dual GPU training (CIFAR-100 data ready from build step):

```bash
./run-ddp.sh 2 script_single_node "--model resnet101 --learning_rate 0.002 --optimizer sgd --batch_size 1024 --num_epochs 10"
```

**Performance results:**
```
Hello from local_rank 0, global_rank 0
Hello from local_rank 1, global_rank 1
Epoch - 0/10: time - 330.76s || loss_train - 4.5418 || accuracy_train - 0.0307
Accuracy on test dataset - 0.0835
Epoch - 1/10: time - 36.49s || loss_train - 4.1656 || accuracy_train - 0.1967
Epoch - 2/10: time - 36.28s || loss_train - 3.5878 || accuracy_train - 0.3856
...
Training took 431.83s for 10 epochs
```

**~36 seconds per epoch** - 2x faster.

### Performance Comparison

| GPUs | Time per Epoch | Speedup | Global Batch Size |
|------|----------------|---------|-------------------|
| **1 GPU** | 74.39s | 1x | 1,024 |
| **2 GPUs** | 36.49s | **2.04x** | 2,048 |

Perfect linear scaling.

---

## Interactive Mode (Development)

### Start Interactive Container

```bash
./run-ddp.sh 2
```

**Container setup:**
```
======================================
■ DDP Training container started successfully!
• Container name: ddp-5L2a5-2-0-1
• GPUs allocated: 2
• GPU indices: 0,1
• SSH Connection: ssh root@localhost -p 2211
======================================
```

### Connect and Explore

```bash
ssh root@localhost -p 2211
```

Password: `root`

**Inside the container:**
```bash
# See available models
python ddp-training.py --show_recommendations

# See training examples
python ddp-training.py --script_options

# Start manual training
./torchrun_script.sh ./ddp-training.py --model resnet101 --learning_rate 0.002 --optimizer sgd --batch_size 1024 --num_epochs 100 --normalize neg_one_one
```

**Live training output:**
```
Auto-detected num_workers: 16
Auto-detected backend: nccl
Training resnet101 on cifar100 dataset
Input size: 224x224, Classes: 100
Using sgd optimizer with lr=0.002, weight_decay=0.0001
Batch size: 1024 (per process)
Workers: 16, Backend: nccl
Normalization: neg_one_one

Epoch 1/100: time=342.36s || loss=4.5536 || acc=0.0320
Epoch 2/100: time=41.24s || loss=4.1759 || acc=0.2016
Epoch 3/100: time=39.61s || loss=3.5726 || acc=0.4026
Epoch 4/100: time=42.59s || loss=2.8204 || acc=0.5177
Epoch 5/100: time=42.74s || loss=2.1378 || acc=0.6044
Test accuracy: 0.6412
```

Convergence in just 5 epochs.

---

## Different Models Demo

### EfficientNet Training

```bash
./run-ddp.sh 2 script_single_node "--model efficientnet_b0 --learning_rate 0.001 --optimizer adamw --batch_size 256 --num_epochs 20"
```

### Vision Transformer

```bash
./run-ddp.sh 2 script_single_node "--model vit_base_patch16_224 --learning_rate 0.0001 --optimizer adamw --batch_size 128 --num_epochs 20"
```

### CIFAR-10 Training

```bash
./run-ddp.sh 2 script_single_node "--model resnet18 --learning_rate 0.001 --optimizer sgd --dataset cifar10 --batch_size 256 --num_epochs 20"
```

**Note:** You'll need to download CIFAR-10 separately:
```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P ./data
cd ./data && tar -xvzf cifar-10-python.tar.gz && cd ..
```

---

## Multi-Node Training

**Note:** Multi-node support is implemented in the framework but has not been fully tested with real multi-machine setups. The following commands show the intended usage pattern.

### Master Node (192.168.1.100)

```bash
./run-ddp.sh 8 auto 192.168.1.100 script_multi_node "--model resnet101 --learning_rate 0.002 --optimizer sgd --batch_size 1024 --num_epochs 50"
```

### Worker Node

```bash
./run-ddp.sh 8 auto 192.168.1.100 script_multi_node "--model resnet101 --learning_rate 0.002 --optimizer sgd --batch_size 1024 --num_epochs 50"
```

**16 GPUs total** across 2 nodes = ~6s per epoch (theoretical).

---

## Real-World Examples

### Quick Experiment

```bash
# Fast test with ResNet50
./run-ddp.sh 2 script_single_node "--model resnet50 --learning_rate 0.001 --optimizer sgd --batch_size 512 --num_epochs 10 --log_every 2"
```

### Production Training

```bash
# Full ResNet101 training
./run-ddp.sh 4 script_single_node "--model resnet101 --learning_rate 0.002 --optimizer sgd --batch_size 1024 --num_epochs 100 --normalize neg_one_one"
```

### Memory-Efficient Training

```bash
# Scaled batch size - same convergence as single GPU
./run-ddp.sh 4 script_single_node "--model resnet101 --learning_rate 0.002 --optimizer sgd --batch_size 1024 --batch_size_scaled --num_epochs 100"
```

With `--batch_size_scaled`: Each GPU gets 256 samples, global batch stays 1024.

---

## Container Management

### List Running Containers

```bash
docker ps --format "table {{.Names}}\t{{.Ports}}" | grep "ddp-"
```

### Monitor Training Progress

```bash
# Live logs
docker logs -f ddp-5L2a5-2-0-1

# Or check log files
tail -f log/2025-07-06_ddp-5L2a5-2-0-1.log
```

### Stop Training

```bash
docker stop ddp-5L2a5-2-0-1
```

Container auto-removes itself.

---

## Model Recommendations

### For Quick Testing
```bash
"--model resnet18 --learning_rate 0.001 --optimizer sgd --batch_size 256 --num_epochs 10"
```

### For Best Performance
```bash
"--model resnet101 --learning_rate 0.002 --optimizer sgd --batch_size 1024 --num_epochs 100 --normalize neg_one_one"
```

### For Large Models
```bash
"--model efficientnet_b0 --learning_rate 0.001 --optimizer adamw --batch_size 256 --weight_decay 1e-5"
```

### For Vision Transformers
```bash
"--model vit_base_patch16_224 --learning_rate 0.0001 --optimizer adamw --batch_size 128 --weight_decay 0.05"
```

---

## Common Issues & Fixes

### GPU Configuration Error
```
Error: Detected unsupported configuration
Current configuration: NPS2, CPX
```
**Fix:** Reconfigure GPUs to NPS1+SPX mode using ROCm tools.

### Port Already in Use
**Fix:** Script auto-assigns ports. If conflict occurs, wait or stop conflicting containers.

### Container Won't Start
```bash
# Check Docker image exists
docker images | grep rocm-ddp

# Rebuild if needed
./build.sh
```

### SSH Connection Failed
**Fix:** Wait seconds for SSH daemon to start, then retry connection.

### Data Directory Missing
```
Error: Training data not found in ./data directory
```
**Fix:** The build script should auto-download CIFAR-100. If it failed, manually download:
```bash
rm -rf ./data  # Clean any partial downloads
mkdir data
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -P ./data
cd data && tar -xvzf cifar-100-python.tar.gz && cd ..
```

### Multi-Node Issues
**Note:** Multi-node training functionality is implemented but not fully validated. For production workloads, stick to single-node multi-GPU setups for now.

---

## Performance Summary

### MI300X Performance (Per GPU)
- **Memory**: 192GB VRAM each
- **ResNet101**: ~37s per epoch (2 GPU mode)
- **Global Batch**: 2048 (1024 per GPU)
- **Scaling**: Linear across GPUs

### Training Time Estimates

| Model | GPUs | Batch/GPU | Time/Epoch | 100 Epochs |
|-------|------|-----------|------------|------------|
| ResNet101 | 1 | 1024 | 74s | ~2.1 hours |
| ResNet101 | 2 | 1024 | 37s | ~1.0 hour |
| ResNet101 | 4 | 1024 | 18s | ~0.5 hour |
| ResNet101 | 8 | 1024 | 9s* | ~0.25 hour* |

*Single-node performance. Multi-node scaling requires further validation.

---

## Next Steps

1. **Start with 1-2 GPUs** for testing
2. **Use interactive mode** for development
3. **Scale to 4-8 GPUs** for production
4. **Try different models** (EfficientNet, ViT)
5. **Multi-node support** is available but requires further testing