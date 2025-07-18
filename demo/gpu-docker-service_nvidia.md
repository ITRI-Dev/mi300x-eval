# NVIDIA GPU Demo Guide

## Getting Started

### Check Your System

First, make sure your NVIDIA GPU is working:

```bash
nvidia-smi
```

You should see something like this:
```
Sun Jul  6 10:58:10 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTXA6000-24Q            On  |   00000000:06:10.0 Off |                    0 |
| N/A   N/A    P8             N/A /  N/A  |     814MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

### Build GPU Lab Image

```bash
cd gpu-lab-builder/
./create_image.sh
```

When asked, type `y` to confirm:
```
Confirm building the above images? (y/N): y
```

Success looks like:
```
■ NVIDIA image built successfully: nvidia-lab
```

### Start the API Service

```bash
# Create backup folder
sudo mkdir -p /backups && sudo chmod -R 777 /backups

# Run setup
sudo ./setup_env.sh
```

Success message:
```
[INFO] Setup completed successfully!
```

---

## Basic Usage

### Check System Status

```bash
curl http://localhost:8000/api/v1/gpu-docker/status
```

**Response:**
```json
{
  "success": true,
  "message": "GPU status retrieved successfully",
  "data": {
    "gpu_type": "nvidia",
    "available_gpu_indices": [0],
    "used_gpu_indices": [],
    "container_gpu_mapping": {},
    "last_updated": "2025-07-05T15:01:03+08:00"
  }
}
```

### Create Your First GPU Container

```bash
curl -X POST http://localhost:8000/api/v1/gpu-docker/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-gpu-container",
    "image": "nvidia-lab:latest",
    "port": ["3002:3002", "4002:4002"],
    "gpu_count": 1,
    "cpus": "2",
    "memory": "8g"
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Container created successfully",
  "data": {
    "name": "my-gpu-container",
    "ssh_port": 2211,
    "gpu_devices": [0],
    "port_mappings": ["3002:3002", "4002:4002"]
  }
}
```

### Connect to Your Container

```bash
ssh root@localhost -p 2211
```

Password: `root`

Once inside, test the GPU:
```bash
python3 /app/smart_gpu_test.py
```

**Expected result:**
```
=== GPU Stress Testing Tool ===
GPU 0: NVIDIA RTXA6000-24Q
Total time: 12.47 seconds
Successful iterations: 5000/5000
Success rate: 100.0%
Average performance: 6887.70 GFLOPS  
✓ All tests completed!
```

---

## Container Management

### List All Containers

```bash
curl http://localhost:8000/api/v1/gpu-docker/list
```

### Stop a Container

```bash
curl -X POST http://localhost:8000/api/v1/gpu-docker/stop \
  -H "Content-Type: application/json" \
  -d '{"name": "my-gpu-container"}'
```

### Start a Container

```bash
curl -X POST http://localhost:8000/api/v1/gpu-docker/start \
  -H "Content-Type: application/json" \
  -d '{"name": "my-gpu-container"}'
```

### Delete a Container

```bash
curl -X POST http://localhost:8000/api/v1/gpu-docker/delete \
  -H "Content-Type: application/json" \
  -d '{"name": "my-gpu-container"}'
```

---

## Snapshots (Save Container States)

### Create a Snapshot

```bash
curl -X POST http://localhost:8000/api/v1/gpu-docker/snapshot \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-gpu-container",
    "snapshot_name": "my-experiment-v1"
  }'
```

### List Snapshots

```bash
curl http://localhost:8000/api/v1/gpu-docker/snapshots
```

### Restore from Snapshot

```bash
curl -X POST http://localhost:8000/api/v1/gpu-docker/restore-snapshot \
  -H "Content-Type: application/json" \
  -d '{
    "snapshot_name": "my-experiment-v1",
    "name": "restored-container",
    "gpu_count": 1,
    "port": ["3003:3003"]
  }'
```

---

## Backups (For Moving Between Machines)

### Create a Backup

```bash
curl -X POST http://localhost:8000/api/v1/gpu-docker/backup \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-gpu-container",
    "backup_name": "my-backup-v1"
  }'
```

### Restore from Backup

```bash
curl -X POST http://localhost:8000/api/v1/gpu-docker/restore \
  -H "Content-Type: application/json" \
  -d '{
    "backup_path": "my-backup-v1.tar",
    "name": "new-container",
    "gpu_count": 1,
    "port": ["3004:3004"]
  }'
```

---

## Common Issues

### Port Already in Use
If you get a port conflict error, just use different ports:
```bash
"port": ["5002:5002", "6002:6002"]
```

### GPU Not Available
If GPU allocation fails, check what's using it:
```bash
curl http://localhost:8000/api/v1/gpu-docker/list
```

Then delete unused containers to free up GPUs.

### Container Won't Start
Try to restart it:
```bash
curl -X POST http://localhost:8000/api/v1/gpu-docker/restart \
  -H "Content-Type: application/json" \
  -d '{"name": "container-name"}'
```

---

## Tips

- **SSH Port**: Automatically assigned starting from 2211
- **Default Password**: `root` for all containers  
- **GPU Memory**: This setup uses 24GB VRAM
- **Performance**: About 6.9K GFLOPS on A6000
- **Snapshots**: Good for quick saves during experiments
- **Backups**: Good for moving containers between different machines