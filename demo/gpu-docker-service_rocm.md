# ROCm MI300X Demo Guide

## Getting Started

### Check Your AMD GPUs

First, make sure your MI300X GPUs are working:

```bash
rocm-smi
```

You should see something like this:
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

2 × MI300X GPUs ready.

### Build ROCm Lab Image

```bash
cd gpu-lab-builder/
./create_image.sh
```

When asked, type `y`:
```
Confirm building the above images? (y/N): y
```

Success looks like:
```
■ ROCm image built successfully: rocm-lab
```

### Start the API Service (Custom Ports)

```bash
# Create backup folder
sudo mkdir -p /backups && sudo chmod -R 777 /backups

# Run setup with custom ports
sudo SSH_PORT_BASE=3300 WEB_PORT=9000 ./setup_env.sh
```

Success message:
```
[INFO] Setup completed successfully!
```

---

## Basic Usage

### Check System Status

```bash
curl http://localhost:9000/api/v1/gpu-docker/status
```

**Response:**
```json
{
  "success": true,
  "message": "GPU status retrieved successfully",
  "data": {
    "gpu_type": "rocm",
    "available_gpu_indices": [0, 1],
    "used_gpu_indices": [],
    "container_gpu_mapping": {},
    "last_updated": "2025-07-05T07:30:47+00:00"
  }
}
```

Both GPUs are available.

### Create Your First Container (1 GPU)

```bash
curl -X POST http://localhost:9000/api/v1/gpu-docker/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "experiment-01",
    "image": "rocm-lab:latest",
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
    "name": "experiment-01",
    "ssh_port": 3311,
    "gpu_devices": [0],
    "port_mappings": ["3002:3002", "4002:4002"]
  }
}
```

### Create Second Container (Different Ports)

```bash
curl -X POST http://localhost:9000/api/v1/gpu-docker/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "experiment-02",
    "image": "rocm-lab:latest",
    "port": ["5002:5002", "6002:6002"],
    "gpu_count": 1,
    "cpus": "4",
    "memory": "16g"
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Container created successfully",
  "data": {
    "name": "experiment-02",
    "ssh_port": 3322,
    "gpu_devices": [1],
    "port_mappings": ["5002:5002", "6002:6002"]
  }
}
```

### Connect and Test GPU Performance

```bash
ssh root@localhost -p 3322
```

Password: `root`

```bash
python3 /app/smart_gpu_test.py
```

**Expected result:**
```
=== GPU Stress Testing Tool ===
HIP_VISIBLE_DEVICES: 1
Selected GPU 0: AMD Instinct MI300X
  Total memory: 191.98 GB
  Multi-processor count: 304
  Compute capability: 9.4

=== Test Completed ===
Total time: 1.04 seconds
Successful iterations: 5000/5000
Success rate: 100.0%
Average performance: 82,514.38 GFLOPS
✓ All tests completed!
```

82K GFLOPS and 192GB memory.

---

## Container Management

### List All Containers

```bash
curl http://localhost:9000/api/v1/gpu-docker/list
```

See both containers running with their GPU assignments.

### Upgrade Container Resources

```bash
curl -X POST http://localhost:9000/api/v1/gpu-docker/update-resources \
  -H "Content-Type: application/json" \
  -d '{
    "name": "experiment-01",
    "cpus": "8"
  }'
```

### Stop/Start/Restart Containers

```bash
# Stop
curl -X POST http://localhost:9000/api/v1/gpu-docker/stop \
  -H "Content-Type: application/json" \
  -d '{"name": "experiment-02"}'

# Start
curl -X POST http://localhost:9000/api/v1/gpu-docker/start \
  -H "Content-Type: application/json" \
  -d '{"name": "experiment-02"}'

# Restart
curl -X POST http://localhost:9000/api/v1/gpu-docker/restart \
  -H "Content-Type: application/json" \
  -d '{"name": "experiment-02"}'
```

---

## Snapshots (Quick Saves)

### Create Snapshots for Both Containers

```bash
# Save experiment-01 state
curl -X POST http://localhost:9000/api/v1/gpu-docker/snapshot \
  -H "Content-Type: application/json" \
  -d '{
    "name": "experiment-01",
    "snapshot_name": "exp01-checkpoint"
  }'

# Save experiment-02 state  
curl -X POST http://localhost:9000/api/v1/gpu-docker/snapshot \
  -H "Content-Type: application/json" \
  -d '{
    "name": "experiment-02",
    "snapshot_name": "exp02-checkpoint"
  }'
```

### List Your Snapshots

```bash
curl http://localhost:9000/api/v1/gpu-docker/snapshots
```

### Restore from Snapshot

First delete a container to free up GPU:
```bash
curl -X POST http://localhost:9000/api/v1/gpu-docker/delete \
  -H "Content-Type: application/json" \
  -d '{"name": "experiment-01"}'
```

Then restore:
```bash
curl -X POST http://localhost:9000/api/v1/gpu-docker/restore-snapshot \
  -H "Content-Type: application/json" \
  -d '{
    "snapshot_name": "exp01-checkpoint",
    "name": "experiment-01-restored",
    "gpu_count": 1,
    "port": ["3003:3003"]
  }'
```

---

## Backups (For Moving Between Machines)

### Create a Backup

```bash
curl -X POST http://localhost:9000/api/v1/gpu-docker/backup \
  -H "Content-Type: application/json" \
  -d '{
    "name": "experiment-02",
    "backup_name": "exp02-backup"
  }'
```

This creates a ~55GB tar file you can move to other machines.

### Restore from Backup (All GPUs Available)

```bash
# Delete all containers first
curl -X POST http://localhost:9000/api/v1/gpu-docker/delete \
  -H "Content-Type: application/json" \
  -d '{"name": "experiment-02"}'

# Restore with BOTH GPUs!
curl -X POST http://localhost:9000/api/v1/gpu-docker/restore \
  -H "Content-Type: application/json" \
  -d '{
    "backup_path": "exp02-backup.tar",
    "name": "dual-gpu-container",
    "gpu_count": 2,
    "port": ["3004:3004"]
  }'
```

Now you have 384GB total VRAM.

---

## Real-World Examples

### Quick Development Setup

```bash
curl -X POST http://localhost:9000/api/v1/gpu-docker/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "dev-workspace",
    "image": "rocm-lab:latest",
    "port": ["8888:8888"],
    "gpu_count": 1,
    "cpus": "4",
    "memory": "16g"
  }'
```

Connect and install your tools:
```bash
ssh root@localhost -p [assigned_port]
# Install Jupyter, your frameworks, etc.
```

### Experiment Version Control

```bash
# Save after each major milestone
curl -X POST http://localhost:9000/api/v1/gpu-docker/snapshot \
  -H "Content-Type: application/json" \
  -d '{
    "name": "dev-workspace",
    "snapshot_name": "model-v1-trained"
  }'
```

### Cross-Machine Deployment

```bash
# On development machine
curl -X POST http://localhost:9000/api/v1/gpu-docker/backup \
  -H "Content-Type: application/json" \
  -d '{
    "name": "production-model",
    "backup_name": "prod-ready-v1"
  }'

# Copy file to production server
scp /backups/backup_prod-ready-v1.tar user@prod-server:/backups/

# On production server
curl -X POST http://prod-server:9000/api/v1/gpu-docker/restore \
  -H "Content-Type: application/json" \
  -d '{
    "backup_path": "prod-ready-v1.tar",
    "name": "production-model",
    "gpu_count": 2
  }'
```

---

## Common Issues

### Port Conflicts
```json
{
  "success": false,
  "message": "Port(s) already in use: [3002, 4002]",
  "error_code": "PORT_CONFLICT"
}
```
**Fix**: Use different ports like `["5002:5002"]`

### No GPU Available
```json
{
  "success": false,
  "message": "GPU allocation failed",
  "error_code": "GPU_ALLOCATION_FAILED"
}
```
**Fix**: Check `curl http://localhost:9000/api/v1/gpu-docker/status` and delete unused containers

---

## Tips

- **SSH Ports**: Start from 3311, 3322, etc.
- **Memory**: Each GPU has 192GB
- **Performance**: Perfect for large language models
- **Snapshots**: ~56GB each, quick to create/restore
- **Backups**: ~55GB each, good for archiving
- **Scaling**: Use both GPUs for 384GB total memory