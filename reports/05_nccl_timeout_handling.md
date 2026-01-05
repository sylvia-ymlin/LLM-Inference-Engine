# NCCL Timeout Synchronization: Problem and Solution

## The Problem

When running distributed training/inference with NCCL, a common failure mode occurs:

**Symptom**: The job hangs indefinitely during an `AllReduce` or other collective operation.

**Root Cause**:
1. One GPU crashes, OOMs, or becomes unresponsive
2. Network partition between nodes occurs
3. Asymmetric workload causes one rank to fall behind

**Default NCCL Behavior**:
- NCCL uses **blocking synchronization** by default
- Timeout is set to **infinity** by default
- No async error reporting to detect stalled ranks

This means a single GPU failure can freeze an entire multi-GPU job without any error message.

## Before: Naive Implementation

```cpp
// DANGEROUS: No timeout, no error handling
Status NcclHandler::Init(int32_t rank, int32_t world_size, ...) {
    ncclGetUniqueId(&id_);
    ncclCommInitRank(&comm_, world_size_, id_, rank_);  // Can hang forever
    return error::Success();
}
```

**Failure Scenario**:
```
Rank 0: ncclAllReduce(...) → Waiting for Rank 1
Rank 1: GPU OOM → Process dies
Rank 0: ... waiting ... waiting ... (forever)
Cluster: No error message, job scheduler thinks everything is fine
```

## After: Timeout-Protected Implementation

```cpp
Status NcclHandler::Init(int32_t rank, int32_t world_size, ...) {
    // Enable async error handling - allows NCCL to report errors without blocking
    setenv("NCCL_ASYNC_ERROR_HANDLING", "1", 0);
    
    // Disable blocking waits - required for timeout to work
    setenv("NCCL_BLOCKING_WAIT", "0", 0);
    
    // Set explicit timeout (30 minutes = 1800 seconds)
    setenv("NCCL_TIMEOUT", "1800", 0);
    
    ncclGetUniqueId(&id_);
    ncclResult_t result = ncclCommInitRank(&comm_, world_size_, id_, rank_);
    
    // Check for async errors after init
    ncclResult_t state;
    ncclCommGetAsyncError(comm_, &state);
    if (state != ncclSuccess) {
        return error::InternalError("NCCL async error: " + ncclGetErrorString(state));
    }
    return error::Success();
}
```

**Failure Scenario (Fixed)**:
```
Rank 0: ncclAllReduce(...) → Waiting for Rank 1
Rank 1: GPU OOM → Process dies
Rank 0: ... waiting ... (timeout after 30 min)
Rank 0: ERROR - NCCL timeout, rank 1 unresponsive
Cluster: Job fails cleanly, resources released, error logged
```

## Environment Variables Explained

| Variable | Value | Purpose |
|----------|-------|---------|
| `NCCL_ASYNC_ERROR_HANDLING` | `1` | Report errors asynchronously instead of hanging |
| `NCCL_BLOCKING_WAIT` | `0` | Allow timeout detection (blocking=1 ignores timeout) |
| `NCCL_TIMEOUT` | `1800` | Seconds before declaring operation failed |
| `NCCL_DEBUG` | `INFO` | (Optional) Enable verbose logging for debugging |

## Impact

| Metric | Before | After |
|--------|--------|-------|
| Failure Detection | Never (infinite hang) | 30 minutes |
| Error Message | None | Clear NCCL error code |
| Resource Release | Manual intervention | Automatic |
| Debugging | Very difficult | Logs available |

## Implementation Location

The timeout configuration is applied in:
- File: `kuiper/source/base/nccl_handler.cpp`
- Function: `NcclHandler::Init()`

This ensures all NCCL operations benefit from timeout protection, including:
- `AllReduce` in MatmulLayer
- `Broadcast` for weight distribution
- `AllGather` for activation collection
