import numpy as np
from numba import cuda


@cuda.jit
def reach_limit_kernel(downstream, length, Dmax, reach_limit, distance_accum):
    i = cuda.grid(1)
    n = length.shape[0]
    if i >= n:
        return

    dist = 0.0
    # link immediately upstream of outlet
    if downstream[i] == -1:
        reach_limit[i] = downstream[i]
        distance_accum[i] = length[i]
        return

    j = downstream[i]
    prev_j = j
    while j != -1 and j < n:
        L = length[j]
        if dist + L > Dmax:
            break
        dist += L
        if downstream[j] == -1:
            prev_j = j
        j = downstream[j]

    reach_limit[i] = prev_j
    distance_accum[i] = dist


def create_upstream_lists_from_reach_limit(reach_limit, n):
    """
    Build flattened upstream adjacency lists from `reach_limit`.
    Returns (flat_upstream, offsets, counts) where for node i the
    upstream indices live at flat_upstream[offsets[i]:offsets[i]+counts[i]].
    """
    counts = np.zeros(n, dtype=np.int32)
    for u in range(n):
        d = int(reach_limit[u])
        if d != -1:
            counts[d] += 1

    offsets = np.empty(n, dtype=np.int32)
    if n > 0:
        offsets[0] = 0
        for i in range(1, n):
            offsets[i] = offsets[i - 1] + counts[i - 1]

    total = int(offsets[-1] + counts[-1]) if n > 0 else 0
    flat = np.empty(total, dtype=np.int32)
    pos = np.zeros(n, dtype=np.int32)
    for u in range(n):
        d = int(reach_limit[u])
        if d != -1:
            idx = offsets[d] + pos[d]
            flat[idx] = u
            pos[d] += 1

    return flat, offsets, counts


@cuda.jit
def routing_step_kernel(curr_state, inflow, k_arr, upstream_flat, upstream_offsets, upstream_counts, next_state, outflow, t_index):
    i = cuda.grid(1)
    n = curr_state.shape[0]
    if i >= n:
        return

    q_i = curr_state[i] * k_arr[i]

    s = 0.0
    off = upstream_offsets[i]
    cnt = upstream_counts[i]
    for j in range(cnt):
        u = upstream_flat[off + j]
        s += curr_state[u] * k_arr[u]

    next_state[i] = curr_state[i] + s - q_i + inflow[i, t_index]
    outflow[i, t_index] = q_i


def routing5_cuda(current_state, inflow, velocity, channel_length, reach_limit, DT=3600, k=None):
    """
    CUDA-accelerated routing (Numba) that does not require cupy.

    Parameters:
    - current_state: numpy array (N,) initial storage states
    - inflow: numpy array (N, T) external inflows per time
    - velocity: scalar or array-like used to compute k if None
    - channel_length: numpy array (N,) used to compute k if None
    - reach_limit: numpy int array (N,) where reach_limit[u] gives downstream node index or -1
    - DT: timestep seconds
    - k: optional scalar or array of linear reservoir coefficients (0..1). If None it is computed.

    Returns (outflow, current_state)
    """
    N = inflow.shape[0]
    T = inflow.shape[1]

    # compute k array
    if k is None:
        k_arr = np.exp(-1 * velocity / channel_length * DT)
        k_arr = 1.0 - k_arr
    else:
        if np.isscalar(k):
            kk = float(k)
            kk = max(0.0, min(1.0, kk))
            k_arr = np.full(N, kk, dtype=np.float32)
        else:
            k_arr = np.array(k, dtype=np.float32)

    k_arr = k_arr.astype(np.float32)

    if current_state is None:
        curr = np.zeros(N, dtype=np.float32)
    else:
        curr = np.array(current_state, dtype=np.float32)

    inflow = np.array(inflow, dtype=np.float32)

    upstream_flat, upstream_offsets, upstream_counts = create_upstream_lists_from_reach_limit(reach_limit, N)

    # device arrays
    curr_d = cuda.to_device(curr)
    next_d = cuda.device_array_like(curr_d)
    k_d = cuda.to_device(k_arr)
    upstream_flat_d = cuda.to_device(upstream_flat)
    upstream_offsets_d = cuda.to_device(upstream_offsets)
    upstream_counts_d = cuda.to_device(upstream_counts)
    inflow_d = cuda.to_device(inflow)

    outflow_d = cuda.device_array((N, T), dtype=np.float32)

    threads_per_block = 128
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    for t in range(T):
        routing_step_kernel[blocks_per_grid, threads_per_block](
            curr_d,
            inflow_d,
            k_d,
            upstream_flat_d,
            upstream_offsets_d,
            upstream_counts_d,
            next_d,
            outflow_d,
            t,
        )

        tmp = curr_d
        curr_d = next_d
        next_d = tmp

    outflow = outflow_d.copy_to_host()
    final_state = curr_d.copy_to_host()
    return outflow, final_state


if __name__ == '__main__':
    # Minimal smoke example if run standalone
    import pandas as pd
    df = pd.read_csv('southfork_rush_tiles.csv')
    idx_upstream_link = df[['up1','up2','up3','up4']].to_numpy(dtype=np.int32)

    # build downstream mapping (same logic as original main)
    n = len(df)
    nup = df['nup'].to_numpy()
    idx = df['idx'].to_numpy()
    downstream = np.ones(shape=(n), dtype=np.int32) * (-1)
    for i in range(len(df)):
        if nup[i] > 0:
            for j in range(nup[i]):
                val = int(idx_upstream_link[i, j])
                if val > 0:
                    upstream = idx[val]
                    downstream[upstream] = i

    inflow = np.ones(shape=(n, 10), dtype=np.float32)
    current_state = np.zeros(shape=(n,), dtype=np.float32)
    velocity = 10.0
    channel_length = df['channel_length'].to_numpy(dtype=np.float32)
    dt = 3600.0
    Dmax = np.float32(velocity * dt)

    d_reach_limit = cuda.device_array(n, dtype=np.int32)
    d_distance_accum = cuda.device_array(n, dtype=np.float32)
    length_d = cuda.to_device(channel_length)

    threads_per_block = 128
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    reach_limit_kernel[blocks_per_grid, threads_per_block](
        cuda.to_device(downstream), length_d, Dmax, d_reach_limit, d_distance_accum
    )

    reach_limit = d_reach_limit.copy_to_host()
    outflow, final_state = routing5_cuda(current_state, inflow, velocity, channel_length, reach_limit, DT=dt)
    print('routing5_cuda smoke run complete')
