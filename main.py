import cupy as cp
import numpy as np
from numba import cuda, int32, float32
import cupyx as cpx

@cuda.jit
def reach_limit_kernel(downstream, length, Dmax, reach_limit, distance_accum):
    """
    GPU kernel (Numba CUDA version) to find how far water can travel in one hour
    along a river network given downstream links.
    """
    i = cuda.grid(1)
    n = length.shape[0]
    if i >= n:
        return

    dist = 0.0
    #j = i
    #analysis for link immediately upstream of outlet
    if downstream[i]==-1:
        reach_limit[i]=downstream[i]
        distance_accum[i] = length[i]
    #analysis for any other link
    j = downstream[i]
    prev_j = j
    while j != -1 and j < n:
        L = length[j]
        if dist + L > Dmax:
            break
        dist += L
        #if i==0:
        #    print(dist)
        if downstream[j]==-1:
            prev_j=j
        j = downstream[j]
        

    #reach_limit[i] = j
    reach_limit[i] = prev_j
    distance_accum[i] = dist

def create_sparse_matrix_upstream_velocity(reach_limit,n):
    col = [] #upstream
    row = [] #downstream
    data = [] #ones
    
    for i in range(0,n):
        downstream = reach_limit[i]
        if downstream ==-1:
            continue
        upstream = i
        value = 1
        col.append(upstream)
        row.append(downstream)
        data.append(value)
    cpdata = cp.array(data,dtype=cp.float32)
    cprow = cp.array(row,dtype=cp.int32)
    cpcol = cp.array(col,dtype=cp.int32)
    sparse_upstream = cpx.scipy.sparse.csc_matrix((cpdata, (cprow,cpcol )), shape=(n, n))
    return sparse_upstream

def routing5(current_state,inflow,velocity,channel_length,sparse_upstream,DT=3600,k=None):
    N = inflow.shape[0] #number of channels
    T = inflow.shape[1] #number of time steps
    # k is a linear reservoir coefficient, between 0 and 1. If None, it is calculated based on velocity and channel length.
    if k is None:
        k = np.exp(-1*velocity / channel_length*DT)
        k = 1-k
        k = cp.asarray(k)
    else:
        if k>1:
            k=1
        if k<0:
            k=0
    #initial condition
    if current_state is None:
        current_state = cp.array(np.zeros(shape=(N),dtype=cp.float32))

    outflow = cp.array(np.zeros(shape=(N,T),dtype=cp.float32))
    for t in range(0,T):
        
        inflow_at_t = inflow[:,t]
        #q_t = current_state * (1-coef) # flow that leaves the channel, m3/s
        q_t = current_state * k
        #print('suma q_t')
        #print(np.sum(q_t))
        inflow_upstream = sparse_upstream * q_t #sum of upstream flows to each channel , m3/s
        #print('suma upstream')
        #print(np.sum(inflow_upstream))
        current_state = current_state +inflow_upstream - q_t
        #np.sum(current_state)
        outflow[:,t] = q_t
        #outflow[:,t] = current_state.copy()
        current_state = cp.asarray(current_state) + cp.asarray(inflow_at_t)

    return(outflow,current_state)

def test_reach_limit():
    # Example network
    n = 8
    length = np.array([500, 600, 400, 1000, 700, 900, 800, 1200], dtype=np.float32)
    downstream = np.array([1, 2, 3, 4, 5, 6, 7, -1], dtype=np.int32)

    velocity = 0.3   # m/s
    dt = 3600.0
    Dmax = np.float32(velocity * dt)

    # Allocate GPU arrays
    d_length = cuda.to_device(length)
    d_downstream = cuda.to_device(downstream)
    d_reach_limit = cuda.device_array(n, dtype=np.int32)
    d_distance_accum = cuda.device_array(n, dtype=np.float32)

    # Launch kernel
    threads_per_block = 128
    blocks = (n + threads_per_block - 1) // threads_per_block

    reach_limit_kernel[blocks, threads_per_block](
        d_downstream, d_length, Dmax, d_reach_limit, d_distance_accum
    )

    # Copy results back
    reach_limit = d_reach_limit.copy_to_host()
    distance_accum = d_distance_accum.copy_to_host()

    print(f"Dmax = {Dmax:.1f} m")
    print("Reach limit IDs:", reach_limit)
    print("Accumulated distance:", distance_accum)


def test_routing5():
    """Run a simple routing5 scenario and verify expected outflow and state."""
    current_state = None
    inflow = cp.array([[1.0, 0.0], [1.0, 0.0]], dtype=cp.float32)
    channel_length = cp.array([100.0, 100.0], dtype=cp.float32)
    sparse_upstream = cpx.scipy.sparse.csc_matrix(
        (
            cp.array([1.0], dtype=cp.float32),
            (cp.array([1], dtype=cp.int32), cp.array([0], dtype=cp.int32)),
        ),
        shape=(2, 2),
    )

    outflow, next_state = routing5(
        current_state,
        inflow,
        velocity=None,
        channel_length=channel_length,
        sparse_upstream=sparse_upstream,
        DT=3600,
        k=0.5,
    )

    expected_outflow = cp.array([[0.0, 0.5], [0.0, 0.5]], dtype=cp.float32)
    expected_state = cp.array([0.5, 1.0], dtype=cp.float32)

    cp.testing.assert_allclose(outflow, expected_outflow, rtol=1e-6, atol=1e-6)
    cp.testing.assert_allclose(next_state, expected_state, rtol=1e-6, atol=1e-6)

    print("routing5 test passed.")


if __name__ == '__main__':
    test_routing5()
    test_reach_limit()
    
