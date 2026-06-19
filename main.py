import cupy as cp
import numpy as np
from numba import cuda, int32, float32
import cupyx as cpx
import pandas as pd

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

def _get_downstream(df,idx_upstream_links):
    
    n = len(df)
    nup = df['nup'].to_numpy()
    idx = df['idx'].to_numpy()
    downstream = np.ones(shape=(n),dtype=np.int32)*(-1)
    for i in range(len(df)):
        if nup[i]>0:
            for j in range(nup[i]):
                val = int(idx_upstream_links[i,j])
                if val > 0:
                    try:
                        upstream = idx[val]
                    except IndexError as e:
                        print('row ',i)
                        print('val ',val)
                        print(e)
                    downstream[upstream] = i
    return downstream


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



if __name__ == '__main__':
    df = pd.read_csv('southfork_rush_tiles.csv')
    idx_upstream_link = df[['up1','up2','up3','up4']].to_numpy(dtype=np.int32)
    threads_per_block = 128
    blocks_per_grid = (len(df) + threads_per_block - 1) // threads_per_block
    downstream = _get_downstream(df,idx_upstream_link)
    inflow = np.ones(shape=(len(df),10),dtype=np.float32) #example inflow for 10 time steps
    current_state = cp.asarray(np.zeros(shape=(len(df)),dtype=np.float32)) #initial state of the system
    velocity = np.float32(10.0) #m/s
    channel_length = df['channel_length'].to_numpy(dtype=np.float32)
    dt = 3600.0
    Dmax = np.float32(velocity * dt)
    d_reach_limit = cuda.device_array(len(df), dtype=np.int32)
    d_distance_accum = cuda.device_array(len(df), dtype=np.float32)

    reach_limit_kernel[blocks_per_grid, threads_per_block](
        cuda.to_device(downstream),
        channel_length,
        Dmax,
        d_reach_limit,
        d_distance_accum
        )
    reach_limit = d_reach_limit.copy_to_host()
    smuv = create_sparse_matrix_upstream_velocity(reach_limit,len(df))
    x,x2 = routing5(current_state,inflow,velocity,channel_length,smuv,DT=3600)
    print('routing5 executed successfully')


