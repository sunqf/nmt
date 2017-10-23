import math
import torch
from torch.autograd import Variable
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple

###
# copy from https://github.com/salesforce/pytorch-qrnn/blob/master/torchqrnn/forget_mult.py
kernel = '''
extern "C"
__global__ void lr_recurrent_multi(float *dst, const float *f, const float *x, int SEQ, int BATCH, int HIDDEN)
{
  /*
  Note: destination is assumed to be one timestep longer than f or x where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  if(hid >= HIDDEN || bid >= BATCH)
     return;
  //
  for (int ts = 0 + 1; ts < SEQ + 1; ts++) {
     // Good sanity check for debugging - only perform additions to a zeroed chunk of memory
     // Addition seems atomic or near atomic - you should get incorrect answers if doubling up via threads
     // Note: the index i needs to be offset by one as f[0] (f_t) is used for dst[1] (h_t) etc
     // To move timesteps, we step HIDDEN * BATCH
     // To move batches, we move HIDDEN
     // To move neurons, we move +- 1
     // Note: dst[dst_i] = ts * 100 + bid * 10 + hid; is useful for debugging
     int i           = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_i       = (ts - 0) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_iminus1 = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     dst[dst_i]      = f[i] * x[i];
     dst[dst_i]      += (1 - f[i]) * dst[dst_iminus1];
  }
}

extern "C"
__global__ void lr_bwd_recurrent_forget_mult(const float *h, const float *f, const float *x, const float *gh, float *gf, float *gx, float *ghinit, int SEQ, int BATCH, int HIDDEN)
{
  /*
  Note: h is assumed to be one timestep longer than f, x, gf, gx, or gh where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  if(hid >= HIDDEN || bid >= BATCH)
     return;
  //
  double running_f = 0;
  for (int ts = SEQ - 1 + 1; ts >= 0 + 1; ts--) {
     int i           = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_i       = (ts - 0) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_iminus1 = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     //
     running_f       += gh[dst_iminus1];
     // Gradient of X
     gx[i]           = f[i] * running_f;
     // Gradient of F
     gf[i]           = (x[i] - h[dst_iminus1]) * running_f;
     //
     // The line below is likely more numerically stable than (1 - f[i]) * running_f;
     running_f       = running_f - f[i] * running_f;
  }
  ghinit[bid * HIDDEN + hid] = running_f;
}

extern "C"
__global__ void rl_recurrent_multi(float *dst, const float *f, const float *x, int SEQ, int BATCH, int HIDDEN)
{
  /*
  Note: destination is assumed to be one timestep longer than f or x where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  if(hid >= HIDDEN || bid >= BATCH)
     return;
  //
  for (int ts = SEQ - 1; ts >= 0; ts--) {
     // Good sanity check for debugging - only perform additions to a zeroed chunk of memory
     // Addition seems atomic or near atomic - you should get incorrect answers if doubling up via threads
     // Note: the index i needs to be offset by one as f[0] (f_t) is used for dst[1] (h_t) etc
     // To move timesteps, we step HIDDEN * BATCH
     // To move batches, we move HIDDEN
     // To move neurons, we move +- 1
     // Note: dst[dst_i] = ts * 100 + bid * 10 + hid; is useful for debugging
     int i           = (ts + 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_i       = (ts + 0) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_iadd1 = (ts + 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     dst[dst_i]      = f[i] * x[i];
     dst[dst_i]      += (1 - f[i]) * dst[dst_iadd1];
  }
}

extern "C"
__global__ void rl_bwd_recurrent_forget_mult(const float *h, const float *f, const float *x, const float *gh, float *gf, float *gx, float *ghinit, int SEQ, int BATCH, int HIDDEN)
{
  /*
  Note: h is assumed to be one timestep longer than f, x, gf, gx, or gh where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  if(hid >= HIDDEN || bid >= BATCH)
     return;
  //
  double running_f = 0;
  for (int ts = 0; ts <= SEQ; ts++) {
     int i           = (ts + 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_i       = (ts + 0) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_iminus1 = (ts + 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     //
     running_f       += gh[dst_iminus1];
     // Gradient of X
     gx[i]           = f[i] * running_f;
     // Gradient of F
     gf[i]           = (x[i] - h[dst_iminus1]) * running_f;
     //
     // The line below is likely more numerically stable than (1 - f[i]) * running_f;
     running_f       = running_f - f[i] * running_f;
  }
  ghinit[bid * HIDDEN + hid] = running_f;
}

'''


###

class CPUForgetMult(torch.nn.Module):
    def __init__(self, direction=0):
        super(CPUForgetMult, self).__init__()
        self.direction = direction

    def forward(self, f, x, hidden_init=None):
        result = []
        ###
        if self.direction == 0:
            forgets = f.split(1, dim=0)
            prev_h = hidden_init
            for i, h in enumerate((f * x).split(1, dim=0)):
                if prev_h is not None: h = h + (1 - forgets[i]) * prev_h
                result.append(h.squeeze())
                prev_h = h
        elif self.direction == 1:
            forgets = f.split(1, dim=0)
            prev_h = hidden_init
            for i,h in enumerate(reversed((f * x).split(1, dim=0))):
                if prev_h is not None: h = h + (1 - forgets[i]) * prev_h
                result.append(h.squeeze())
                prev_h = h
        ###
        return torch.stack(result)


class GPUForgetMult(torch.autograd.Function):
    lr_forget_mult = None
    lr_bwd_forget_mult = None
    rl_forget_mult = None
    rl_bwd_forget_mult = None
    stream = None

    def __init__(self, direction=0):
        super(GPUForgetMult, self).__init__()
        if not self.lr_forget_mult or not self.lr_bwd_forget_mult:
            GPUForgetMult.compile()

        self.direction = direction

    @staticmethod
    def compile():
        program = Program(kernel.encode(), 'recurrent_forget_mult.cu'.encode())
        ptx = program.compile()

        m = function.Module()
        m.load(bytes(ptx.encode()))

        GPUForgetMult.lr_forget_mult = m.get_function('lr_recurrent_forget_mult')
        GPUForgetMult.lr_bwd_forget_mult = m.get_function('lr_bwd_recurrent_forget_mult')


        GPUForgetMult.rl_forget_mult = m.get_function('rl_recurrent_forget_mult')
        GPUForgetMult.rl_bwd_forget_mult = m.get_function('rl_bwd_recurrent_forget_mult')

        Stream = namedtuple('Stream', ['ptr'])
        GPUForgetMult.stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)

    def forward(self, f, x, hidden_init=None):
        seq_size, batch_size, hidden_size = f.size()
        result = f.new(seq_size + 1, batch_size, hidden_size)
        # We only zero the result array (result[0]) if we don't set a hidden initial state
        # All other values (result[1:]) are overwritten by default

        ###
        grid_hidden_size = min(hidden_size, 512)
        grid = (math.ceil(hidden_size / grid_hidden_size), batch_size)

        if self.direction == 0:
            if hidden_init is not None:
                result[0, :, :] = hidden_init
            else:
                result = result.zero_()
            self.lr_forget_mult(grid=grid, block=(grid_hidden_size, 1),
                                args=[result.data_ptr(), f.data_ptr(), x.data_ptr(), seq_size, batch_size, hidden_size],
                                stream=self.stream)
            self.save_for_backward(f, x, hidden_init)
            self.result = result
            return result[1:, :, :]
        elif self.direction == 1:
            if hidden_init is not None:
                result[-1, :, :] = hidden_init
            else:
                result = result.zero_()

            self.rl_forget_mult(grid=grid, block=(grid_hidden_size, 1),
                                args=[result.data_ptr(), f.data_ptr(), x.data_ptr(), seq_size, batch_size, hidden_size],
                                stream=self.stream)
            self.save_for_backward(f, x, hidden_init)
            self.result = result
            return result[:-1, :, :]

    def backward(self, grad_h):
        f, x, hidden_init = self.saved_tensors
        h = self.result
        ###
        seq_size, batch_size, hidden_size = f.size()
        # Zeroing is not necessary as these will be overwritten
        grad_f = f.new(*f.size())
        grad_x = f.new(*f.size())
        grad_h_init = f.new(batch_size, hidden_size)
        ###
        grid_hidden_size = min(hidden_size, 512)
        grid = (math.ceil(hidden_size / grid_hidden_size), batch_size)

        if self.direction == 0:
            self.lr_bwd_forget_mult(grid=grid, block=(grid_hidden_size, 1),
                                    args=[h.data_ptr(), f.data_ptr(), x.data_ptr(), grad_h.data_ptr(), grad_f.data_ptr(),
                                       grad_x.data_ptr(), grad_h_init.data_ptr(), seq_size, batch_size, hidden_size],
                                    stream=self.stream)
        elif self.direction == 1:
            self.rl_bwd_forget_mult(grid=grid, block=(grid_hidden_size, 1),
                                    args=[h.data_ptr(), f.data_ptr(), x.data_ptr(), grad_h.data_ptr(), grad_f.data_ptr(),
                                       grad_x.data_ptr(), grad_h_init.data_ptr(), seq_size, batch_size, hidden_size],
                                    stream=self.stream)

        ###
        if hidden_init is not None:
            return grad_f, grad_x, grad_h_init
        return grad_f, grad_x


class ForgetMult(torch.nn.Module):
    r"""ForgetMult computes a simple recurrent equation:
    h_t = f_t * x_t + (1 - f_t) * h_{t-1}
    This equation is equivalent to dynamic weighted averaging.
    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - F (seq_len, batch, input_size): tensor containing the forget gate values, assumed in range [0, 1].
        - hidden_init (batch, input_size): tensor containing the initial hidden state for the recurrence (h_{t-1}).
        - use_cuda: If True, use the fast element-wise CUDA kernel for recurrence. If False, uses naive for loop. Default: True.
    """

    def __init__(self, direction=0):
        super(ForgetMult, self).__init__()
        self.direction = direction

    def forward(self, f, x, hidden_init=None, use_cuda=True):
        # Use CUDA by default unless it's available
        use_cuda = use_cuda and torch.cuda.is_available()
        # Ensure the user is aware when ForgetMult is not GPU version as it's far faster
        if use_cuda: assert f.is_cuda and x.is_cuda, 'GPU ForgetMult with fast element-wise CUDA kernel requested but tensors not on GPU'
        ###
        # Avoiding 'RuntimeError: expected a Variable argument, but got NoneType' when hidden_init is None
        if hidden_init is None: return GPUForgetMult(self.direction)(f, x) if use_cuda else CPUForgetMult(self.direction)(f, x)
        return GPUForgetMult(self.direction)(f, x, hidden_init) if use_cuda else CPUForgetMult(self.direction)(f, x, hidden_init)

