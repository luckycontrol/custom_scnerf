import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.autograd.set_detect_anomaly(True)

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, part, weight=None, **kwargs):
        self.part = part
        self.kwargs = kwargs
        self.weights = weight
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        
        # 3차원 공간 학습을 위한 코드
        if self.part == "render":
            for freq in freq_bands:
                for p_fn in self.kwargs['periodic_fns']:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                    out_dim += d

        # 카메라 파라미터 학습을 위한 코드
        else:
            for i, _ in enumerate(freq_bands):
                for p_fn in self.kwargs['periodic_fns']:
                    def weighted_periodic_fn(x, p_fn=p_fn, freq=freq_bands[i], weight=self.weights[i]):
                        return p_fn(x * freq) * weight
                    # embed_fns.append(lambda x, p_fn=p_fn, freq=freq_bands[i], weight=self.weights[i]: p_fn(x * freq) * weight)
                    embed_fns.append(weighted_periodic_fn)
                    out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim 

    def embed(self, inputs):
        inputs = [fn(inputs) for fn in self.embed_fns]
        for i in range(len(inputs)):
            inputs[i] = inputs[i].detach().requires_grad_(True)

        output = torch.cat(inputs, -1)
        return output
        # return torch.cat([fn(inputs) for fn in self.embed_fns], -1)   

# nerfmm - 수정: 3차원 공간 학습, 카메라 파라미터 학습 구분
def get_embedder(device, part, progress, multires, i=0):

    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
        'include_input' : True,
        'input_dims' : 3,
        'max_freq_log2' : multires-1,
        'num_freqs' : multires,
        'log_sampling' : True,
        'periodic_fns' : [torch.sin, torch.cos],
    }

    start = 0.1
    end = 0.5

    # 카메라파라미터 학습 완료 후 3차원 공간 학습을 위한 코드
    if part == "render":
        embedder_obj = Embedder(part, **embed_kwargs)

    # 카메라 파라미터 학습을 위한 코드
    else:
        alpha = (progress.data - start) / (end - start) * multires
        k = torch.arange(multires, dtype=torch.float32, device=device)

        weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2

        embedder_obj = Embedder(part, weight, **embed_kwargs)
    
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [DenseLayer(input_ch, W, activation="relu")] + [DenseLayer(W, W, activation="relu") if i not in self.skips else DenseLayer(W + input_ch, W, activation="relu") for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([DenseLayer(input_ch_views + W, W//2, activation="relu")])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = DenseLayer(W, W, activation="linear")
            self.alpha_linear = DenseLayer(W, 1, activation="linear")
            self.rgb_linear = DenseLayer(W//2, 3, activation="linear")
        else:
            self.output_linear = DenseLayer(W, output_ch, activation="linear")

    # for nerfmm - 가중치를 출력하는 함수
    def log(self):
        print(f'[pts_linears]')
        for i, layer in enumerate(self.pts_linears):
            print(f'layer {i} weight:')
            print(f'{layer.weight}')
        
        print(f'[views_linears]')
        for i, layer in enumerate(self.views_linears):
            print(f'layer {i} weight:')
            print(f'{layer.weight}')
        
        print(f'[output_linear]')
        print(f'{self.output_linear.weight}')

    # for nerfmm - 가중치를 초기화하는 함수
    def reset(self):
        for i, layer in enumerate(self.pts_linears):
            layer.reset_parameters()
        
        for i, layer in enumerate(self.views_linears):
            layer.reset_parameters()
        
        self.output_linear.reset_parameters()

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))
        
        
def fix_seeds(random_seed):

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
