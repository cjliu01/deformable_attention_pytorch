from deform_attention import MSDeformAttn
import torch

if __name__ == '__main__':
    query = torch.randn((3, 128, 128))
    input_flatten = torch.randn((3, 17728, 128))
    reference_points = torch.randn((3, 128, 4, 2))
    input_spatial_shapes = torch.tensor([[8, 8], [16, 16], [32, 32], [128, 128]])
    attn = MSDeformAttn(d_model=128, n_levels=4)

    print(attn(query, reference_points, input_flatten, input_spatial_shapes).shape)