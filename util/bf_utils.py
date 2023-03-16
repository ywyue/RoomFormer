import copy
import math
import numpy as np
import torch
from torch import nn

from detectron2.utils.registry import Registry

# need an easier place to avoid circular dependencies.
POLY_LOSS_REGISTRY = Registry("POLY_LOSS")
POLY_LOSS_REGISTRY.__doc__ = """
Registry for loss computations on predicted polygons.
"""

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def clip_and_normalize_polygons(polys, inf_value=2.01):    
    min_x, _ = polys[:, :, 0].min(dim=-1)
    min_y, _ = polys[:, :, 1].min(dim=-1)

    polys[torch.isinf(polys)] = -np.inf
    max_x, _ = polys[:, :, 0].max(dim=-1)
    max_y, _ = polys[:, :, 1].max(dim=-1)

    polys[torch.isinf(polys)] = inf_value

    min_xy = torch.stack((min_x, min_y), dim=-1)
    max_xy = torch.stack((max_x, max_y), dim=-1) - min_xy

    polys = (polys - min_xy.unsqueeze(1)) / max_xy.unsqueeze(1)

    return polys

def pad_polygons(polys):
    count = len(polys)
    max_vertices = max([len(p) for p in polys])
    pad_count = [max_vertices - len(p) for p in polys]

    # add between the first and second vertices.
    xs = [np.linspace(polys[i][0][0] + 0.00001, polys[i][1][0] - 0.00001, num=pad_count[i]) for i in range(count)]
    ys = [np.linspace(polys[i][0][1] + 0.00001, polys[i][1][1] - 0.00001, num=pad_count[i]) for i in range(count)]

    xys = [np.stack((xs[i], ys[i]), axis=-1) for i in range(count)]
    polys = [np.concatenate((polys[i][:1], xys[i], polys[i][1:])) for i in range(count)]

    return np.stack(polys)

def rasterize_instances(rasterizer, instances, shape, offset=0.0):
    if shape[0] != shape[1]:
        raise ValueError("expected square")
    
    device = instances[0].gt_boxes.device
    all_polygons = clip_and_normalize_polygons(torch.from_numpy(pad_polygons(list(itertools.chain.from_iterable([
        [p[0].reshape(-1, 2) for p in inst.gt_masks.polygons] for inst in instances])))).float().to(device))

    # to me it seems the offset would need to be in _pixel_ space?
    return rasterizer(all_polygons * float(shape[1].item()) + offset, shape[1].item(), shape[0].item(), 1.0)    

def get_union_box(p, box):
    # compute the enclosing box.
    all_points = torch.cat((p, box.view(-1, 2, 2)), dim=-2)
    min_xy = torch.min(all_points, dim=-2)[0]
    max_xy = torch.max(all_points, dim=-2)[0]

    return torch.cat((min_xy, max_xy), dim=-1)

def sample_ellipse_fast(x, y, r1, r2, count=32, dt=0.01):    
    batch_size, num_el = r1.shape
    device = r1.device
    num_integrals = int(round(2 * math.pi / dt))
    
    thetas = dt * torch.arange(num_integrals, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_el, 1)    
    thetas_c = torch.cumsum(thetas, dim=-1)
    dpt = torch.sqrt((r1.unsqueeze(-1) * torch.sin(thetas_c)) ** 2 + (r2.unsqueeze(-1) * torch.cos(thetas_c)) ** 2)
    circumference = dpt.sum(dim=-1)

    run = torch.cumsum(torch.sqrt((r1.unsqueeze(-1) * torch.sin(thetas + dt)) ** 2 + (r2.unsqueeze(-1) * torch.cos(thetas + dt)) ** 2), dim=-1)
    sub = (count * run) / circumference.unsqueeze(-1)

    # OK, now find the smallest point >= 0..count-1
    counts = torch.arange(count, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_el, num_integrals, 1)
    diff = sub.unsqueeze(dim=-1) - counts
    diff[diff < 0] = 10000.0

    idx = diff.argmin(dim=2)
    thetas = torch.gather(thetas + dt, -1, idx)

    xy = torch.stack((x.unsqueeze(-1) + r1.unsqueeze(-1) * torch.cos(thetas), y.unsqueeze(-1) + r2.unsqueeze(-1) * torch.sin(thetas)), dim=-1)

    return xy

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
