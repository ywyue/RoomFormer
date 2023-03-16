import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

from torch.autograd import Function

import native_rasterizer

MODE_BOUNDARY = "boundary"
MODE_MASK = "mask"
MODE_HARD_MASK = "hard_mask"

MODE_MAPPING = {
    MODE_BOUNDARY: 0,
    MODE_MASK: 1,
    MODE_HARD_MASK: 2
}
    
class SoftPolygonFunction(Function):
    @staticmethod
    def forward(ctx, vertices, width, height, inv_smoothness=1.0, mode=MODE_BOUNDARY):
        ctx.width = width
        ctx.height = height
        ctx.inv_smoothness = inv_smoothness
        ctx.mode = MODE_MAPPING[mode]

        vertices = vertices.clone()
        ctx.device = vertices.device
        ctx.batch_size, ctx.number_vertices = vertices.shape[:2]
        
        rasterized = torch.FloatTensor(ctx.batch_size, ctx.height, ctx.width).fill_(0.0).to(device=ctx.device)

        contribution_map = torch.IntTensor(
            ctx.batch_size,
            ctx.height,
            ctx.width).fill_(0).to(device=ctx.device)
        rasterized, contribution_map = native_rasterizer.forward_rasterize(vertices, rasterized, contribution_map, width, height, inv_smoothness, ctx.mode)
        ctx.save_for_backward(vertices, rasterized, contribution_map)

        return rasterized #, contribution_map

    @staticmethod
    def backward(ctx, grad_output):
        vertices, rasterized, contribution_map = ctx.saved_tensors

        grad_output = grad_output.contiguous()

        #grad_vertices = torch.FloatTensor(
        #    ctx.batch_size, ctx.height, ctx.width, ctx.number_vertices, 2).fill_(0.0).to(device=ctx.device)
        grad_vertices = torch.FloatTensor(
            ctx.batch_size, ctx.number_vertices, 2).fill_(0.0).to(device=ctx.device)
        grad_vertices = native_rasterizer.backward_rasterize(
            vertices, rasterized, contribution_map, grad_output, grad_vertices, ctx.width, ctx.height, ctx.inv_smoothness, ctx.mode)

        return grad_vertices, None, None, None, None
    
class SoftPolygon(nn.Module):
    MODES = [MODE_BOUNDARY, MODE_MASK, MODE_HARD_MASK]
    
    def __init__(self, inv_smoothness=1.0, mode=MODE_BOUNDARY):
        super(SoftPolygon, self).__init__()

        self.inv_smoothness = inv_smoothness

        if not (mode in SoftPolygon.MODES):
            raise ValueError("invalid mode: {0}".format(mode))
            
        self.mode = mode

    def forward(self, vertices, width, height, p, color=False):
        return SoftPolygonFunction.apply(vertices, width, height, self.inv_smoothness, self.mode)

def pnp(vertices, width, height):
    device = vertices.device
    batch_size = vertices.size(0)
    polygon_dimension = vertices.size(1)

    y_index = torch.arange(0, height).to(device)
    x_index = torch.arange(0, width).to(device)
    
    grid_y, grid_x = torch.meshgrid(y_index, x_index)
    xp = grid_x.unsqueeze(0).repeat(batch_size, 1, 1).float()
    yp = grid_y.unsqueeze(0).repeat(batch_size, 1, 1).float()

    result = torch.zeros((batch_size, height, width)).bool().to(device)

    j = polygon_dimension - 1
    for vn in range(polygon_dimension):
        from_x = vertices[:, vn, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)        
        from_y = vertices[:, vn, 1].unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)

        to_x = vertices[:, j, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)
        to_y = vertices[:, j, 1].unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)

        has_condition = torch.logical_and((from_y > yp) != (to_y > yp), xp < ((to_x - from_x) * (yp - from_y) / (to_y - from_y) + from_x))
        
        if has_condition.any():
            result[has_condition] = ~result[has_condition]

        j = vn

    signed_result = -torch.ones((batch_size, height, width), device=device)
    signed_result[result] = 1.0

    return signed_result

# used for verification purposes only.
class SoftPolygonPyTorch(nn.Module):
    def __init__(self, inv_smoothness=1.0):
        super(SoftPolygonPyTorch, self).__init__()

        self.inv_smoothness = inv_smoothness

    # vertices is N x P x 2
    # todo, implement inside outside.
    def forward(self, vertices, width, height, p, color=False):
        device = vertices.device
        batch_size = vertices.size(0)
        polygon_dimension = vertices.size(1)

        inside_outside = pnp(vertices, width, height)

        # discrete points we will sample from.
        y_index = torch.arange(0, height).to(device)
        x_index = torch.arange(0, width).to(device)
        
        grid_y, grid_x = torch.meshgrid(y_index, x_index)
        grid_x = grid_x.unsqueeze(0).repeat(batch_size, 1, 1).float()
        grid_y = grid_y.unsqueeze(0).repeat(batch_size, 1, 1).float()
        
        # do this "per dimension"
        distance_segments = []
        over_segments = []
        color_segments = []
        for from_index in range(polygon_dimension):
            segment_result = torch.zeros((batch_size, height, width)).to(device)
            from_vertex = vertices[:, from_index].unsqueeze(-1).unsqueeze(-1)
            
            if from_index == (polygon_dimension - 1):
                to_vertex = vertices[:, 0].unsqueeze(-1).unsqueeze(-1)
            else:
                to_vertex = vertices[:, from_index + 1].unsqueeze(-1).unsqueeze(-1)
                
            x2_sub_x1 = to_vertex[:, 0] - from_vertex[:, 0]
            y2_sub_y1 = to_vertex[:, 1] - from_vertex[:, 1]
            square_segment_length = x2_sub_x1 * x2_sub_x1 + y2_sub_y1 * y2_sub_y1 + 0.00001
            
            # figure out if this is a major/minor segment (todo?)
            x_sub_x1 = grid_x - from_vertex[:, 0]
            y_sub_y1 = grid_y - from_vertex[:, 1]
            x_sub_x2 = grid_x - to_vertex[:, 0]
            y_sub_y2 = grid_y - to_vertex[:, 1]
            
            # dot between the given point and first vertex and first vertex and second vertex.
            dot = ((x_sub_x1 * x2_sub_x1) + (y_sub_y1 * y2_sub_y1)) / square_segment_length

            # needlessly computed sometimes.
            x_proj = grid_x - (from_vertex[:, 0] + dot * x2_sub_x1)
            y_proj = grid_y - (from_vertex[:, 1] + dot * y2_sub_y1)

            from_closest = dot < 0
            to_closest = dot > 1
            interior_closest = (dot >= 0) & (dot <= 1)

            segment_result[from_closest] = x_sub_x1[from_closest] ** 2 + y_sub_y1[from_closest] ** 2
            segment_result[to_closest] = x_sub_x2[to_closest] ** 2 + y_sub_y2[to_closest] ** 2
            segment_result[interior_closest] = x_proj[interior_closest] ** 2 + y_proj[interior_closest] ** 2

            distance_map = -segment_result
            distance_segments.append(distance_map)

            signed_map = torch.sigmoid(-distance_map * inside_outside / self.inv_smoothness)
            over_segments.append(signed_map)

        F_max, F_arg = torch.max(torch.stack(distance_segments, dim=-1), dim=-1)
        F_theta = torch.gather(torch.stack(over_segments, dim=-1), dim=-1, index=F_arg.unsqueeze(-1))[..., 0]

        return F_theta
