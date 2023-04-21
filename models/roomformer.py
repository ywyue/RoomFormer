# Modified from Deformable DETR
# Yuanwen Yue

import torch
import torch.nn.functional as F
from torch import nn
import math

from util.misc import NestedTensor, nested_tensor_from_tensor_list, interpolate, inverse_sigmoid

from .backbone import build_backbone
from .matcher import build_matcher
from .losses import custom_L1_loss, MaskRasterizationLoss
from .deformable_transformer import build_deforamble_transformer
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class RoomFormer(nn.Module):
    """ This is the RoomFormer module that performs floorplan reconstruction """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_polys, num_feature_levels,
                 aux_loss=True, with_poly_refine=False, masked_attn=False, semantic_classes=-1):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of possible corners
                         in a single image.
            num_polys: maximal number of possible polygons in a single image. 
                       num_queries/num_polys would be the maximal number of possible corners in a single polygon.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_poly_refine: iterative polygon refinement
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_polys = num_polys
        assert  num_queries % num_polys == 0
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.coords_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.num_feature_levels = num_feature_levels

        self.query_embed = nn.Embedding(num_queries, 2)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_poly_refine = with_poly_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.coords_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.coords_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        
        if with_poly_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.coords_embed = _get_clones(self.coords_embed, num_pred)
            nn.init.constant_(self.coords_embed[0].layers[-1].bias.data[2:], -2.0)
        else:
            nn.init.constant_(self.coords_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.coords_embed = nn.ModuleList([self.coords_embed for _ in range(num_pred)])

        self.transformer.decoder.coords_embed = self.coords_embed
        self.transformer.decoder.class_embed = self.class_embed
        
        # Semantically-rich floorplan
        self.room_class_embed = None
        if semantic_classes > 0:
            self.room_class_embed = nn.Linear(hidden_dim, semantic_classes)


        self.num_queries_per_poly = num_queries // num_polys

        # The attention mask is used to prevent object queries in one polygon attending to another polygon, default false
        if masked_attn:
            self.attention_mask = torch.ones((num_queries, num_queries), dtype=torch.bool)
            for i in range(num_polys):
                self.attention_mask[i * self.num_queries_per_poly:(i + 1) * self.num_queries_per_poly,
                i * self.num_queries_per_poly:(i + 1) * self.num_queries_per_poly] = False
        else:
            self.attention_mask = None

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x C x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_coords": The normalized corner coordinates for all queries, represented as
                               (x, y). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        bs = samples.tensors.shape[0]
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)


        query_embeds = self.query_embed.weight
        tgt_embeds = self.tgt_embed.weight
        
        hs, init_reference, inter_references, inter_classes = self.transformer(srcs, masks, pos, query_embeds, tgt_embeds, self.attention_mask)

        num_layer = hs.shape[0]
        outputs_class = inter_classes.reshape(num_layer, bs, self.num_polys, self.num_queries_per_poly)
        outputs_coord = inter_references.reshape(num_layer, bs, self.num_polys, self.num_queries_per_poly, 2)
        
        out = {'pred_logits': outputs_class[-1], 'pred_coords': outputs_coord[-1]}

        # hack implementation of room label prediction, not compatible with auxiliary loss
        if self.room_class_embed is not None:
            outputs_room_class = self.room_class_embed(hs[-1].view(bs, self.num_polys, self.num_queries_per_poly, -1).mean(axis=2))
            out = {'pred_logits': outputs_class[-1], 'pred_coords': outputs_coord[-1], 'pred_room_logits': outputs_room_class}
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_coords': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for multiple polygons.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth polygons and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and coords)
    """
    def __init__(self, num_classes, semantic_classes, matcher, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of classes for corner validity (binary)
            semantic_classes: number of semantic classes for polygon (room type, door, window)
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.semantic_classes = semantic_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.raster_loss = MaskRasterizationLoss(None)


    def loss_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels"
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        bs = src_logits.shape[0]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape, self.num_classes-1,
                                    dtype=torch.float32, device=src_logits.device)
        target_classes[idx] = target_classes_o


        loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)

        losses = {'loss_ce': loss_ce}

        # hack implementation of room label/door/window prediction
        if 'pred_room_logits' in outputs:
            room_src_logits = outputs['pred_room_logits']
            room_target_classes_o = torch.cat([t["room_labels"][J] for t, (_, J) in zip(targets, indices)])
            room_target_classes = torch.full(room_src_logits.shape[:2], self.semantic_classes-1,
                                        dtype=torch.int64, device=room_src_logits.device)
            room_target_classes[idx] = room_target_classes_o
            loss_ce_room = F.cross_entropy(room_src_logits.transpose(1, 2), room_target_classes)
            losses = {'loss_ce': loss_ce, 'loss_ce_room': loss_ce_room}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty corners
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([sum(v["lengths"]) for v in targets], device=device) / 2
        # Count the number of predictions that are NOT "no-object" (invalid corners)
        card_pred = (pred_logits.sigmoid() > 0.5).flatten(1, 2).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses


    def loss_polys(self, outputs, targets, indices):
        """Compute the losses related to the polygons:
           1. L1 loss for polygon coordinates
           2. Dice loss for polygon rasterizated binary masks
        """
        assert 'pred_coords' in outputs
        idx = self._get_src_permutation_idx(indices)
        bs = outputs['pred_coords'].shape[0]
        src_polys = outputs['pred_coords'][idx]
        target_polys = torch.cat([t['coords'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_len =  torch.cat([t['lengths'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_coords = custom_L1_loss(src_polys.flatten(1,2), target_polys, target_len)

        losses = {}
        losses['loss_coords'] = loss_coords

        # omit the rasterization loss for semantically-rich floorplan
        if self.semantic_classes == -1:
            loss_raster_mask = self.raster_loss(src_polys.flatten(1,2), target_polys, target_len)
            losses['loss_raster'] = loss_raster_mask

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'polys': self.loss_polys
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            # bin_targets = copy.deepcopy(targets)
            # for bt in bin_targets:
            #     bt['labels'] = torch.zeros_like(bt['labels'])
            # indices = self.matcher(enc_outputs, bin_targets)
            indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                # l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices)
                l_dict = self.get_loss(loss, enc_outputs, targets, indices)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args, train=True):
    num_classes = 1 # valid or invalid corner

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    model = RoomFormer(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_polys=args.num_polys,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_poly_refine=args.with_poly_refine,
        masked_attn=args.masked_attn,
        semantic_classes=args.semantic_classes
    )

    if not train:
        return model

    device = torch.device(args.device)
    matcher = build_matcher(args)
    weight_dict = {
                    'loss_ce': args.cls_loss_coef, 
                    'loss_ce_room': args.room_cls_loss_coef,
                    'loss_coords': args.coords_loss_coef,
                    'loss_raster': args.raster_loss_coef
                    }
    weight_dict['loss_dir'] = 1

    enc_weight_dict = {}
    enc_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
    weight_dict.update(enc_weight_dict)
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'polys', 'cardinality']
    # num_classes, matcher, weight_dict, losses
    criterion = SetCriterion(num_classes, args.semantic_classes, matcher, weight_dict, losses)
    criterion.to(device)

    return model, criterion
