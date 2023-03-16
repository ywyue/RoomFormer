# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.poly_ops import get_all_order_corners
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    We do the matching in polygon (room) level
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_coords: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_coords: This is the relative weight of the L1 error of the polygon coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_coords = cost_coords
        assert cost_class != 0 or cost_coords != 0, "all costs cant be 0"

    def calculate_angles(self, polygon):
        vect1 = polygon.roll(1, 0)-polygon
        vect2 = polygon.roll(-1, 0)-polygon
        cos_sim = ((vect1 * vect2).sum(1)+1e-9)/(torch.norm(vect1, p=2, dim=1)*torch.norm(vect2, p=2, dim=1)+1e-9)
        # cos_sim = F.cosine_similarity(vect1, vect2)
        # angles = torch.acos(torch.clamp(cos_sim, -1 + 1e-7 , 1 - 1e-7))
        # if torch.isnan(angles).sum() >=1:
        #     print('a')
        # return angles
        return cos_sim

    def calculate_src_angles(self, polygon):
        vect1 = polygon.roll(1, 1)-polygon
        vect2 = polygon.roll(-1, 1)-polygon

        cos_sim = ((vect1 * vect2).sum(-1)+1e-9)/(torch.norm(vect1, p=2, dim=-1)*torch.norm(vect2, p=2, dim=-1)+1e-9)

        return cos_sim

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_polys, num_queries_per_poly] with the classification logits
                 "pred_coords": Tensor of dim [batch_size, num_polys, num_queries_per_poly, 2] with the predicted polygons coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_polys, num_queries_per_poly] (where num_target_polys is the number of ground-truth
                           polygons in the target) containing the class labels
                 "coords": Tensor of dim [num_target_polys, num_queries_per_poly * 2] containing the target polygons coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order), max(index_i) = num_polys - 1
                - index_j is the indices of the corresponding selected targets (in order), max(index_j) = num_target_polys - 1
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_polys, num_target_polys)
        """
        with torch.no_grad():
            bs, num_polys = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            src_prob = outputs["pred_logits"].flatten(0,1).sigmoid()
            src_polys = outputs["pred_coords"].flatten(0, 1).flatten(1, 2)

            # Also concat the target labels and coords
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_polys = torch.cat([v["coords"] for v in targets])
            tgt_len = torch.cat([v["lengths"] for v in targets])

            # Compute the pair-wise classification cost. 
            # We just use the L1 distance between prediction probality and target labels (inc. no-object calss)
            cost_class = torch.cdist(src_prob, tgt_ids, p=1)

            # Compute the L1 cost between coords
            # Here we does not consider no-object corner in target since we filter out no-object corners in results
            cost_coords = torch.zeros([src_polys.shape[0], tgt_polys.shape[0]], device=src_polys.device)
            for i in range(tgt_polys.shape[0]):
                tgt_polys_single = tgt_polys[i, :tgt_len[i]]
                all_polys = get_all_order_corners(tgt_polys_single)
                cost_coords[:, i] = torch.cdist(src_polys[:, :tgt_len[i]], all_polys , p=1).min(axis=-1)[0]

            # Final cost matrix
            C = self.cost_coords * cost_coords + self.cost_class * cost_class
            C = C.view(bs, num_polys, -1).cpu()

            sizes = [len(v["coords"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_coords=args.set_cost_coords)
