"""SORT utils."""


def bbox_iou(boxes1: "torch.Tensor", boxes2: "torch.Tensor") -> "torch.Tensor":
    """Compute IoU between all pairs of boxes.

    Args:
        boxes1: N 2D boxes in format (x1, y1, x2, y2, Optional[score])
        boxes2: M 2D boxes in format (x1, y1, x2, y2, Optional[score])

    Returns:
        Tensor: IoU (N, M).
    """
    area1, area2 = bbox_area(boxes1), bbox_area(boxes2)
    inter = bbox_intersection(boxes1, boxes2)

    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou
