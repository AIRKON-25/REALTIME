# evaluation_utils.py
import math
from typing import List, Dict, Any, Sequence

import numpy as np
import torch
import torchvision.ops as tvops  # GPU NMS
import torchvision

from utils.edge.geometry_utils import parallelogram_from_triangle, aabb_of_poly4, iou_aabb_xywh

def orientation_from_triangle(tri: np.ndarray) -> float:
    tri = np.asarray(tri)
    if tri.shape[0] < 3:
        return 0.0
    vec = tri[2] - tri[1]
    if np.linalg.norm(vec) < 1e-6:
        vec = tri[1] - tri[0]
    if np.linalg.norm(vec) < 1e-6:
        return 0.0
    angle = math.atan2(float(vec[1]), float(vec[0]))
    return angle % math.pi

def orientation_error_deg(pred_tri: np.ndarray, gt_tri: np.ndarray) -> float:
    ap = orientation_from_triangle(pred_tri)
    ag = orientation_from_triangle(gt_tri)
    diff = abs(ap - ag)
    diff = min(diff, math.pi - diff)
    return math.degrees(diff)

def _aabb_metrics(boxA_xywh, boxB_xywh):
    """네 기존 iou_aabb_xywh() 재사용 + IoS만 추가 계산"""
    iou = iou_aabb_xywh(boxA_xywh, boxB_xywh)

    ax0, ay0, aw, ah = boxA_xywh
    bx0, by0, bw, bh = boxB_xywh
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih

    areaA = max(0.0, aw) * max(0.0, ah)
    areaB = max(0.0, bw) * max(0.0, bh)
    ios = inter / max(min(areaA, areaB), 1e-9)  # Intersection over Smaller

    return iou, ios


def _nms_iou_or_ios(dets, iou_thr=0.5, contain_thr=None, topk=300):
    dets_sorted = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep = []
    for d in dets_sorted:
        boxA = aabb_of_poly4(d["poly4"]) 
        suppress = False
        for k in keep:
            boxB = aabb_of_poly4(k["poly4"]) 
            iou, ios = _aabb_metrics(boxA, boxB)
            if (iou >= iou_thr) or (contain_thr is not None and ios >= contain_thr):
                suppress = True
                break
        if not suppress:
            keep.append(d)
        if len(keep) >= topk:
            break
    return keep


def _resolve_conf_threshold(conf_cfg, cam_id) -> float:
    """Return a float confidence threshold for the given cam id."""
    if isinstance(conf_cfg, dict):
        if cam_id in conf_cfg:
            return float(conf_cfg[cam_id])
        cam_key = str(cam_id)
        if cam_key in conf_cfg:
            return float(conf_cfg[cam_key])
        if conf_cfg:
            return float(next(iter(conf_cfg.values())))
        raise ValueError("conf_th dictionary is empty")
    return float(conf_cfg)


def _decode_predictions_impl(
    cam_id,
    outputs,
    strides,
    clip_cells=None,
    conf_th=0.15,
    nms_iou=0.5,
    topk=300,
    contain_thr=0.7,     #작은 객체의 큰 객체 대비 겹침 정도
    score_mode="obj",      # "obj" | "cls" | "obj*cls"
    use_gpu_nms=False      # True면 torchvision.ops.nms 사용
):
    """
    outputs: [(reg,obj,cls)] * L, 각 텐서 shape = (B, C, Hs, Ws)
    - reg는 셀 단위 오프셋을 예측 (train.py와 동일)
    - centers = ((x+0.5)*s, (y+0.5)*s)
    - 최종 점: centers + reg*stride  (px)
    """
    assert score_mode in ("obj", "cls", "obj*cls")
    B = outputs[0][0].shape[0]
    batch_results = []

    for b in range(B):
        dets = []
        boxes_for_nms = []
        scores_for_nms = []

        for l, (reg, obj, cls) in enumerate(outputs):
            stride = float(strides[l]) if not isinstance(strides[l], torch.Tensor) else float(strides[l].item())

            # 점수 맵 만들기
            obj_map = torch.sigmoid(obj[b, 0])  # (Hs,Ws)
            cls_logits = cls[b]
            if cls_logits.shape[0] == 1:
                cls_score_map = torch.sigmoid(cls_logits[0])
                cls_id_map = torch.zeros_like(cls_score_map, dtype=torch.long)
            else:
                cls_prob = torch.softmax(cls_logits, dim=0)
                cls_score_map, cls_id_map = torch.max(cls_prob, dim=0)

            if score_mode == "obj":
                score_map = obj_map
            elif score_mode == "cls":
                score_map = cls_score_map
            else:  # "obj*cls"
                score_map = obj_map * cls_score_map

            thr = _resolve_conf_threshold(conf_th, cam_id)
            keep = score_map > thr
            if keep.sum().item() == 0:
                continue

            # 좌표 준비
            keep_idx = keep.nonzero(as_tuple=False)  # (K,2) [y,x]
            ys = keep_idx[:, 0].float()
            xs = keep_idx[:, 1].float()
            scores = score_map[keep]
            cls_scores = cls_score_map[keep]
            cls_ids = cls_id_map[keep]

            # 회귀 맵: (Hs,Ws,3,2)
            reg_map = reg[b].detach().permute(1, 2, 0).reshape(obj_map.shape[0], obj_map.shape[1], 3, 2)
            pred_off = reg_map[keep]  # (K,3,2) in cells

            # (옵션) tanh 클립
            if clip_cells is not None:
                R = float(clip_cells)
                pred_off = R * torch.tanh(pred_off / max(1e-6, R))

            # anchors (px) & 절대 좌표 (px)
            centers = torch.stack(((xs + 0.5) * stride, (ys + 0.5) * stride), dim=-1)  # (K,2)
            pred_tri = centers[:, None, :] + pred_off * stride  # (K,3,2) in px

            # dets 작성 + NMS용 AABB 수집
            tri_np = pred_tri.cpu().numpy()
            scores_np = scores.detach().cpu().numpy()
            cls_scores_np = cls_scores.detach().cpu().numpy()
            cls_ids_np = cls_ids.detach().cpu().numpy()
            for tri_pts, sc, cls_sc, cls_idx in zip(tri_np, scores_np, cls_scores_np, cls_ids_np):
                poly4 = parallelogram_from_triangle(tri_pts[0], tri_pts[1], tri_pts[2])
                dets.append({
                    "score": float(sc),
                    "cls_score": float(cls_sc),
                    "class_id": int(cls_idx),
                    "cls": int(cls_idx),
                    "poly4": poly4,
                    "tri": tri_pts
                })
                # NMS용 xyxy
                x0, y0 = poly4[:,0].min(), poly4[:,1].min()
                x1, y1 = poly4[:,0].max(), poly4[:,1].max()
                boxes_for_nms.append([x0, y0, x1, y1])
                scores_for_nms.append(float(sc))

        # NMS
        if not dets:
            batch_results.append([])
            continue

        if use_gpu_nms and len(dets) > 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            boxes_t = torch.tensor(boxes_for_nms, dtype=torch.float32, device=device)
            scores_t = torch.tensor(scores_for_nms, dtype=torch.float32, device=device)
            keep_idx = torchvision.ops.nms(boxes_t, scores_t, nms_iou)
            keep_idx = keep_idx[:topk].detach().cpu().tolist()
            dets = [dets[i] for i in keep_idx]
            # ← GPU NMS 이후 "포함율" 후처리(선택)
            if contain_thr is not None:
                dets = _nms_iou_or_ios(dets, iou_thr=1.1, contain_thr=contain_thr, topk=topk)
                # iou_thr=1.1로 두면 사실상 IoU 조건은 무시되고 IoS만 적용하는 후처리
                    
            batch_results.append(dets)
        else:
            dets_nms = _nms_iou_or_ios(
                dets,
                iou_thr=nms_iou,
                contain_thr=(float(contain_thr) if isinstance(contain_thr, (int, float)) else None),
                topk=topk
            )
            batch_results.append(dets_nms)

    return batch_results


def decode_predictions(*args, **kwargs):
    """Backward-compatible wrapper for decoding predictions.

    기존 버전(outputs,strides,...)과 신규 버전(cam_id,outputs,strides,...) 호출 모두 지원한다.
    cam_id가 주어지지 않으면 기본값 0을 사용한다.
    """

    positional_opt_names: Sequence[str] = (
        "clip_cells",
        "conf_th",
        "nms_iou",
        "topk",
        "contain_thr",
        "score_mode",
        "use_gpu_nms",
    )

    if len(args) >= 3 and not isinstance(args[0], (list, tuple)):
        # 신규 시그니처(cam_id, outputs, strides, ...)
        cam_id = args[0]
        outputs = args[1]
        strides = args[2]
        remaining = list(args[3:])
    else:
        # 구 버전(outputs, strides, ...)
        cam_id = kwargs.pop("cam_id", 0)
        outputs = args[0]
        strides = args[1]
        remaining = list(args[2:])

    for name, value in zip(positional_opt_names, remaining):
        kwargs.setdefault(name, value)

    return _decode_predictions_impl(cam_id, outputs, strides, **kwargs)

def evaluate_single_image(preds, gt_tris, gt_classes=None, iou_thr=0.5):
    gt_tris = np.asarray(gt_tris)
    num_gt = gt_tris.shape[0]
    if gt_classes is not None:
        gt_classes = np.asarray(gt_classes).reshape(-1)
        if gt_classes.shape[0] != num_gt:
            raise ValueError("gt_classes length must match gt_tris length")
    preds_sorted = sorted(preds, key=lambda d: d["score"], reverse=True)
    if num_gt == 0:
        records = [(det["score"], 0, 0.0, None, int(det.get("class_id", det.get("cls", 0)))) for det in preds_sorted]
        return records, 0

    gt_boxes = [aabb_of_poly4(parallelogram_from_triangle(tri[0], tri[1], tri[2])) for tri in gt_tris]
    matched = [False] * num_gt
    records = []

    for det in preds_sorted:
        pred_box = aabb_of_poly4(det["poly4"])
        det_cls = int(det.get("class_id", det.get("cls", 0)))
        best_iou = 0.0
        best_idx = -1
        for idx, gt_box in enumerate(gt_boxes):
            if matched[idx]:
                continue
            if gt_classes is not None and det_cls != int(gt_classes[idx]):
                continue
            # aabb IoU
            ax0, ay0, aw, ah = pred_box
            bx0, by0, bw, bh = gt_box
            ax1, ay1 = ax0 + aw, ay0 + ah
            bx1, by1 = bx0 + bw, by0 + bh
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1), min(ay1, by1)
            iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
            inter = iw * ih
            union = aw * ah + bw * bh - inter + 1e-9
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_thr:
            matched[best_idx] = True
            orient_err = orientation_error_deg(det["tri"], gt_tris[best_idx])
            records.append((det["score"], 1, best_iou, orient_err, det_cls))
        else:
            records.append((det["score"], 0, best_iou, None, det_cls))

    return records, sum(matched)

def _average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def compute_detection_metrics(records, total_gt):
    if not records:
        return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "mAOE_deg": float('nan')}

    records.sort(key=lambda r: r[0], reverse=True)
    tps = np.array([r[1] for r in records], dtype=np.float32)
    fps = 1.0 - tps
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    denom = np.maximum(tp_cum + fp_cum, 1e-9)
    precisions = tp_cum / denom

    if total_gt > 0:
        recalls = tp_cum / total_gt
    else:
        recalls = np.zeros_like(tp_cum)

    precision = float(precisions[-1]) if precisions.size > 0 else 0.0
    recall = float(recalls[-1]) if total_gt > 0 and recalls.size > 0 else 0.0
    map50 = _average_precision(recalls, precisions) if total_gt > 0 else 0.0

    orient_errors = [r[3] for r in records if r[1] == 1 and r[3] is not None]
    mAOE = float(np.mean(orient_errors)) if orient_errors else float('nan')

    return {"precision": precision, "recall": recall, "map50": map50, "mAOE_deg": mAOE}
