"""Side-by-side GT / prediction flow video.

Both panels share one colour scale. Normalising each by its own magnitude range would make a
prediction that is uniformly half the true magnitude look identical to a correct one.
"""
import os

import cv2
import matplotlib.colors
import numpy as np


def flow_to_rgb(flow, max_mag):
    """(2, H, W) flow -> uint8 BGR. Hue is direction, value is magnitude against a fixed scale."""
    fx, fy = flow[0], flow[1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    ang = (np.arctan2(fy, fx) + np.pi) / (2 * np.pi)
    hsv = np.zeros(fx.shape + (3,), dtype=np.float32)
    hsv[..., 0] = ang
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(mag / max_mag, 0.0, 1.0) if max_mag > 0 else 0.0
    rgb = (matplotlib.colors.hsv_to_rgb(hsv) * 255).astype(np.uint8)
    # matplotlib gives RGB, cv2 writes BGR. ascontiguousarray because the reversing slice leaves
    # a negative stride, which cv2.putText will not accept.
    return np.ascontiguousarray(rgb[..., ::-1])


def _label(img, text):
    out = img.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def write_flow_video(gt, pred, mask, out_path, rgb_dir=None, files=None, fps=10.0,
                     percentile=99.0):
    """gt/pred: (N, 2, H, W). mask: (N, H, W). Writes GT | prediction [| camera] per frame.

    The shared colour scale is the given percentile of |GT| over valid pixels across the whole
    sequence, not the max, which a few near-field road pixels would otherwise own.
    """
    valid = mask > 0
    gt_mag = np.sqrt(gt[:, 0] ** 2 + gt[:, 1] ** 2)
    max_mag = float(np.percentile(gt_mag[valid], percentile)) if valid.any() else 1.0

    n, _, h, w = gt.shape
    panels = 3 if rgb_dir else 2
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * panels, h))
    if not writer.isOpened():
        raise SystemExit("could not open %s for writing" % out_path)

    for i in range(n):
        # Masking must not promote the panel out of uint8 -- cv2.putText refuses anything else.
        m = (mask[i][..., None] > 0).astype(np.uint8)
        row = [_label(flow_to_rgb(gt[i], max_mag) * m, "GT"),
               _label(flow_to_rgb(pred[i], max_mag) * m, "pred")]

        if rgb_dir:
            # Tensor filenames are 1-based over windows.csv's 0-based rows.
            idx = int(files[i].rsplit("_", 1)[1].split(".")[0]) - 1
            img = cv2.imread(os.path.join(rgb_dir, "%05d.png" % idx))
            row.append(_label(img if img is not None else np.zeros((h, w, 3), np.uint8),
                              "camera"))

        writer.write(np.concatenate(row, axis=1).astype(np.uint8))

    writer.release()
    print("  colour scale: |flow| = %.2f px maps to full brightness (p%.0f of GT)"
          % (max_mag, percentile))
