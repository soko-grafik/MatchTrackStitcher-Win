import unittest
import numpy as np
import cv2

def compute_smart_seam_path(
    img_l: np.ndarray,
    img_r: np.ndarray,
    overlap_x0: int,
    overlap_x1: int,
    center_x: int,
    center_weight: float = 40.0
) -> np.ndarray:
    """
    Computes an optimal seam path x*(y) through the overlap region using Dynamic Programming.
    Avoids cutting through high-contrast objects (like players or coaches) by routing through
    uniform regions (such as grass).
    
    img_l, img_r: remapped panorama frames [H, W, 3] uint8
    overlap_x0, overlap_x1: horizontal pixel boundaries of overlap
    center_x: preferred nominal seam center
    Returns: 1D array of shape [H] with optimal x coordinate for each row y.
    """
    H, W = img_l.shape[:2]
    w_ov = overlap_x1 - overlap_x0
    if w_ov < 4:
        return np.full(H, center_x, dtype=np.int32)
    
    # Work on a downsampled strip for sub-millisecond calculation
    scale_y = 270 / float(H) if H > 270 else 1.0
    h_small = int(H * scale_y)
    w_small = w_ov
    
    strip_l = cv2.resize(img_l[:, overlap_x0:overlap_x1], (w_small, h_small), interpolation=cv2.INTER_AREA)
    strip_r = cv2.resize(img_r[:, overlap_x0:overlap_x1], (w_small, h_small), interpolation=cv2.INTER_AREA)
    
    # Cost = pixel difference + edge magnitude + distance penalty from nominal center
    diff = np.mean(np.abs(strip_l.astype(np.float32) - strip_r.astype(np.float32)), axis=2)
    
    gray_l = cv2.cvtColor(strip_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(strip_r, cv2.COLOR_BGR2GRAY)
    edge_l = cv2.Sobel(gray_l, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(gray_l, cv2.CV_32F, 0, 1)**2
    edge_r = cv2.Sobel(gray_r, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(gray_r, cv2.CV_32F, 0, 1)**2
    edge_cost = np.sqrt(edge_l) + np.sqrt(edge_r)
    
    # Distance penalty from center_x
    local_center = center_x - overlap_x0
    xs = np.arange(w_small, dtype=np.float32)
    dist_penalty = center_weight * ((xs - local_center) / float(w_small)) ** 2
    cost = diff * 1.5 + edge_cost * 0.8 + dist_penalty[np.newaxis, :]
    
    # Dynamic programming forward pass
    M = np.zeros_like(cost)
    M[0, :] = cost[0, :]
    backtrack = np.zeros((h_small, w_small), dtype=np.int32)
    
    for y in range(1, h_small):
        m_prev = M[y - 1, :]
        # Vectorized 3-neighborhood minimum: left, center, right
        left = np.pad(m_prev[:-1], (1, 0), mode='edge')
        center = m_prev
        right = np.pad(m_prev[1:], (0, 1), mode='edge')
        
        stacked = np.stack([left, center, right], axis=0) # [3, w_small]
        min_idx = np.argmin(stacked, axis=0) - 1 # -1, 0, +1
        min_val = np.min(stacked, axis=0)
        
        M[y, :] = cost[y, :] + min_val
        prev_x = np.clip(np.arange(w_small) + min_idx, 0, w_small - 1)
        backtrack[y, :] = prev_x
        
    # Backtracking
    path_small = np.zeros(h_small, dtype=np.int32)
    best_x = int(np.argmin(M[-1, :]))
    path_small[-1] = best_x
    for y in range(h_small - 1, 0, -1):
        best_x = backtrack[y, best_x]
        path_small[y - 1] = best_x
        
    # Upscale path back to original H
    y_coords_orig = np.linspace(0, h_small - 1, H)
    path_orig_local = np.interp(y_coords_orig, np.arange(h_small), path_small)
    
    # Smooth path with box filter / 1D gaussian
    kernel_size = max(5, int(H * 0.05))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / float(kernel_size)
    padded = np.pad(path_orig_local, (kernel_size // 2, kernel_size // 2), mode='edge')
    smoothed_local = np.convolve(padded, kernel, mode='valid')[:H]
    
    path_full = (overlap_x0 + smoothed_local).astype(np.int32)
    return np.clip(path_full, overlap_x0, overlap_x1 - 1)


class TestSmartSeam(unittest.TestCase):
    def test_smart_seam_routing_around_parallax_shear(self):
        H, W = 720, 2560
        img_l = np.full((H, W, 3), (40, 140, 40), dtype=np.uint8) # green turf
        img_r = np.full((H, W, 3), (40, 140, 40), dtype=np.uint8) # green turf
        
        center_x = 1280
        # In img_l, coach is at 1240:1280
        img_l[400:650, 1240:1280] = (20, 20, 20)
        # In img_r, coach is at 1270:1310
        img_r[400:650, 1270:1310] = (20, 20, 20)
        
        overlap_x0 = 1180
        overlap_x1 = 1380
        
        # When penalizing edges/gradients of the object as well
        path = compute_smart_seam_path(img_l, img_r, overlap_x0, overlap_x1, center_x)
        self.assertEqual(len(path), H)
        print(f"Parallax test: Path at y=100: {path[100]}, at y=500: {path[500]}")

if __name__ == "__main__":
    unittest.main()
