import numpy as np
from typing import List, Dict, Tuple

def _ensure_increasing_x(poly: np.ndarray) -> np.ndarray:
    """保证一条折线按 x 递增；poly 形状 [n,2]，列为 [x,y]。"""
    if poly.shape[0] < 2:
        return poly
    return poly if poly[0,0] <= poly[-1,0] else poly[::-1].copy()

def _interp_on_grid(poly: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """在给定 x_grid 上线性插值 y(x)，返回形状 [len(x_grid), 2] 的 [x, y]。"""
    x, y = poly[:,0], poly[:,1]
    y_i = np.interp(x_grid, x, y)
    return np.stack([x_grid, y_i], axis=1)

def lanes_to_segments(
    lane_lines: List[np.ndarray],
    step: float = 5.0,
    assume_left_is_larger_y: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """
    输入:
      lane_lines: N 条“车道线”(边界线)，每条为 [n_i,2] 的 [x,y] 点列（x 约单调）。
    输出:
      一个长度为 N-1 的列表；每个元素是 dict:
        {
          'left':   [m,2],  # 左边界
          'right':  [m,2],  # 右边界
          'center': [m,2],  # 中心线 = (left + right) / 2
        }
    约定:
      - x 为行驶方向坐标，y 为横向坐标
      - assume_left_is_larger_y=True 表示 “y 大的一侧为左侧”
        如与你的数据系不一致，可改为 False 或在结果里互换左右。
    """
    # 1) 每条线按 x 递增
    lines = [_ensure_increasing_x(np.asarray(L, dtype=float)) for L in lane_lines if len(L) >= 2]

    if len(lines) < 2:
        return []

    # 2) 按横向位置排序，得到从右到左或从左到右的序列
    #    用每条线的 y-均值作为其横向位置代表
    y_means = np.array([L[:,1].mean() for L in lines])
    order = np.argsort(y_means)           # y 小在前，y 大在后
    lines_sorted = [lines[i] for i in order]

    # 如果 “y 大的是左侧”，则 lines_sorted[0] 是最右边界，[-1] 是最左边界
    # 若你的坐标系相反，把 assume_left_is_larger_y 设为 False 来翻转
    if not assume_left_is_larger_y:
        lines_sorted = lines_sorted[::-1]

    segments = []
    for i in range(len(lines_sorted) - 1):
        left  = lines_sorted[i+1]  # 更靠左的那条
        right = lines_sorted[i]    # 更靠右的那条

        # 3) 取两条线在 x 上的重叠区间，避免外推
        x_min = max(left[0,0],  right[0,0])
        x_max = min(left[-1,0], right[-1,0])
        if x_max <= x_min:
            # 没有重叠可用，跳过或按需要改成外推
            continue

        # 4) 在重叠区间上按 step 生成统一网格（每 1 m 一个点，默认）
        m = int(np.floor((x_max - x_min) / step)) + 1
        x_grid = x_min + np.arange(m) * step

        # 5) 插值出左右边界 & 计算中心线
        left_i  = _interp_on_grid(left,  x_grid)   # [m,2]
        right_i = _interp_on_grid(right, x_grid)   # [m,2]
        center  = np.stack([x_grid, 0.5*(left_i[:,1] + right_i[:,1])], axis=1)
        segments.append(np.stack([center, left_i, right_i], axis=0))
        # segments.append({
        #     'center': center,
        #     'left': left_i,
        #     'right': right_i,
        # })
    return np.array(segments)
