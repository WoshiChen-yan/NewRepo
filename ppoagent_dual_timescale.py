import numpy as np

from ppoagent import PPOAgent


class PPOAgentDualTimeScale(PPOAgent):
    """
    PPOAgent additive variant for dual-timescale fusion.
    It keeps the exact base API and adds an optional trend-guided split adjustment.
    """

    def adjust_split_actions_by_trend(self, split_actions, trend_row, alpha):
        """
        Args:
            split_actions: list[int], len=num_dests
            trend_row: 1D risk array in [0,1], len=num_dests
            alpha: slow-stream influence factor in [0,1]
        Returns:
            adjusted split_actions list[int]
        """
        if split_actions is None:
            return split_actions

        if trend_row is None:
            return split_actions

        adjusted = list(split_actions)
        trend_row = np.asarray(trend_row, dtype=np.float32)

        for i, level in enumerate(adjusted):
            risk = float(trend_row[i]) if i < len(trend_row) else 0.0
            shift = int(round(max(0.0, min(1.0, alpha)) * max(0.0, min(1.0, risk)) * 2.0))
            adjusted[i] = int(min(4, max(0, int(level) + shift)))

        return adjusted
