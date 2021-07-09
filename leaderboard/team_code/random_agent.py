import numpy as np

from team_code.auto_pilot import AutoPilot


def get_entry_point():
    return 'RandomAgent'


class RandomAgent(AutoPilot):
    def _get_control(self, target, far_target, tick_data, _draw):
        # pos = self._get_position(tick_data)
        # theta = tick_data['compass']
        # speed = tick_data['speed']
        #
        # # Steering.
        # angle_unnorm = self._get_angle_to(pos, theta, target)
        # _ = self._should_brake()

        target_speed = 0.0

        steer = np.random.uniform(-1, 1)
        throttle = np.random.uniform(-1, 1)
        brake = bool(np.random.binomial(1, 0.05))

        if brake:
            throttle = 0.0

        return steer, throttle, brake, target_speed
