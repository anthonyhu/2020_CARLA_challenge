import numpy as np

from team_code.auto_pilot import AutoPilot

NOISE_LEVEL = 0.3  #0.02  #0.1


def get_entry_point():
    return 'NoisyAgent'


class NoisyAgent(AutoPilot):
    def _get_control(self, target, far_target, tick_data, _draw):
        steer, throttle, brake, target_speed = super()._get_control(target, far_target, tick_data, _draw)

        steer += NOISE_LEVEL * np.random.randn()
        steer = np.clip(steer, -1.0, 1.0)

        throttle += NOISE_LEVEL * np.random.randn()
        throttle = np.clip(throttle, 0.0, 1.0)

        # TODO: No noise on break for now

        return steer, throttle, brake, target_speed
