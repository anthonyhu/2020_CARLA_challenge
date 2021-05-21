import os
import numpy as np
import cv2
import torch
import carla

from PIL import Image, ImageDraw

from world_model.trainer import WorldModelTrainer
from carla_project.src.converter import Converter
from carla_project.src.dataset import preprocess_semantic
from carla_project.src.common import COLOR

from team_code.map_agent import MapAgent
from team_code.pid_controller import PIDController


DEBUG = int(os.environ.get('HAS_DISPLAY', 0))


def get_entry_point():
    return 'WorldModelAgent'


def debug_display(tick_data, bev, next_state, steer, throttle, brake, desired_speed, step):
    _rgb = Image.fromarray(tick_data['rgb'])
    _draw_rgb = ImageDraw.Draw(_rgb)

    _combined = np.hstack([_rgb, np.zeros_like(_rgb), np.zeros_like(_rgb)])
    #np.hstack([tick_data['rgb_left'], _rgb, tick_data['rgb_right']])

    bev_resized = COLOR[np.argmax(bev, axis=0)]
    bev_h, bev_w = bev_resized.shape[:2]

    # next state
    next_state_plot = COLOR[np.argmax(next_state, axis=0)]
    # height = _combined.shape[0]
    # width = int(bev_w * height / bev_h)
    # bev_resized = bev_resized.resize((width, height), resample=Image.NEAREST)

    bev_filler = np.zeros((bev_h, _combined.shape[1], 3), dtype=np.uint8)
    bev_filler[:, :bev_w] = bev_resized
    bev_filler[:, bev_w:2*bev_w] = next_state_plot
    _combined = np.vstack([_combined, bev_filler])
    _combined = Image.fromarray(_combined)
    _draw = ImageDraw.Draw(_combined)
    _draw.text((5, 10), 'Steer: %.3f' % steer)
    _draw.text((5, 30), 'Throttle: %.3f' % throttle)
    _draw.text((5, 50), 'Brake: %s' % brake)
    _draw.text((5, 70), 'Speed: %.3f' % tick_data['speed'])
    _draw.text((5, 90), 'Desired: %.3f' % desired_speed)

    cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)


class WorldModelAgent(MapAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self.converter = Converter()
        self.world_model = WorldModelTrainer.load_from_checkpoint(path_to_conf_file)
        self.world_model.cuda()
        self.world_model.eval()

    def _init(self):
        super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def tick(self, input_data):
        result = super().tick(input_data)
        result['image'] = result['rgb']#np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)

        theta = result['compass']
        theta = 0.0 if np.isnan(theta) else theta
        theta = theta + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        gps = self._get_position(result)
        far_node, _ = self._command_planner.run_step(gps)
        target = R.T.dot(far_node - gps)
        target *= 5.5
        target += [128, 256]
        target = np.clip(target, 0, 256)

        result['target'] = target

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        # img = torchvision.transforms.functional.to_tensor(tick_data['image'])
        # img = img[None].cuda()

        # preprocess input
        bev = Image.fromarray(tick_data['topdown'])
        bev = bev.crop((128, 0, 128 + 256, 256))
        bev = np.array(bev)
        bev = preprocess_semantic(bev)
        bev = bev.unsqueeze(0).cuda()

        action = self.world_model.policy(bev)

        next_state = self.world_model.transition_model(bev, action)

        predicted_steering = action[0, 0].item()
        predicted_speed = action[0, 1].item()

        steer = predicted_steering# self._turn_controller.step(predicted_steering)
        steer = np.clip(steer, -1.0, 1.0)

        speed = tick_data['speed']
        brake = predicted_speed < 0.1 or (speed / predicted_speed) > 1.2

        delta = np.clip(predicted_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        if DEBUG:
            debug_display(
                    tick_data, bev[0].cpu().numpy(), next_state[0].cpu().numpy(),
                    steer, throttle, brake, predicted_speed,
                    self.step)

        return control
