import os
import numpy as np
import cv2
import torch
import carla

from PIL import Image, ImageDraw

from world_model.trainer import WorldModelTrainer
from world_model.utils import preprocess_bev_state
from carla_project.src.converter import Converter
from carla_project.src.common import COLOR

from team_code.map_agent import MapAgent
from team_code.pid_controller import PIDController


DEBUG = int(os.environ.get('HAS_DISPLAY', 0))


def get_entry_point():
    return 'WorldModelAgent'


def debug_display(tick_data, bev, next_state, steer, throttle, brake, desired_speed, command_name, step):
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
    _draw.text((5, 110), f'Command {command_name}')

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

        #self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=10) #n=40

    def tick(self, input_data):
        result = super().tick(input_data)
        #result['image'] = result['rgb'] #np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)
        gps = self._get_position(result)

        # Route command
        near_node, near_command = self._waypoint_planner.run_step(gps)
        result['command'] = near_command

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
        bev = preprocess_bev_state(bev).unsqueeze(0).cuda()
        # route command
        route_command = torch.LongTensor([tick_data['command'].value]).unsqueeze(0).cuda()

        action = self.world_model.policy(bev, route_command)

        if self.world_model.config.MODEL.TRANSITION.ENABLED:
            next_state = self.world_model.transition_model(bev, action)
        else:
            next_state = torch.zeros_like(bev)

        predicted_steering = action[0, 0].item()
        desired_speed = action[0, 1].item()

        steer = predicted_steering# self._turn_controller.step(predicted_steering)
        steer = np.clip(steer, -1.0, 1.0)

        speed = tick_data['speed']
        brake = desired_speed < 0.1 or (speed / desired_speed) > 1.2

        delta = desired_speed - speed
        #delta = np.clip(delta, 0.0, 0.25)
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
                    steer, throttle, brake, desired_speed, tick_data['command'].name,
                    self.step)

        return control
