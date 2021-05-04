import os
import sys
sys.path.append('/homes/ah2029/CARLA/PythonAPI/carla')
sys.path.append('/homes/ah2029/CARLA/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg')
sys.path.append('leaderboard')
sys.path.append('leaderboard/team_code')
sys.path.append('scenario_runner')

from argparse import ArgumentParser, RawTextHelpFormatter

from leaderboard.leaderboard_evaluator import run


if __name__ == '__main__':
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='8000',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="60.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        required=True)
    # TODO check what this flag does
    parser.add_argument('--scenarios', default='leaderboard/data/all_towns_traffic_scenarios_public.json',
                        help='Name of the scenario annotation file to be mixed with the route.')
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")

    arguments = parser.parse_args()

    checkpoint_file = os.path.join(os.path.dirname(arguments.agent_config),
                                   os.path.basename(arguments.routes).split('.')[0] + '.txt')
    arguments.checkpoint = checkpoint_file
    run(arguments)
