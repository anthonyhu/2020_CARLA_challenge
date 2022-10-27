#!/bin/bash
if [[ $# -ne 2 ]] ; then
    echo 'Need to specify routes and port'
    exit 1
fi

ROUTES=$1
PORT=$2
TM_PORT=$((PORT + 6000))
checkpoint_filename="${ROUTES#leaderboard/data/}"
checkpoint_filename="lbc_${checkpoint_filename%.*}_${PORT}.json"

echo "Evaluating without scenarios"
echo "Port ${PORT} and traffic port ${traffic_port}"
echo "Saving results in ${checkpoint_filename}"

export TEAM_AGENT=leaderboard/team_code/image_agent.py
export TEAM_CONFIG=weights/lbc.ckpt
export CARLA_ROOT=/home/anthony/softwares/CARLA_0.9.10.1

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export SCENARIOS=leaderboard/data/no_scenarios.json
export RESUME=False

python3 run_agent.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${checkpoint_filename} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}

