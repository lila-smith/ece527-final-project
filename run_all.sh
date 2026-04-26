#!/usr/bin/env bash
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
    python3 -m pip install -r requirements.txt
    echo "Installed Python dependencies"
else
    echo "No virtual environment found, proceeding without activation"
fi
make all
./run.sh 100
./run.sh 300
./run.sh 1000
./run.sh 3000
./run.sh 10000
./run.sh 30000
./run.sh 100000
./run.sh 300000
./run.sh 1000000
