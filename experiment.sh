#PBS -l nodes=gpu02 -l mem=8GB -l gpu_mem=8GB

cd /vol/gpudata/ap8415/adversarial-robustness-toolbox 

. /vol/gpudata/cuda/9.0.176/setup.sh
export PATH=/vol/cuda/9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/9.0.176/lib64:/vol/cuda/9.0.176/lib:$LD_LIBRARY_PATH

source activate venv

echo "\n\n\n FIVE LAYER DNN EXPERIMENT:\n\n"

python3 experiments/early_vs_late_dropout/dropout_five_layer_dnn.py

echo "\n\n\n SIX LAYER DNN EXPERIMENT:\n\n"

python3 experiments/early_vs_late_dropout/dropout_six_layer_dnn.py
