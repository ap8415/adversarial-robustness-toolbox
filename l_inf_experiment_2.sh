#PBS -l nodes=gpu06 -l mem=4GB -l gpu_mem=2GB

cd /vol/gpudata/ap8415/adversarial-robustness-toolbox 

. /vol/gpudata/cuda/9.0.176/setup.sh
export PATH=/vol/cuda/9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/9.0.176/lib64:/vol/cuda/9.0.176/lib:$LD_LIBRARY_PATH

source activate venv

python3 experiments/linf_bound_dropout.py three_layer_dnn 20
python3 experiments/linf_bound_dropout.py three_layer_dnn 25
python3 experiments/linf_bound_dropout.py three_layer_dnn 30
python3 experiments/linf_bound_dropout.py three_layer_dnn 35



