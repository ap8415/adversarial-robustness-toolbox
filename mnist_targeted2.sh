#PBS -l nodes=gpu01 -l mem=6GB -l gpu_mem=3GB

cd /vol/gpudata/ap8415/adversarial-robustness-toolbox

. /vol/gpudata/cuda/9.0.176/setup.sh
export PATH=/vol/cuda/9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/9.0.176/lib64:/vol/cuda/9.0.176/lib:$LD_LIBRARY_PATH

source activate venv

echo "\n\n\n FIRST EXPERIMENT\n\n"
python3 foolbox_exp/experiment_targeted.py five_layer_dnn l2

