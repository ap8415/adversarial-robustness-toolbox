#PBS -l nodes=gpu02 -l mem=8GB -l gpu_mem=8GB

cd /vol/gpudata/ap8415/adversarial-robustness-toolbox 

. /vol/gpudata/cuda/9.0.176/setup.sh
export PATH=/vol/cuda/9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/9.0.176/lib64:/vol/cuda/9.0.176/lib:$LD_LIBRARY_PATH

source activate venv

python3 experiments/dropout.py two_layer_dnn carlini_lInf
python3 experiments/dropout.py three_layer_dnn carlini_lInf
python3 experiments/dropout.py five_layer_dnn carlini_lInf
python3 experiments/dropout.py six_layer_dnn carlini_lInf
python3 experiments/dropout.py simple_cnn carlini_lInf
python3 experiments/dropout.py leNet5 carlini_lInf

