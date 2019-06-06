#PBS -l nodes=gpu02 -l mem=8GB -l gpu_mem=8GB

cd /vol/gpudata/ap8415/adversarial-robustness-toolbox 

. /vol/gpudata/cuda/9.0.176/setup.sh
export PATH=/vol/cuda/9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/9.0.176/lib64:/vol/cuda/9.0.176/lib:$LD_LIBRARY_PATH

source activate venv

python3 experiments/two_layer_dnn/dropout.py
python3 experiments/three_layer_dnn/dropout.py
python3 experiments/six_layer_dnn/dropout.py
python3 experiments/five_layer_dnn/dropout.py
python3 experiments/simple_cnn/dropout.py
python3 experiments/leNet5/dropout_fc.py
python3 experiments/leNet5/dropout_full.py
python3 experiments/leNet5/dropout_pooling.py


