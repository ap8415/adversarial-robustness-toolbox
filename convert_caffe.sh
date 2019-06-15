#PBS -l nodes=gpu15 -l mem=6GB -l gpu_mem=5GB

cd /vol/gpudata/ap8415/adversarial-robustness-toolbox 

. /vol/gpudata/cuda/9.0.176/setup.sh
export PATH=/vol/cuda/9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/9.0.176/lib64:/vol/cuda/9.0.176/lib:$LD_LIBRARY_PATH

source activate venv

python3 caffe_weight_converter/caffe_weight_converter.py \
   'slimnet' \
   'caffe_models/lenet5/lenet5.prototxt' \
   'caffe_models/lenet5/slimnet_fx_310k_iter_39000.caffemodel' --verbose

