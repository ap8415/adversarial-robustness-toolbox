#PBS -l nodes=gpu15 -l mem=6GB -l gpu_mem=5GB

cd /vol/gpudata/ap8415/adversarial-robustness-toolbox 

. /vol/gpudata/cuda/9.0.176/setup.sh
export PATH=/vol/cuda/9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/9.0.176/lib64:/vol/cuda/9.0.176/lib:$LD_LIBRARY_PATH

source activate venv

python3 caffe_weight_converter/caffe_weight_converter.py \
   'lenet_300_100' \
   'caffe_models/lenet300100/lenet300100.prototxt' \
   'caffe_models/lenet5300100/caffe_lenet5_sparse.caffemodel' --verbose

