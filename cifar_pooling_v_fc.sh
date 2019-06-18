#PBS -l nodes=gpu01 -l mem=8GB -l gpu_mem=8GB

cd /vol/gpudata/ap8415/adversarial-robustness-toolbox 

. /vol/gpudata/cuda/9.0.176/setup.sh
export PATH=/vol/cuda/9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/9.0.176/lib64:/vol/cuda/9.0.176/lib:$LD_LIBRARY_PATH

source activate venv

echo "\n\n\n CIFAR10 VGG POOLING DROPOUT"
python3 foolbox_exp/cifar_pooling_vs_fc.py pooling
echo "\n\n\n CIFAR10 VGG DENSE DROPOUT"
python3 foolbox_exp/cifar_pooling_vs_fc.py dense



