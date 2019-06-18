#PBS -l nodes=gpu02 -l mem=6GB -l gpu_mem=3GB

cd /vol/gpudata/ap8415/adversarial-robustness-toolbox 

. /vol/gpudata/cuda/9.0.176/setup.sh
export PATH=/vol/cuda/9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/9.0.176/lib64:/vol/cuda/9.0.176/lib:$LD_LIBRARY_PATH

source activate venv

echo "\n\n\n MNIST VGG POOLING DROPOUT"
python3 foolbox_exp/pooling_vs_fc.py VGG pooling
echo "\n\n\n MNIST VGG DENSE DROPOUT"
python3 foolbox_exp/pooling_vs_fc.py VGG dense
echo "\n\n\n MNIST leNet5 POOLING DROPOUT"
python3 foolbox_exp/pooling_vs_fc.py leNet5 pooling
echo "\n\n\n MNIST leNet5 DENSE DROPOUT"
python3 foolbox_exp/pooling_vs_fc.py leNet5 dense
