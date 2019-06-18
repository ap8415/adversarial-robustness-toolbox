#PBS -l nodes=gpu02 -l mem=6GB -l gpu_mem=3GB

cd /vol/gpudata/ap8415/adversarial-robustness-toolbox 

. /vol/gpudata/cuda/9.0.176/setup.sh
export PATH=/vol/cuda/9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/9.0.176/lib64:/vol/cuda/9.0.176/lib:$LD_LIBRARY_PATH

source activate venv

echo "\n\nEARLY DROPOUT 5 LAYER DNN\n\n\n"
python3 foolbox_exp/early_v_late.py five_layer_dnn early
echo "\n\nLATE DROPOUT 5 LAYER DNN\n\n\n"
python3 foolbox_exp/early_v_late.py five_layer_dnn late
echo "\n\nEARLY DROPOUT 6 LAYER DNN\n\n\n"
python3 foolbox_exp/early_v_late.py six_layer_dnn early
echo "\n\nLATE DROPOUT 6 LAYER DNN\n\n\n"
python3 foolbox_exp/early_v_late.py six_layer_dnn late
