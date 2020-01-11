# command to run: 
# nohup ./collect_results_MNIST.sh 2>&1 &

# evaluation of single methods
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.0 -m Base -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.0 -m Drop -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.0 -m CDrop -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.0 -m LLCDrop -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.0 -m VI -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.0 -m LLVI -c NC

CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.0 -m Base -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.0 -m Drop -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.0 -m CDrop -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.0 -m LLCDrop -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.0 -m VI -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.0 -m LLVI -c NC

# LS + methods
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.1 -m Base -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.1 -m Drop -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.1 -m CDrop -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.1 -m LLCDrop -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.1 -m LLVI -c NC

CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.1 -m Base -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.1 -m Drop -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.1 -m CDrop -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.1 -m LLCDrop -c NC
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.1 -m LLVI -c NC

# methods + TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.0 -m Base -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.0 -m Drop -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.0 -m CDrop -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.0 -m LLCDrop -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.0 -m LLVI -c TS

CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.0 -m Base -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.0 -m Drop -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.0 -m CDrop -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.0 -m LLCDrop -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.0 -m LLVI -c TS

# LS + methods + TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.1 -m Base -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.1 -m Drop -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.1 -m CDrop -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.1 -m LLCDrop -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a LeNet -s 0.1 -m LLVI -c TS

CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.1 -m Base -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.1 -m Drop -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.1 -m CDrop -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.1 -m LLCDrop -c TS
CUDA_VISIBLE_DEVICES=1 python framework.py -d MNIST -a ResNet -s 0.1 -m LLVI -c TS
