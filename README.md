To use Jacinle python library, 

export PATH=<path_to_jacinle>/bin:$PATH
jac-run xxx.py to replace python3 xxx.py. 

You can also use the jac-crun <gpu_ids> xxx.py to set the gpus you want to use. Here, <gpu_ids> is a comma-separated list of gpu ids, following the convension of CUDA_VISIBLE_DEVICES.
