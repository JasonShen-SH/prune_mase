# Pruning and Training for MASE - Group 3

## Overall Pipeline

Our project has developed a ”trainable” pruning method which could automatically prunes and compresses the network, while achieving comparable performance after training.

The architecture of our framework is segmented into four key components: **Pruning, Quantization, Training, and Huffman Coding**。

Each component is executed through an autonomous pass within <code>transform.py</code>, allowing for the flexible selection and combination of passes to suit specific requirements.

Outlined below is our pipeline:

<img src="imgs/overall_pipeline.png" width=800>

&nbsp;&nbsp;

## Getting started with the experiments

Please execute all of our programs in the **machop("mase/machop")** directory.

Our test function is **"test_group3.py"** inside the existing testing framework, run in command line using:
```yaml
python test/passes/graph/transforms/prune/test_group3.py
```

You can also execute the transform function via the command line using 
```yaml
./ch transform --config configs/examples/group3.toml
```

You might change configuration as you wish. 

As there are too many configurations, we kept them inside toml file at <code>configs/example/prune_retrain_group3.toml</code>
Please refer to the file for default parameter values and to change them.

Below is a demonstration of an actual output:
```yaml
# Pruning:
pass_config:
{'weight': {'sparsity': 0.2, 'scope': 'local', 'granularity': 'elementwise', 'method': 'l1-norm'}, 'activation': {'sparsity': 0.1, 'scope': 'local', 'granularity': 'elementwise', 'method': 'l1-norm'}}
-------------------------------------
number of Conv2d parameters before pruning:  4576384
model size before pruning:  18320936.0
flop of Conv2d layers before pruning:  1215037440
-------------------------------------
number of Conv2d parameters after pruning:  3659670
model size after pruning:  14661248.0
flop of Conv2d layers after pruning:  541312576
-------------------------------------
reduced percentage of Conv2d parameters:  0.20031404707297285
reduced percentage of model size:  0.19975442302729507
reduced percentage of Conv2d flops:  0.5544889744302859
-------------------------------------
INFO     model is successfully pruned and saved!

# Quantization:
There is quantization at feature_layers_0, mase_op: conv2d
There is quantization at feature_layers_3, mase_op: conv2d
There is quantization at feature_layers_7, mase_op: conv2d
There is quantization at feature_layers_10, mase_op: conv2d
There is quantization at feature_layers_14, mase_op: conv2d
There is quantization at feature_layers_17, mase_op: conv2d
model size after quantization:  3665312.0
INFO     model is successfully quantized and saved

# Fine-tuning:
INFO     Loaded pytorch checkpoint from ../mase_output/vgg_cifar10_prune/software/transforms/prune/state_dict.pt
Epoch 0:   1%|▍                                              | 1/98 [00:27<44:02,  0.04it/s, v_num=0, train_acc_step=0.885]

# Huffman coding
huffman used bytes:  1344395.25
INFO     Transformation is completed
```


&nbsp;&nbsp;

## Pruning Methods

Specifically, below are all the pruning methods that we've implemented:

Weight pruning:

<img src="imgs/weight_wise.png" width=500>


Activation pruning:

<img src="imgs/activation_pruning.png" width=500>

Please refer to **pruning_methods.py** for their specifc names. 

For the detailed analysis on their principles and performance, as well as the multiple evaluating metrics, please refer to the report.

&nbsp;&nbsp;

## Post-prune Quantization & Huffman Coding

Additionally, inspired by the methodology from[DEEP COMPRESSION: COMPRESSING DEEP NEURAL NETWORKS WITH PRUNING, TRAINED QUANTIZATION AND HUFFMAN CODING](https://arxiv.org/pdf/1510.00149.pdf), we've implemented **post-prune quantization** and **Huffman Coding**. 

By default, these two further model compression techniques are enabled, but you can choose to disable them by commenting <code>passes.quantize</code> and set <code>is_huffman = false</code>

Note that quantization must be valid for Huffman encoding.

&nbsp;&nbsp;

## Train from scratch && Transferability to other models and datasets

By default, the model loads the **pre-trained VGG7 model** for pruning and training.

If desired, you can opt to **train from scratch** by setting <code>load_name = None</code>.

Moreover, you are free to select different datasets and models. The **ResNet18** network and **colored-MNIST** are fully compatible with our pipeline and yield satisfactory results. To utilize these, please modify the toml configuration as follows:
```yaml
model = "resnet18"  # ResNet18
dataset = "mnist"  # colored-MNIST
```

&nbsp;&nbsp;

## Contact

Feel free to contact us at ruiqi.shen23@imperial.ac.uk if you have any encountered problems.


