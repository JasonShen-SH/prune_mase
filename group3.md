# Pruning and Training for MASE

## Overall Pipeline

It mainly consists of four parts: **pruning, quantization, training, and Huffman Coding**. 

Each part is implemented by an independent pass in the transform, you can flexibly select and combine passes as needed.

The following is our pipeline:

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
./ch transform --config configs/examples/vgg_cifar10_prune_retrain.toml
```

You might change configuration as you wish. 

As there are too many configurations, we kept them inside toml file at <code>configs/example/prune_retrain_group3.toml</code>
Please refer to the file for default parameter values and to change them.

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


