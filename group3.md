We have developed a "trainable" pruning pipeline which could automatically prunes and compresses the network, while achieving comparable performance after training.

It mainly consists of four parts: **pruning, quantization, training, and Huffman Coding**. 

Each part is implemented by an independent pass in the transform, you can flexibly select and combine passes as needed.

The following is our pipeline:

<img src="imgs/overall_pipeline.png" width=800>

&nbsp;&nbsp;

Please execute all of our programs in the **machop(i.e."mase/machop")** directory.


Our test function is **"test_group3.py"** inside the existing testing framework, run in command line using:
```yaml
python test/passes/graph/transforms/prune/test_group3.py
```

You can also execute the transform function via the command line using 
```yaml
./ch transform --config configs/examples/vgg_cifar10_prune_retrain.toml
```

You might change configuration as you wish. As there are too many configurations, we kept them inside toml file at <code>configs/example/prune_retrain_group3.toml</code>
Please refer to the file for default parameter values and to change them.

&nbsp;&nbsp;

Specifically, below are all the pruning methods that we've implemented:

weight pruning:

<img src="imgs/weight_wise.png" width=500>


Activation pruning:

<img src="imgs/activation_pruning.png" width=500>

Please refer to **pruning_methods.py** for their specifc names. For the detailed analysis on their principles and performance, please refer to the report.


&nbsp;&nbsp;

Additionally, inspired by the methodology from[DEEP COMPRESSION: COMPRESSING DEEP NEURAL NETWORKS WITH PRUNING, TRAINED QUANTIZATION AND HUFFMAN CODING](https://arxiv.org/pdf/1510.00149.pdf), we've implemented **post-prune quantization** and **Huffman Encoding & Decoding**. 

By default, these two further model compression techniques are enabled, but you can choose to disable them by commenting <code>passes.quantize</code> and set <code>is_huffman = false</code>

Note that quantization must be valid for Huffman encoding.

&nbsp;&nbsp;

By default, the model loads the **pre-trained VGG7 model** for pruning and training.

If desired, you can opt to **train from scratch** by setting <code>load_name = None</code>.

Moreover, you are free to select different datasets and models. The **ResNet18** network and **colored-MNIST** are fully compatible with our pipeline and yield satisfactory results. To utilize these, please modify the toml configuration as follows:
```yaml
model = "resnet18"  # ResNet18
dataset = "mnist"  # colored-MNIST
```



