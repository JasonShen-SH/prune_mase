We have developed a "trainable" pruning pipeline which could automatically prunes and compresses the network, while achieving comparable performance after training.

It mainly consists of four parts: pruning, quantization, training, and Huffman Coding. Each part is implemented by an independent pass in the transform, which makes the pruning process flexible as we can select and combine passes as needed.
以下是我们的pipeline:
<img src="imgs/overall_pipeline.png" width=400>

Please execute all of our programs in the **machop(i.e."mase/machop")** directory.

Our test function is **"test_group3.py"** inside the existing testing framework,您可以直接在程序的config中选择实验配置，

您同时也可以通过命令行来执行transform函数，这需要您找到配置文件并修改，它位于"configs/example/prune_retrain_group3.toml"
参数的默认值请于toml文件中查看

根据pruning的对象,我们将其分类为weight pruning和activation pruning,以下是它们的所有methods:
weight pruning:
<img src="imgs/weight_wise.png" width=400>

Activation pruning:
<img src="imgs/activation_pruning.png" width=400>

您可随意选择，选择请于**pruning_methods


