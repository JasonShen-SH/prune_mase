We have developed a "trainable" pruning pipeline which could automatically prunes and compresses the network, while achieving comparable performance after training.

It mainly consists of four parts: pruning, quantization, training, and Huffman Coding. Each part is implemented by an independent pass in the transform, which makes the pruning process flexible as we can select and combine passes as needed.

Please execute all of our programs in the machop(i.e."mase/machop") directory.

Our test function is "test_group3.py" put inside the existing testing framework,您可以直接在程序的config中选择实验配置，
实验配置及其默认选项如下：
| metrics  | 列2标题 | 列3标题 |
| ------- | ------- | ------- |
| 单元格1 | 单元格2 | 单元格3 |
| 单元格4 | 单元格5 | 单元格6 |


您同时也可以通过命令行来执行transform函数，这需要您修改配置文件，位于config
