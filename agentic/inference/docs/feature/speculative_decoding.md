# 投机解码特性

## 特性说明

本方案分为3个部分，SAM（Suffix Automaton, 自动后缀机）投机解码、后缀树（Suffix Tree）投机解码、投机解码的自适应开关。

### SAM投机解码

SAM投机解码采用后缀自动机存储请求的上下文后缀，并用于投机解码中后缀的匹配和查询，论文：AM Decoding: Speculative Decoding via Suffix Automaton (https://arxiv.org/abs/2411.10666). 

SAM投机解码有2套实现方案，分别是采用Python实现的同步建图，和采用C++实现的异步建图方案。同步建图方案在第一次收到请求时建图，建图完成后做查询和投机；异步建图方案在收到请求后要求一个子线程异步建图，在建图过程中不开启投机，建图完成后再做投机解码。

异步建图功能需要单独编译`samfast`包，请进入`Pangu_infer/pangu_infer/pangu/accelerators/speculative_decoding/samfast`参考`README.md`单独编译。

### 后缀树投机解码

后缀树投机解码使用后缀树存储请求的上下文后缀，并用于投机解码中后缀的匹配和查询，论文：SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications (https://arxiv.org/abs/2411.04975).

这里实现的后缀树投机参考VLLM的后缀树投机解码接口，调用snowflake的ArcticInference (https://github.com/snowflakedb/ArcticInference)来做后缀树投机。需要通过`pip install arctic_inference`预先安装对应的python包。

### 投机解码的自适应开关

受限于算子性能，目前的n-gram/SAM投机解码只在一定的batch size以下，和一定的序列长度以下才有性能增益。因此，我们设置了基于batch size的自适应开关，在batch size小于用户指定的阈值时开启投机解码；如果batch size大于阈值，就自动关掉投机解码，回退到非投机解码。

## 启动脚本

参考`Pangu_infer/scripts/pangu_7b/serving_pangu_7b.sh`，需要在`additional_config`中添加`speculative_decoding`如下：

```shell
"speculative_decoding": {
"enabled": false,
"method": "suffix",
"num_speculative_tokens": 3,
"asynchronous_sam": false,
"suffix_max_tree_depth": 64,
"suffix_max_spec_factor": 4,
"suffix_min_token_prob": 0.1,
"suffix_max_model_len": 4000,
"suffix_max_cached_requests": 10000,
"batch_size_auto_switch": false,
"batch_size_auto_switch_thre": 8
}
```

另外需要在vllm启动的选项中加入`--speculative-config '{"method": "ngram", "num_speculative_tokens": NUM_SPEC_TOKENS}'`，其中`NUM_SPEC_TOKENS`是投机的token数，否则会触发vllm/v1/spec_decode/metrics.py中的assertion报错。

### 变量解释
- "enabled": `bool`, 开启投机解码功能，默认为False；
- "method": `str`, 投机解码方法，可以在`["sam", "suffix"]`中选择，默认为"sam"；
- "num_speculative_tokens": `int`，投机的token个数，默认为3；

#### SAM投机解码相关变量
- "asynchronous_sam": `bool`，开启异步SAM建图功能，默认为False；

#### 后缀树投机解码相关变量
所有后缀树投机相关变量都以"suffix"为前缀。
- "suffix_max_tree_depth": `int`，后缀树的最大深度，默认为64；
- "suffix_max_spec_factor" `int`，后缀树投机中的因子$\alpha$，调控最长返回序列的参数，返回的最长序列长度是$MAX\_SPEC=\alpha\cdot p$，其中`p`是查询后缀的长度。在Pangu_infer中返回序列长度也可以通过"num_speculative_tokens"调节；
- "suffix_min_token_prob" `float`，接受token的最小概率阈值，默认值是0.1，参考后缀树论文；
- "suffix_max_model_len": `int`，后缀树存储的最大模型长度，默认值是4000；
- "suffix_max_cached_requests": `int`，最多存储的请求个数，默认值是10000.

#### 投机解码自适应开关的相关变量
- "batch_size_auto_switch": `bool`，开启batch size的自适应开关，默认为False；
- "batch_size_auto_switch_thre": `int`，batch size自适应开关的阈值，默认为1024.
