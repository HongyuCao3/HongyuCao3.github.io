# 使用nltk的标注获得深度学习模型分词的标注结果

## 问题描述
如标题所示，笔者希望能够解决nltk分词和深度学习模型分词不一致导致的直接使用`pos_tag`进行词性标注的不准确问题。

## 思路概述
总体来说，思路是将nltk分词的结果一一对应到深度学习模型的分词结果。

笔者通过`print`出深度学习模型的分词结果，得知其类型为`<class 'transformers.tokenization_utils_base.BatchEncoding'>`。这种类型有方法`char_to_token`，该方法可以根据单词在原文中的位置（原文中单词第一个字母的对应下标`char_idx`）找到分词之后的位置（分词得到的tokens的list中的下标`token_idx`）。使用上述方法可以实现思路。

## 代码实现
```python
from transformers import AutoTokenizer
from nltk import word_tokenize, pos_tag

ptm_path = "facebook/bart-base"
auto_tok = AutoTokenizer.from_pretrained(ptm_path)

text = "Fast forward about 20 years , and it 's fair to say he has done just that ."

auto_tok_res = auto_tok(text)
nltk_tok_res = word_tokenize(text)

nltk_tag = pos_tag(nltk_tok_res)
for word, tag in nltk_tag:
    char_idx = text.index(word)
    token_idx = auto_tok_res.char_to_token(char_idx)
    token = auto_tok_res.tokens()[token_idx]
    print(token + " " + tag)
```

## 结果分析
上述代码的输出如下所示，可以看到得到了深度学习模型分词结果和nltk分词结果的标注的对应关系
```
Fast RB
Ġforward RB
Ġabout IN
Ġ20 CD
Ġyears NNS
Ġ, ,
Ġand CC
Ġit PRP
Ġ' VBZ
Ġfair JJ
Ġto TO
Ġsay VB
Ġhe PRP
Ġhas VBZ
Ġdone VBN
Ġjust RB
Ġthat DT
Ġ. .
```
## 可能遇到的其他问题和解决方案
暂无