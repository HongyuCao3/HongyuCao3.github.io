# nltk的词性标注是否能用于bart模型的标注

## 本文解决的问题具体描述
如标题所示，这一问题产生的基本原因：nltk库中的分词函数`pos_tag`需要使用nltk自带的`word_tokenzie`进行分词。上述分词器和深度学习模型中的分词器词表不一致，本文探索对深度学习模型的分词结果使用`pos_tag`进行标注是否会报错。

## 解决问题使用的代码和测试结果
 - 进行测试的代码
    ```python
    import nltk
    from nltk.tag import pos_tag
    from nltk.tokenize import word_tokenize
    from transformers import BartTokenizer

    text = "And now for something completely different"
    print("nltk tokenize : ")
    print(pos_tag(word_tokenize(text)))
    print("Bart tokenize : ")
    bart_tok = BartTokenizer.from_pretrained("facebook/bart-base")
    tok_res = bart_tok.tokenize(text)
    print(pos_tag(tok_res)) # 也能用
    ```
 - 输出结果
    ```
    nltk tokenize : 
    [('And', 'CC'), ('now', 'RB'), ('for', 'IN'), ('something', 'NN'), ('completely', 'RB'), ('different', 'JJ')]
    Bart tokenize : 
    [('And', 'CC'), ('Ġnow', 'VB'), ('Ġfor', 'JJ'), ('Ġsomething', 'VBG'), ('Ġcompletely', 'RB'), ('Ġdifferent', 'JJ')]
    ```
 - 分析
    - 可以明显的看出两个tokenizer得到的分词结果不同，bart分词的结果有奇怪的字符`Ġ`出现，具体原因有待进一步探索。
    - 分词之后的词性标注不完全相同，如果希望使用那么可能需要<font color = red>根据nltk分词标注的span去获得深度学习模型分词的结果</font>。

## 可能遇到的其他问题及解决方案
 - nltk在运行时需要下载语料，如果没有下载会有报错信息提示使用语句如`nltk.download('punkt')`解决问题。如上述语句仍不能解决问题，google搜索“nltk download失败”参考其他博客内容。