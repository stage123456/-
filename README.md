# 文本分类项目：邮件分类

## 项目简介
本项目使用多项式朴素贝叶斯分类器对邮件进行分类，支持两种特征提取模式：高频词特征选择和TF-IDF特征加权。通过切换特征模式，可以比较不同特征提取方法对分类效果的影响。

## 核心代码实现

### 1. 文本预处理
```python
def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            # 过滤无效字符
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            # 使用jieba.cut()方法对文本切词处理
            line = cut(line)
            # 过滤长度为1的词
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words
```

### 2. 特征提取实现

#### 高频词特征选择
```python
def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    # 遍历邮件建立词库
    for filename in filename_list:
        all_words.append(get_words(filename))
    # 统计词频并返回top_num个高频词
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]
```

#### TF-IDF特征加权
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(documents, max_features=100):
    """使用TF-IDF提取特征"""
    vectorizer = TfidfVectorizer(max_features=max_features)
    return vectorizer.fit_transform(documents)
```

### 3. 分类器实现
```python
from sklearn.naive_bayes import MultinomialNB

def train_classifier(features, labels):
    """训练多项式朴素贝叶斯分类器"""
    model = MultinomialNB()
    model.fit(features, labels)
    return model
```

## 特征模式切换方法

### 1. 模式切换实现
```python
def switch_feature_mode(mode='high_freq', top_num=100):
    """
    切换特征提取模式
    :param mode: 'high_freq' 或 'tfidf'
    :param top_num: 特征数量
    :return: 特征向量
    """
    if mode == 'high_freq':
        # 高频词特征选择
        top_words = get_top_words(top_num)
        features = []
        for words in all_words:
            word_map = list(map(lambda word: words.count(word), top_words))
            features.append(word_map)
        return np.array(features)
    else:
        # TF-IDF特征加权
        documents = [' '.join(words) for words in all_words]
        return get_tfidf_features(documents, max_features=top_num)
```

### 2. 使用示例
```python
# 使用高频词特征
features_high_freq = switch_feature_mode(mode='high_freq', top_num=100)
model_high_freq = train_classifier(features_high_freq, labels)

# 使用TF-IDF特征
features_tfidf = switch_feature_mode(mode='tfidf', top_num=100)
model_tfidf = train_classifier(features_tfidf, labels)
```

## 特征模式对比

### 1. 高频词特征选择
- **优点**：
  - 实现简单，计算效率高
  - 对于简单文本分类任务效果良好
- **缺点**：
  - 忽略了词的区分能力
  - 无法处理词的权重问题

### 2. TF-IDF特征加权
- **优点**：
  - 考虑了词的区分能力
  - 能够更好地衡量词的重要性
- **缺点**：
  - 计算复杂度较高
  - 需要额外的平滑处理

## 使用说明

1. 准备数据：
   - 将邮件文本文件放在 `邮件_files` 目录下
   - 确保文件名格式为 `{序号}.txt`

2. 选择特征模式：
   - 高频词特征：`mode='high_freq'`
   - TF-IDF特征：`mode='tfidf'`

3. 训练模型：
   ```python
   # 设置特征数量
   top_num = 100
   
   # 选择特征模式并训练
   features = switch_feature_mode(mode='high_freq', top_num=top_num)
   model = train_classifier(features, labels)
   ```

4. 预测新邮件：
   ```python
   def predict(filename, model, mode='high_freq'):
       words = get_words(filename)
       if mode == 'high_freq':
           current_vector = np.array(tuple(map(lambda word: words.count(word), top_words)))
       else:
           current_vector = get_tfidf_features([' '.join(words)], max_features=len(top_words))
       return '垃圾邮件' if model.predict(current_vector.reshape(1, -1)) == 1 else '普通邮件'
   ```
   ---
<img src="https://github.com/stage123456/-/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-06-19%20014011.png" width="200" alt="截图">
