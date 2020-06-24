Sentence Matching With Deep Self-Attention and Co-Attention Features

# Abstract
Sentence matching refers to extracting the logical and semantic relations between two sentences which is widely applied in many natural language processing tasks such as natural language inference, paraphrase identification, and question answering. However, many previous methods simply use a siamese network to capture the semantic feature and apply attention mechanism to align the semantic feature of two sentences. In this paper, we propose a deep and effective neural network based on attention mechanism to learn richer sematic feature and align feature of two sentences, each layer include two sublayers sematic encoder and aligned encoder.  of which one uses a self-attention network for the semantic feature and another one uses a cross-attention network for the align feature. Experiments on three benchmark datasets prove that self-attention network and cross-attention network can efficiently learn the sematic and align feature of two sentences, which helps our method achieve state-of-the-art results.

# 1 Introduction



# 2 Our Apporach
	 
Figure 1 
In this section, we introduce our proposed approach DAMM for sentence matching. Figure 1 gives an illustration of the overall architecture. 总体描述：巴拉巴拉

## 2.1 Embedding Layer
The goal of embedding layer is to represent each word to a d-dimensional vector by using a pre-trained vector such as GLove , Word2Vec and Fasttext. In our model, we use Glove vector to get the fixed vector for P and Q.

## 2.2 Self-Encoder Layer
In Self-Encoder Layer, 

## 2.3 Cross-Encoder Layer

## 2.4 Pooling Layer

## 2.5 Predict Layer

# 3 Experiments
## 3.1 Datasets
## 3.2 Implementation Details
## 3.3 消融实验
## 3.4 Analysis

# 4 Related Work

# 5 Conclusion

