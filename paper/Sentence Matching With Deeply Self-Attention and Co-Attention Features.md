Sentence Matching With Deep Self-Attention and Co-Attention Features

# Abstract
Sentence matching refers to extracting the logical and semantic relations between two sentences which is widely applied in many natural language processing tasks such as natural language inference, paraphrase identification, and question answering. However, many previous methods simply use a siamese network to capture the semantic feature and apply attention mechanism to align the semantic feature of two sentences. In this paper, we propose a deep and effective neural network based on attention mechanism to learn richer sematic feature and align feature of two sentences, each layer include two sublayers sematic encoder and aligned encoder of which one uses a self-attention network for the semantic feature and another one uses a cross-attention network for the align feature. Experiments on three benchmark datasets prove that self-attention network and cross-attention network can efficiently learn the sematic and align feature of two sentences, which helps our method achieve state-of-the-art results.

# 1 Introduction




# 2 Related Work


# 3 Our Apporach
![](images/framework.png)	 


  In this section, we introduce our proposed sentence matching networks Deep Attention Matching Model (DAMM) which are composed of the following major components: embedding layer, encoder and alignment block, pooling layer, prediction layer. Figure 1 shows the overall architecture of our model. The input of model are two sentences as $a = (a_1, a_2, ..., a_I)$ with a length $I$ and $b = (b_1, b_2, ... , b_J)$ with a length $J$ where $a_i$ is the $i^{th}$ word of sentence $a$ and $b_j$ is the $j^{th}$ word of sentence $b$.  The sentence matching's goal is to give a label $y$ to represent the relationship be $a$ and $b$.
  
  In DAMM, each sentence are first embedded by the embedding layer into a matrix. And then, N same-structured blocks encoder the matrix. Each block has a self-attention encoder, cross-attention encoder and alignment layer. The output of last block is feeded into pooling layer to get the total represent of the whole sentence. Finally, DAMM use the two vector as input and predicts the final target. 
  



## 3.1 Embedding Layer

  The goal of embedding layer is to represent each word to a d-dimensional vector by using a pre-trained vector such as GloVe , Word2Vec and Fasttext. In our model, we use GloVe vector to get the fixed vector for $a$ and $b$ and the vector is fixed during training. 

## 3.2 Self-Attention Encoder
  In order to capture the richer sematic features of each sentence, DAMM employ a multi-head self-attention network. 

## 3.3 Cross-Attention Encoder Layer


## 3.4 Alignment Layer



## 3.5 Pooling Layer



## 3.6 Prediction Layer
$v_a$ and $v_b$ are the sentence $a$ and sentence $b$ vector represention from the output of the pooling layer. Prediction layer first integrate $v_a$ and $v_b$ into the final feature vector $v$ which is usful for a symmetric task:
$$v = [v_a; v_b; v_a-v_b; v_a * v_b]$$

Here, the operations $-$, $*$ are the element-wise subtraction and element-wise product, and $[;]$ is the concatenation. 

With the aggregated features $v$, we employ a two-layer feed-forward neural network for classification task and activation function is abdoted after first layer. We use cross entropy loss function to train our model.
$$ y = softmax(f(vw_1)w_2)$$

$$ Loss = Cross-Entropy() $$





# 4 Experiments
## 4.1 Datasets
## 4.2 Implementation Details
## 4.3 Ablation Study
## 4.4 Analysis

# 5 Result

# 6 Conclusion

