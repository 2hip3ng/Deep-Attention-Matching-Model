Sentence Matching With Deep Self-Attention and Co-Attention Features

# Abstract
Sentence matching refers to extracting the logical and semantic relations between two sentences which is widely applied in many natural language processing tasks such as natural language inference, paraphrase identification, and question answering. However, many previous methods simply use a siamese network to capture the semantic feature and apply attention mechanism to align the semantic feature of two sentences. In this paper, we propose a deep and effective neural network based on attention mechanism to learn richer sematic feature and align feature of two sentences, each layer include two sublayers sematic encoder and aligned encoder of which one uses a self-attention network for the semantic feature and another one uses a cross-attention network for the align feature. Experiments on three benchmark datasets prove that self-attention network and cross-attention network can efficiently learn the sematic and align feature of two sentences, which helps our method achieve state-of-the-art results.

*key words:* sentence matching, natural language processing, attention mechanism

# 1 Introduction

Sentence matching is a   **what **  task?









# 2 Related Work








# 3 Our Apporach
![framework](images/framework.png)	 


  In this section, we introduce our proposed sentence matching networks Deep Attention Matching Model (DAMM) which are composed of the following major components: embedding layer, encoder and alignment block, pooling layer, prediction layer. Figure 1 shows the overall architecture of our model. The input of model are two sentences as $a = (a_1, a_2, ..., a_I)$ with a length $I$ and $b = (b_1, b_2, ... , b_J)$ with a length $J$ where $a_i$ is the $i^{th}$ word of sentence $a$ and $b_j$ is the $j^{th}$ word of sentence $b$.  The sentence matching's goal is to give a label $y$ to represent the relationship be $a$ and $b$.

  In DAMM, each sentence are first embedded by the embedding layer into a matrix. And then, N same-structured blocks encoder the matrix. Each block has a self-attention encoder, cross-attention encoder and alignment layer. The output of last block is feeded into pooling layer to get the total represent of the whole sentence. Finally, DAMM use the two vector as input and predicts the final target. 




## 3.1 Embedding Layer

  The goal of embedding layer is to represent each token of the sentence to a d-dimensional vector by using a pre-trained vector such as GloVe , Word2Vec and Fasttext. In our model, we use GloVe vector (840B Glove) to get the fixed vector for $sentence \ a$ and $sentence \ b$ and the vector is fixed during training.  Now, we have $ sentence \ a $ representation $A \in R^{la * d}$ and $sentence \ b$ representation $B \in R^{lb * d}$, where $la$ refers to the max sequence length of $ sentence \ a$, $ lb $ refers to the max sequence length of $ sentence \ b $. 


## 3.2 Self-Attention Encoder
  In the Self-Attention Encoder, the $ sentence \ a$ representation $A$ and the $sentence \ b$ representation $B$ are passed through multi sublayer network which are composed of multi-head self-attention layer and feedforward layer to capture the richer sematic features of each sentence themselves. 


  First, the input of the multi-head self-attention network consists of queries matrix Q_i, keys matrix K_i and values matrix V_i which are respectively using a linear tranformer on the representation $A$. Then, the scaled dot-product attention is employed to compute the self-attention output. Finally, we concatenate the multi-head self-attention outputs and feed into a two layer feed-forward network with $gelu$ activation functions. Formulations for $H_B$ are similar and omitted here. This process is described by the following formulas:

  $$Q_i = AW_{i}^{Q}$$
  $$K_i = AW_{i}^{K}$$
  $$V_i = AW_{i}^{V}$$
  $$Att_i = softmax(\frac{Q_iK_i^T}{\sqrt{d_q}})V_i$$
  $$M_A = [Att_1;Att_2;...;Att_h] $$

 $$H_A = gelu(M_A W_1) W_2$$ 

where $h$ is number of the head of the multi-head self-attention network, $i$ is integer between 1 to $h$ , the projections are  parameter matrices $W_i^Q \in R^{d*d_q}$ , $W_i^K \in R^{d*d_k}$ , $W_i^V \in R^{d*d_v}$ , $W_1 \in R^{d*d'}$ , $W_2 \in R^{d'*d}$ ,  $[;...;]$denotes the concatenation operation .

## 3.3 Cross-Attention Encoder

  In a sentence matching  model, the sentences interaction features could be import as same as the sentences sematic features which are output of Self-Attention Encoder above. For the sentences interaction features, our model employ a Cross-Attention Encoder to capture. The Cross-Attention Encoder is the similar with the Self-Attention Encoder. We calulate the  interaction from $sentence \ a$  to $sentence \ b$  as following , we omitted the another dirction here:

$$Q_{i} = AW_{iA}^{Q}$$

 $$K_i = BW_{i}^{K}$$
  $$V_i = BW_{i}^{V}$$
  $$Att_i = softmax(\frac{Q_iK_i^T}{\sqrt{d_q}})V_i$$
  $$M_{B2A} = [Att_1;Att_2;...;Att_h] $$

 $$H_{B2A} = gelu(M_A W_1) W_2$$ 

where $H_{B2A}$ denote the interaction feature from $sentence \ a$ sematic feature $H_A$ to $sentence \ b$ sematic feature $H_B$ .


## 3.4 Alignment Layer

After the Self-Attention Encoder and Cross-Attention Encoder, we have two feature matrix $H_A$ and $H_{B2A}$ which respectively represent the $sentence \ a$ sematic matrix and interaction matirx. The two feature both are important component for a sentence matching task. Howerver, for a deep network and convince, we need to align the two matrix as following:

$$H_A = [H_A; H_{B2A}; H_A-H_{B2A}; H_A* H_{B2A}]$$

$$H_B = [H_B; H_{A2B}; H_B-H_{A2B}; H_B * H_{A2B}]$$



where $H_A \in R^{la* 4d}$ .





## 3.5 Pooling Layer
The pooling layer's goal is to converts the vectors $H_A$ and $H_B$ to fixed-length vector $v_a$ and $v_b$ which will be fed into prediction layer to classify. As we all know, both average and max pooling are useful strategies for sentence matching. We also consider that some key words in two sentences may have an important impact for the final classification, and TextCNN is a good way to extract key words features. Hence, we combine the max pooling strategy and TextCNN in our model. Our experiments show that this leads to significantly better results. Formulations for $v_b$ are similar and omitted here. This process is described by the following formulas:

$$v_{a}^{max} = \max\limits_{i=1}^{la} \ H_{A, i}$$

$$v_a^{cnn} = TextCNN(H_A)$$

$$v_a = [v_a^{max}; v_a^{cnn}]$$

where $TextCNN$ have a detail explanation in ##参考文献##, and [;] is a concatenate operation.




## 3.6 Prediction Layer
In our models, $v_a$ and $v_b$ are the $ sentence \ a $ and $ sentence \ b $ feature vectors from the output of the pooling layer. The prediction layer is to aggregate the $v_a$ and $v_b$ in a proper way, and then predict the label using a feed-forward neural network. We first aggregate $v_a$ and $v_b$ in various ways which are usful for a symmetric task as follows:
$$v = [v_a; v_b; v_a-v_b; v_a * v_b]$$



Here, the operations $-$, $*$ are the element-wise subtraction and element-wise product, and $[;]$ is the concatenation. This aggregation methold could be regraded as deep sematic feature interaction. 



Finally, with the aggregated feature $v$, we employ a two-layer feed-forward neural network for classification task and $gelu$ activation function is abdoted after first layer. We use multi-class cross-entropy loss function to train our model.

$$\hat{y} = softmax(gelu(vW_{o1})W_{o2})$$
$$ Loss = - \sum_{j=1}^{C} y_jlog(\hat{y_j}) $$

where $v \in R^{1*d}$ , $W_{o1} \in R^{d*d'}$ , $W_{o2} \in R^{d' * C}$ and C is the number of label classes, $y$ is the ground truth.





# 4 Experiments
In this section, we evaluate our model on three tasks: natural language inference, paraphrase identification and answer sentence selection. 



## 4.1 Datasets





## 4.2 Implementation Details





## 4.3 Ablation Study





## 4.4 Analysis











# 5 Result





# 6 Conclusion





# 7. Reference



