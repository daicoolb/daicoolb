---
layout: post
title: Keras 实现自编码器
description: ""
category: 研究
tags: [Linux]
---

## 一、自编码器简介
无监督特征学习 (Unsupervised Feature Learning）是一种仿人脑的对特征逐层抽象提取的过程，学习过程中有两点：一是无监督学习，即对训练数据不需要进行标签化标注，这种学习是对数据内容的组织形式的学习，提取的是频繁出现的特征；二是逐层抽象，特征是需要不断抽象的。

自编码器（AutoEncoder），即可以使用自身的高阶特征自我编码，自编码器其实也是一种神经网络，其输入和输出是一致的，借助了稀疏编码的思想，目标是使用稀疏的高阶特征重新组合来重构自己。

![]({{ site.url }}/assets/images/autoencoder_schema.jpg)

## 二、完整代码

    import numpy as np  
    np.random.seed(1337)  # for reproducibility  
      
    from keras.datasets import mnist  
    from keras.models import Model #泛型模型  
    from keras.layers import Dense, Input  
    import matplotlib.pyplot as plt  
      
    # X shape (60,000 28x28), y shape (10,000, )  
    (x_train, _), (x_test, y_test) = mnist.load_data()  
      
    # 数据预处理  
    x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized  
    x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized  
    x_train = x_train.reshape((x_train.shape[0], -1))  
    x_test = x_test.reshape((x_test.shape[0], -1))  
    print(x_train.shape)  
    print(x_test.shape)  
      
    # 压缩特征维度至2维  
    encoding_dim = 2  
      
    # this is our input placeholder  
    input_img = Input(shape=(784,))  
      
    # 编码层  
    encoded = Dense(128, activation='relu')(input_img)  
    encoded = Dense(64, activation='relu')(encoded)  
    encoded = Dense(10, activation='relu')(encoded)  
    encoder_output = Dense(encoding_dim)(encoded)  
      
    # 解码层  
    decoded = Dense(10, activation='relu')(encoder_output)  
    decoded = Dense(64, activation='relu')(decoded)  
    decoded = Dense(128, activation='relu')(decoded)  
    decoded = Dense(784, activation='tanh')(decoded)  
      
    # 构建自编码模型  
    autoencoder = Model(inputs=input_img, outputs=decoded)  
      
    # 构建编码模型  
    encoder = Model(inputs=input_img, outputs=encoder_output)  
      
    # compile autoencoder  
    autoencoder.compile(optimizer='adam', loss='mse')  
      
    # training  
    autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True)  
      
    # plotting  
    encoded_imgs = encoder.predict(x_test)  
    plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, s=3)  
    plt.colorbar()  
    plt.show() 

## 三、程序解读

自编码，简单来说就是把输入数据进行一个压缩和解压缩的过程。原来有很多特征，压缩成几个来代表原来的数据，解压之后恢复成原来的维度，再和原数据进行比较。它是一种非监督算法，只需要输入数据，解压缩之后的结果与原数据本身进行比较。程序的主要功能是把 datasets.mnist 数据的 28/28=784 维的数据，压缩成 2 维的数据，然后在一个二维空间中可视化出分类的效果。

首先，导入数据并进行数据预处理，本例使用Model模块的Keras的泛化模型来进行模型搭建，便于我们从模型中间导出数据并进行可视化。进行模型搭建的时候，注意要进行逐层特征提取，最终压缩至2维，解码的过程要跟编码过程一致相反。随后对Autoencoder和encoder分别建模，编译、训练。将编码模型的预测结果通过Matplotlib可视化出来，就可以看到原数据的二维编码结果在二维平面上的聚类效果，还是很明显的。

![]({{ site.url }}/assets/images/autoencoder_figure.jpg)
    
最后看到可视化的结果，自编码模型可以把这几个数字给区分开来，我们可以用自编码这个过程来作为一个特征压缩的方法，和PCA的功能一样，效果要比它好一些，因为它是非线性的结构。 
