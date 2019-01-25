
# Intro

The source codes for the paper "[Network of Recurrent Neural Networks: Design for Emergence](https://link.springer.com/chapter/10.1007/978-3-030-04179-3_8)":

    @inproceedings{wang2018network,
      title={Network of Recurrent Neural Networks: Design for Emergence},
      author={Wang, Chaoming and Zeng, Yi},
      booktitle={International Conference on Neural Information Processing},
      pages={89--102},
      year={2018},
      organization={Springer}
    }
    
or
    
    @article{Wang2017Network,
      title={Network of Recurrent Neural Networks},
      author={Wang, Chao Ming},
      journal={arXiv preprint arXiv:1710.03414},
      year={2017},
    }  

The experiment requirements are:

    tensorflow-gpu=1.2
    python>=3.5
    xlwt>=1.0
    tqdm
    numpy


# Usage 

### Step 1: install tensorflow

1, Install [minconda](https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh) ,

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    
2, Type this in ``~/.bash_profile``

    export PATH="$HOME/miniconda3/bin:$PATH"
    
3, Then try launching a new terminal and create an environment named ``py3k``:
    
    conda create -n py3k anaconda python=3.5
    source activate py3k
    
4, Let's install ``tensorflow-gpu=1.2``:

    conda install numpy
    conda install tensorflow-gpu=1.2

### Step 2: build data

1, download data

You can download CoNLL-2003 from this repository [NER/corpus/CoNLL-2003/](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003). 
Or, you can directly use the data we have downloaded in directory ``data/``.

At the same time, you should download the [MNIST](http://yann.lecun.com/exdb/mnist/) AND put it into the directory ``data/``.

2, build data

Run ``ner_build_data.py``.

    python ner_eval_ner.py --glove_filename ${glove_data_path}/glove.6B.300d.txt


### Step 3: run model

1, If you want to suppress ``I`` log messages in tensorflow, please
type following commands in Terminal:
    
    export TF_CPP_MIN_LOG_LEVEL=2

Meanwhile, if you want to specify what GPUs you are going to use, please 
type the following command in terminal:
    
    export CUDA_VISIBLE_DEVICES=1,2

The above command means to use GPUs which ID is ``1`` or ``2``.

2, Run ``Named Entity Recognition`` task for single experiment:

    python ner_eval.py --lr 0.001 \
                --lr_decay 0.95 \
                --nepoch_no_imprv 5 \
                --nepochs 40 \
                --model_name rnn-relu \
                --hidden_size 197
    
Then you will see following similar outputs:
    
    Epoch 2 out of 40
    703/703 [==============================] - 19s - train loss: 2.7989
    - dev acc 96.38 - f1 82.98
    - new best score!
    
    Epoch 3 out of 40
    703/703 [==============================] - 20s - train loss: 2.2628     
    - dev acc 96.83 - f1 84.91
    - new best score!
    
    Epoch 4 out of 40
    703/703 [==============================] - 21s - train loss: 1.9565     
    - dev acc 97.08 - f1 86.21
    - new best score!

3, Run ``MNIST Classification`` task for single experiment:

    python mnist_eval.py --lr 0.001 \
                    --dropout 0.5 \
                    --lr_decay 0.9 \
                    --hidden 198 \
                    --model irnn

And you will see the same output as the above ``NER`` task.

4, Also, you can type following commands:

    cd run/
    bash ner-20w.sh
    bash mnist-test.sh
    
    