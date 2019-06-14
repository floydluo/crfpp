# New Python Wrapper Toolkit for CRF++

## 1. Set up and Run the Demo

#### Install CRF++

If Mac, install crf++ at first.

By:

```Shell
$ brew install crf++ 
```

If Window, the crf++ package is included in this project. Don't need to install it.

#### Train a Model

```shell
# generate a model 1a-v50
$ python train.py -m 1a -v vect-50

# generate a model 1abdp
$ python train.py -m 1abdp

# generate a model 2abp
$ python train.py -m 2abp
```


## 2. TODO