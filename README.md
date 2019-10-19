# GBML
A collection of Gradient-Based Meta-Learning Algorithms with pytorch

* [MAML](http://proceedings.mlr.press/v70/finn17a)

```python
python3 main.py --alg=MAML
```

* [Reptile](https://openai.com/blog/reptile/)

```python
python3 main.py --alg=Reptile
```

* [CAVIA](http://proceedings.mlr.press/v97/zintgraf19a)

```python
python3 main.py --alg=CAVIA
```



## Results on miniImagenet

* Without pre-trained encoder (Use 64 channels by default. The exceptions are in parentheses)

|                | 5way 1shot          | 5way 1shot (ours) | 5way 5shot          | 5way 5shot (ours) |
| -------------- | ------------------- | ----------------- | ------------------- | ----------------- |
| MAML           | 48.70 ± 1.84%       | 49.00 %           | 63.11 ± 0.92%       | -                 |
| Reptile        | 47.07 ± 0.26%       | -                 | 62.74 ± 0.37%       | -                 |
| CAVIA          | 49.84 ± 0.68% (128) | 50.07 % (64)      | 64.63 ± 0.53% (128) | 64.64% (64)       |
| iMAML          | 49.30 ± 1.88%       | -                 | -                   | -                 |
| Meta-Curvature | 55.73 ± 0.94% (128) | -                 | 70.30 ± 0.72% (128) | -                 |

* With pre-trained encoder (To be implemented.)

|                | 5way 1shot    | 5way 1shot (ours) | 5way 5shot    | 5way 5shot (ours) |
| -------------- | ------------- | ----------------- | ------------- | ----------------- |
| Meta-SGD       | 56.58 ± 0.21% | -                 | 68.84 ± 0.19% | -                 |
| LEO            | 61.76 ± 0.08% | -                 | 77.59 ± 0.12% | -                 |
| Meta-Curvature | 61.85 ± 0.10% | -                 | 77.02 ± 0.11% | -                 |

## Dependencies

* Python >= 3.6
* Pytorch >= 1.2
* [Higher](https://github.com/facebookresearch/higher) 
* [Torchmeta](https://github.com/tristandeleu/pytorch-meta) 



## To do

* Add ~~ResNet~~ and Pre-trained encoder
* Add iMAML, Meta-Curvature

