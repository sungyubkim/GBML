# GBML
A collection of Gradient-Based Meta-Learning Algorithms with pytorch

* [MAML](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)
* [Reptile](https://openai.com/blog/reptile/)
* [CAVIA]()



## Results

|                | 5way 1shot                   | 5way 1shot (ours)     | 5way 5shot                   | 5way 5shot (ours) |
| -------------- | ---------------------------- | --------------------- | ---------------------------- | ----------------- |
| MAML           | 48.70 ± 1.84%                | 49.00 %               | 63.11 ± 0.92%                | -                 |
| Reptile        | 47.07 ± 0.26%                | -                     | 62.74 ± 0.37%                | -                 |
| CAVIA          | 49.84 ± 0.68% (128 channels) | 50.07 % (64 channels) | 64.63 ± 0.53% (128 channels) | -                 |
| iMAML          | 49.30 ± 1.88%                | -                     | -                            | -                 |
| Meta-Curvature | 55.73 ± 0.94% (128 channels) | -                     | 70.30 ± 0.72% (128 channels) | -                 |

## Dependencies

* Python >= 3.6
* Pytorch >= 1.2
* [Higher](https://github.com/facebookresearch/higher) 
* [Torchmeta](https://github.com/tristandeleu/pytorch-meta) 



## To do

* Add ~~ResNet~~, and Pre-trained feature extractor
* Add iMAML, Meta-Curvature

