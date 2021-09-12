## Introduction

differential_privacy module includes three files: defense.py, dp_engine.py, eps_log.py

**defense.py** includes some model and manage functions:

* check if model is compatible
* modify model for opacus
* create engine and attach
* calculate privacy spent

**dp_engine.py** includes main algorithms for differential privacy:

* hook
* add noise
* calculate spent

**eps_log.py** includes recording functions:

* privacy parameters
* epsilon in training process

## Usage

In `differential_privacy.sh`, add follow parameters:
  * --differential_privacy\
      use differential privacy module
  * --dp_sigma\
      noise multiplier parameter
  * --dp_delta\
      fault tolerance in epsilon - delta differential privacy 
  * --grad_norm\
      gradient clipping parameter, norm number

## References

> [1] 
> Mironov, Ilya. "Rényi differential privacy." 2017 IEEE 30th Computer Security Foundations Symposium (CSF). IEEE, 2017.
> 
> [2] 
> Abadi, Martin, et al. "Deep learning with differential privacy." Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security. ACM, 2016. 
>
> [3] 
> Mironov, Ilya, Kunal Talwar, and Li Zhang. "R'enyi Differential Privacy of the Sampled Gaussian Mechanism." arXiv preprint arXiv:1908.10530 (2019).  
>
> [4] 
>Goodfellow, Ian. "Efficient per-example gradient computations." arXiv preprint arXiv:1510.01799 (2015).  
>
> [5] 
>McMahan, H. Brendan, and Galen Andrew. "A general approach to adding differential privacy to iterative training procedures." arXiv preprint arXiv:1812.06210 (2018).



