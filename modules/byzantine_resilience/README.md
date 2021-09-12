## Introduction

byzantine_resilience module inclues two files: defense.py and attack.py

**defense.py** includes some aggregate algorithms:

* Avg
* Trimmed-mean
* Median
* Krum
* MKrum
* Bulyan
* Dnc

**attack.py** includes Gaussian and ARG-TAL algorithms to get malicious gradients:

* AGR-TAL
* Gaussian

## Usage

In `Byzantine_defense.sh`, add follow parameters:
  * --use_gradient\
      client send gradients to server
  * --byzantine_aggregate\
      use byzantine defense module
  * --by_attack Gaussian\
      use Gaussian attack method, or ARG-TAL 
  * --by_defense Krum\
      use Krum defense method, or Avg, Trimmed-mean, Median, MKrum, Bulyan, Dnc  
  * --num_worker 10\
      total clients number
  * --by_workers 2\
      byzantine clients number
  * --all_grads\
      all_grads means byzantine clients know all gradients of clients, default False

## References

> [1] 
> Xie C, Koyejo O, Gupta I. Phocas: dimensional byzantine-resilient stochastic gradient descent[J]. arXiv preprint arXiv:1805.09682, 2018.
>
> [2]
> Blanchard P, El Mhamdi E.M., et al. Machine learning with adversaries: Byzantine tolerant gradient descent[C]. International Conference on Neural Information Processing Systems,2017:118-128.
>
> [3]
> Mhamdi E M E, Guerraoui R, Rouault S. The hidden vulnerability of
> distributed learning in Byzantium[C]. International Conference on Machine
> Learning. 2018 : 3521–3530.
>
> [4]
> Shejwalkar, V. and Houmansadr, A. Manipulating the Byzantine:
> Optimizing Model Poisoning Attacks and Defenses for Federated Learning[C]. The Network and Distributed System Security Symposium. Internet Society,2021.
