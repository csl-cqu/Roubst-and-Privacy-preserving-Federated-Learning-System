# Backdoor Validation & Defense Module

The validation module and the defense module do not rely on each other, so they can be turned on independently. Both modules require `--use_gradient` argument **NOT** being enabled.

## Validation

Use argument `--backdoor-test --backdoor-test-frequency 15` to turn on validation for backdoor defense. If do so, the accuracy of the backdoor task (both training dataset and testing dataset) will be logged by wandb.

## Defense

Use argument `--backdoor-defense --backdoor-defense-shrink 0.5 --backdoor-defense-noise 0.04` to turn on backdoor defense.

## Reference

[1] Bagdasaryan E, Veit A, Hua Y, et al. How to backdoor federated learning. International Conference on Artificial Intelligence and Statistics. 2020: 2938-2948.

[2] Sun Z, Kairouz P, Suresh A T, et al. Can you really backdoor federated learning? arXiv preprint arXiv:1911.07963, 2019.
