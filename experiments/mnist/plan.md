# MNIST DP-SGD disparity study

We study whether differential privacy hurts a rare MNIST digit more than a common one. Digit **8** is the rare class; digit **2** is the control. The only intended difference between training methods is how privacy is applied.

## Experiment settings

- **Data:** Keep each training example labeled 8 with probability **9%**. Keep all other training examples and the full test set. Use the same selected training examples within each comparison.
- **Seeds:** Run seeds **0–4**. For each seed, use the same data selection and initial model weights in Phases 1, 2, and 4. Pairing runs by seed controls one source of randomness; repeating across seeds shows how much results vary.
- **CNN:** `Conv2d(1, 32, 3, stride=1, padding=0) → Tanh → Conv2d(32, 16, 3, stride=1, padding=0) → Tanh → Flatten → Linear(16 × 24 × 24, 10)`. No pooling. Scale pixels to `[0, 1]`.
- **Training:** Cross-entropy loss; SGD with learning rate `0.01`, momentum `0`, and no scheduler; **60 epochs**; training and test batch sizes **256**.
- **Software:** Use current libraries and record their exact versions.

## Four phases

1. **Without privacy:** Train the CNN with ordinary SGD. No gradient clipping or DP noise.
2. **Vanilla DP-SGD:** Train the same CNN with per-example clipping and Gaussian noise. Use clipping norm `C=1`, fixed noise multiplier `σ=0.8`, RDP accounting, and `δ=10⁻⁶`. Record the achieved `ε` (about `5.9` in the paper). Do not tune `σ` to a target `ε` for this run.
3. **Compare:** For each seed and digit, calculate **privacy cost = Phase 1 accuracy − Phase 2 accuracy**. Compare digits 8 and 2, show all-digit and overall test accuracy, and report the mean and standard deviation across seeds.
4. **Test DPSGD-Global-Adapt:** Apply the paper's mitigation using the same data, model, and seeds. Account for its extra private count when calculating privacy loss. Compare it with Phase 2 using the authors' published benchmark configuration to see whether digit 8 improves and the disparity shrinks.
   - *Note on Global-Adapt settings:* In the paper and the authors' script (`mnist_script.sh`), Phase 4 uses learning rate `0.1` (increased from `0.01` because Global-Adapt scales gradients downwards), strict clipping bound `Z=50`, clipping norm `C=1`, noise multiplier `σ=0.8`, threshold `τ=0.7`, and private count noise `σ₂=10`. Because count noise `σ₂=10` is large, its additional privacy loss is minimal, yielding a closely matched achieved privacy level (`ε ≈ 5.90` at `δ=10⁻⁶`).

## References

- [Paper: *Disparate Impact in Differential Privacy from Gradient Misalignment*](https://arxiv.org/abs/2206.07737)
- [Authors' MNIST experiment script](https://github.com/layer6ai-labs/fair-dp/blob/c61a163e766fde2e634b40c8afbd82e42644f5b7/experiment_scripts/mnist_script.sh)
- [Authors' model and configuration](https://github.com/layer6ai-labs/fair-dp/tree/c61a163e766fde2e634b40c8afbd82e42644f5b7)
