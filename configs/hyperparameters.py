## Architechture
lpips_type = 'alex'
first_inv_type = 'psp'
optim_type = 'adam'
use_last_w_pivots = False
train_batch_size=1
## Locality regularization
locality_regularization_interval = 1
use_locality_regularization = False
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
regulizer_alpha = 30

## Optimization
pti_learning_rate = 3e-4

# Loss
pt_l2_lambda = 1
pt_lpips_lambda = 1
pt_att_lambda=1

# ALO
alo_steps=100

# fine-tune
ft_steps = 100