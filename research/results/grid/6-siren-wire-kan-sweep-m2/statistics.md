# Grid Sweep Statistics
# Grid Sweep m2 (SIREN + WIRE + KAN, with epoch sweep)
# Generated: 2026-09-29 17:05

============================================================
  SIREN  (216 configurations)
============================================================

## Parameters Swept
  hidden_sizes: ['[128, 128, 128, 128]', '[128, 128, 128]', '[256, 256]', '[64, 64, 64]']
  omega_0: ['15.0', '30.0', '45.0']
  lr: ['0.0002', '0.0005', '0.001']
  n_per_param: ['512']
  n_epochs: ['10000', '1500', '2000', '3000', '4500', '6500']

## Numeric Ranges
  n_params_total:
    min:  8897
    max:  68097
    mean: 40465
  mean_rel_err:
    min:  0.000213283
    max:  0.560279
    mean: 0.00659243
  mean_correct_digits:
    min:  1.25
    max:  4.875
    mean: 3.87442
  elapsed_s:
    min:  11.0686
    max:  122.113
    mean: 44.7911

## Best Configuration (by mean_correct_digits)
  hidden_sizes: [128, 128, 128, 128]
  omega_0: 30.0
  lr: 0.0002
  n_per_param: 512
  n_epochs: 2000
  n_params_total: 50689
  final_loss: 3.821484642685391e-05
  min_loss: 3.3618031011428684e-05
  mean_rel_err: 0.00021328299840332128
  max_rel_err: 0.000715623543318532
  min_correct_digits: 4
  mean_correct_digits: 4.875
  elapsed_s: 24.239464282989502

## Worst Configuration (by mean_correct_digits)
  hidden_sizes: [128, 128, 128, 128]
  omega_0: 45.0
  lr: 0.001
  n_per_param: 512
  n_epochs: 1500
  n_params_total: 50689
  final_loss: 0.29557836055755615
  min_loss: 0.253009170293808
  mean_rel_err: 0.560278860772729
  max_rel_err: 0.7941273211764627
  min_correct_digits: 1
  mean_correct_digits: 1.25
  elapsed_s: 17.88455581665039

============================================================
  WIRE  (432 configurations)
============================================================

## Parameters Swept
  omega_0: ['10.0', '15.0', '5.0']
  sigma_0: ['10.0', '15.0', '5.0']
  entry_width: ['32', '64']
  n_blocks: ['2', '3', '4']
  lr: ['0.0005']
  epochs: ['10000', '1500', '2000', '3000', '4000', '5500', '7000', '8500']

## Numeric Ranges
  n_params_total:
    min:  16833
    max:  132225
    mean: 62209
  mean_rel_err:
    min:  7.6525e-05
    max:  1
    mean: 0.00532554
  mean_correct_digits:
    min:  1
    max:  5.375
    mean: 4.37905
  elapsed_s:
    min:  66.1239
    max:  852.478
    mean: 334.696

## Best Configuration (by mean_correct_digits)
  omega_0: 15.0
  sigma_0: 5.0
  entry_width: 64
  n_blocks: 4
  lr: 0.0005
  epochs: 8500
  n_params_total: 132225
  final_loss: 4.061452636960894e-05
  min_loss: 2.426469472993631e-05
  mean_rel_err: 0.0001374371336294902
  max_rel_err: 0.0004334527518653261
  min_correct_digits: 4
  mean_correct_digits: 5.375
  elapsed_s: 706.3027038574219

## Worst Configuration (by mean_correct_digits)
  omega_0: 15.0
  sigma_0: 15.0
  entry_width: 64
  n_blocks: 4
  lr: 0.0005
  epochs: 5500
  n_params_total: 132225
  final_loss: 0.3692616820335388
  min_loss: 5.861130921402946e-05
  mean_rel_err: 0.9996829612816295
  max_rel_err: 1.0016756086436043
  min_correct_digits: 0
  mean_correct_digits: 1.0
  elapsed_s: 456.3774609565735

============================================================
  KAN  (648 configurations)
============================================================

## Parameters Swept
  width: ['16', '32', '64']
  depth: ['1', '2', '3']
  G: ['12', '5', '8']
  k: ['3']
  lr: ['0.0005', '0.001', '0.002']
  epochs: ['10000', '1500', '2000', '3000', '4000', '5500', '7000', '8500']

## Numeric Ranges
  n_params_total:
    min:  1198
    max:  139662
    mean: 25948.2
  mean_rel_err:
    min:  0.0238565
    max:  0.268213
    mean: 0.0812217
  mean_correct_digits:
    min:  1.375
    max:  3.125
    mean: 2.29398
  elapsed_s:
    min:  65.4731
    max:  1093.04
    mean: 374.234

## Best Configuration (by mean_correct_digits)
  width: 32
  depth: 3
  G: 8
  k: 3
  lr: 0.0005
  epochs: 3000
  n_params_total: 27854
  final_loss: 0.00022276905656326562
  min_loss: 0.00019484231597743928
  mean_rel_err: 0.024124139095202734
  max_rel_err: 0.07128887504736688
  min_correct_digits: 2
  mean_correct_digits: 3.125
  elapsed_s: 290.38147473335266

## Worst Configuration (by mean_correct_digits)
  width: 16
  depth: 1
  G: 12
  k: 3
  lr: 0.0005
  epochs: 7000
  n_params_total: 2094
  final_loss: 0.004999496042728424
  min_loss: 0.0043983981013298035
  mean_rel_err: 0.2682125531268861
  max_rel_err: 0.4106827839956768
  min_correct_digits: 1
  mean_correct_digits: 1.375
  elapsed_s: 317.4272882938385

============================================================
  TOTAL: 1296 configurations across 3 architectures
============================================================