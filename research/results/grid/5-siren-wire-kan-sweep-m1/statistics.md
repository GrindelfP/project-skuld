# Grid Sweep Statistics
# Grid Sweep m1 (SIREN + WIRE + KAN, no epoch sweep)
# Generated: 2026-09-29 17:05

============================================================
  SIREN  (36 configurations)
============================================================

## Parameters Swept
  hidden_sizes: ['[128, 128, 128, 128]', '[128, 128, 128]', '[256, 256]', '[64, 64, 64]']
  omega_0: ['15.0', '30.0', '45.0']
  lr: ['0.0002', '0.0005', '0.001']
  n_per_param: ['512']

## Numeric Ranges
  n_params_total:
    min:  8897
    max:  68097
    mean: 40465
  mean_rel_err:
    min:  0.000439266
    max:  0.560279
    mean: 0.0174403
  mean_correct_digits:
    min:  1.25
    max:  4.875
    mean: 3.92014
  elapsed_s:
    min:  14.6312
    max:  25.6957
    mean: 20.0774

## Best Configuration (by mean_correct_digits)
  hidden_sizes: [64, 64, 64]
  omega_0: 45.0
  lr: 0.0005
  n_per_param: 512
  n_params_total: 8897
  final_loss: 0.00018263430683873594
  min_loss: 0.00011619592987699434
  mean_rel_err: 0.0004471767150228656
  max_rel_err: 0.001889531218290586
  min_correct_digits: 4
  mean_correct_digits: 4.875
  elapsed_s: 18.69748878479004

## Worst Configuration (by mean_correct_digits)
  hidden_sizes: [128, 128, 128, 128]
  omega_0: 45.0
  lr: 0.001
  n_per_param: 512
  n_params_total: 50689
  final_loss: 0.29557836055755615
  min_loss: 0.253009170293808
  mean_rel_err: 0.560278860772729
  max_rel_err: 0.7941273211764627
  min_correct_digits: 1
  mean_correct_digits: 1.25
  elapsed_s: 24.35249972343445

============================================================
  WIRE  (54 configurations)
============================================================

## Parameters Swept
  omega_0: ['10.0', '15.0', '5.0']
  sigma_0: ['10.0', '15.0', '5.0']
  entry_width: ['32', '64']
  n_blocks: ['2', '3', '4']
  lr: ['0.0005']

## Numeric Ranges
  n_params_total:
    min:  16833
    max:  132225
    mean: 62209
  mean_rel_err:
    min:  0.000324192
    max:  0.0030766
    mean: 0.0012814
  mean_correct_digits:
    min:  3.625
    max:  4.625
    mean: 4.0625
  elapsed_s:
    min:  69.617
    max:  141.644
    mean: 102.627

## Best Configuration (by mean_correct_digits)
  omega_0: 5.0
  sigma_0: 5.0
  entry_width: 64
  n_blocks: 3
  lr: 0.0005
  n_params_total: 99329
  final_loss: 0.0005026179715059698
  min_loss: 0.0004308212664909661
  mean_rel_err: 0.0003241920130873626
  max_rel_err: 0.0007388111235361665
  min_correct_digits: 4
  mean_correct_digits: 4.625
  elapsed_s: 112.09407019615173

## Worst Configuration (by mean_correct_digits)
  omega_0: 15.0
  sigma_0: 10.0
  entry_width: 32
  n_blocks: 3
  lr: 0.0005
  n_params_total: 25089
  final_loss: 0.0006564916693605483
  min_loss: 0.0003288424923084676
  mean_rel_err: 0.0020206525309736535
  max_rel_err: 0.0028207750872128057
  min_correct_digits: 3
  mean_correct_digits: 3.625
  elapsed_s: 102.54720783233643

============================================================
  KAN  (81 configurations)
============================================================

## Parameters Swept
  width: ['16', '32', '64']
  depth: ['1', '2', '3']
  G: ['12', '5', '8']
  k: ['3']
  lr: ['0.0005', '0.001', '0.002']

## Numeric Ranges
  n_params_total:
    min:  1198
    max:  139662
    mean: 25948.2
  mean_rel_err:
    min:  0.0318833
    max:  0.159473
    mean: 0.0749502
  mean_correct_digits:
    min:  1.875
    max:  3
    mean: 2.27469
  elapsed_s:
    min:  70.6496
    max:  170.132
    mean: 115.406

## Best Configuration (by mean_correct_digits)
  width: 64
  depth: 2
  G: 8
  k: 3
  lr: 0.001
  n_params_total: 55566
  final_loss: 0.0002553650992922485
  min_loss: 0.00022186970454640687
  mean_rel_err: 0.032847008103956056
  max_rel_err: 0.09968559415848026
  min_correct_digits: 3
  mean_correct_digits: 3.0
  elapsed_s: 109.49482274055481

## Worst Configuration (by mean_correct_digits)
  width: 32
  depth: 1
  G: 8
  k: 3
  lr: 0.0005
  n_params_total: 3150
  final_loss: 0.006129727698862553
  min_loss: 0.005236348137259483
  mean_rel_err: 0.1101971315004234
  max_rel_err: 0.17002063495938272
  min_correct_digits: 1
  mean_correct_digits: 1.875
  elapsed_s: 74.30202531814575

============================================================
  TOTAL: 171 configurations across 3 architectures
============================================================