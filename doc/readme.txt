Change Log:
1. libfft修改為只計算到語音訊號的power。
2. 修改 Libs/feature_utils.py:
   2.1. 新增compute_power_lfbe(line:156), 
           gen_train_lfbe_1600x1_with_cfft_power(line:163)
