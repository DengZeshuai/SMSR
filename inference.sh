# python main.py --dir_data /home/dengzeshuai/datasets --data_test Set5 --scale 2 --model SMSR --save SMSR_X2 --pre_train experiment/SMSR_X2/model/model_1000.pt --test_only --cpu True

python main.py --dir_data /home/dengzeshuai/datasets --data_test DemoCenter --scale 2 --model SMSR --save SMSR_X2 --pre_train experiment/SMSR_X2/model/model_1000.pt --test_only --cpu True

# compute the inference time for SMSR X2
python main.py --dir_data /home/dengzeshuai/datasets --data_test DemoCenter --demo_name Urban100 --scale 2 --model SMSR --save SMSR_X2 --pre_train experiment/SMSR_X2/model/model_1000.pt --test_only --cpu True --patch_size 192

# compute the inference time for SMSR X4
python main.py --dir_data /home/dengzeshuai/datasets --data_test DemoCenter --demo_name Urban100 --scale 4 --model SMSR --save SMSR_X4 --pre_train experiment/SMSR_X4/model/model_1000.pt --test_only --cpu True --patch_size 384