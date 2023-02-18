# SSL-SSTD
This is code for paper: [Self-spectral learning with GAN based spectral--spatial target detection for hyperspectral image](https://www.sciencedirect.com/science/article/pii/S0893608021002252)

## Citation

```
@article{xie2021self,
  title={Self-spectral learning with GAN based spectral--spatial target detection for hyperspectral image},
  author={Xie, Weiying and Zhang, Jiaqing and Lei, Jie and Li, Yunsong and Jia, Xiuping},
  journal={Neural Networks},
  volume={142},
  pages={375--387},
  year={2021},
  publisher={Elsevier}
}
```

## Environment
```
matlab 2017a
tensorflow
```

## How to use
1.run the SSL.py to obtain the encoder features for detection
### Train
change the path of the dataset and set the train_modal=True
```
python SSL.py
```
### Test
set the train_modal=False
```
python SSL.py
```
2. open the test folder in the matlab software
change some paths in the test.m

- change the path of the decoder features called encoderfile_our
- change the path of save results called savefile_our
- change the path of dataset called file

run the test.m

## Contact
If you have any question, please contact me with email (jq.zhangcn@foxmail.com).
