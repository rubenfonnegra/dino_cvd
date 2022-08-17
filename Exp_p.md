## Train DINO


```
python main_dino.py --arch vit_small \
                    --momentum_teacher 0.995 \
                    --batch_size_per_gpu 32 \
                    --data_path /home/ruben-kubuntu/Devs/dino_cvd/cvd_data/train/dino/ \
                    --output_dir /home/ruben-kubuntu/Devs/dino_cvd/Results/small_cvd_s \
                    --epochs 501 --num-channels 1
```

```
python main_dino.py --arch vit_small \
                    --momentum_teacher 0.995 \
                    --batch_size_per_gpu 32 \
                    --data_path /home/ruben-kubuntu/Devs/dino_cvd/cvd_data/train/dino/ \
                    --output_dir /home/ruben-kubuntu/Devs/dino_cvd/Results/small_cvd_g \
                    --epochs 101 --num-channels 1

```

### Ablations 


```
python main_dino.py --arch vit_small \
                    --momentum_teacher 0.995 \
                    --batch_size_per_gpu 32 \
                    --data_path /home/ruben-kubuntu/Devs/dino_cvd/cvd_data/train/dino/ \
                    --output_dir /home/ruben-kubuntu/Devs/dino_cvd/Results/sm_cvd_1 \
                    --epochs 101 --num-channels 1
```

```

python main_dino.py --arch vit_small \
                    --momentum_teacher 0.99 \
                    --batch_size_per_gpu 32 \
                    --data_path /home/ruben-kubuntu/Devs/dino_cvd/cvd_data/train/dino/ \
                    --output_dir /home/ruben-kubuntu/Devs/dino_cvd/Results/sm_cvd_2 \
                    --epochs 101 --num-channels 1
```

```
python main_dino.py --arch vit_small \
                    --momentum_teacher 0.98 \
                    --batch_size_per_gpu 32 \
                    --data_path /home/ruben-kubuntu/Devs/dino_cvd/cvd_data/train/dino/ \
                    --output_dir /home/ruben-kubuntu/Devs/dino_cvd/Results/sm_cvd_3 \
                    --epochs 101 --num-channels 1
```


## Extract features

```
python extract_features.py  --arch vit_small --imsize 480 --multiscale 0 \
                            --train_data_path cvd_data/train/dino/ \
                            --test_data_path cvd_data/test/dino/ \
                            --pretrained_weights Results/small_cvd_s/checkpoint0300.pth \
                            --output_dir Features/small_cvd_s/ --num-channels 1

```
