### This repo is based on [CorrNet (CVPR 2023)](https://github.com/hulianyuyy/CorrNet).

## Implementation
The implementation for the CorrNet (line 18) is given in [./modules/resnet.py](https://github.com/hulianyuyy/CorrNet_CSLR/blob/main/modules/resnet.py).  

It's then equipped with the BasicBlock in ResNet in line 58 [./modules/resnet.py](https://github.com/hulianyuyy/CorrNet_CSLR/blob/main/modules/resnet.py).

We later found that the Identification Module with only spatial decomposition could perform on par with what we report in the paper (spatial-temporal decomposition) and is slighter faster, and thus implement it as such.

## Data Preparation
You can choose any one of following datasets to verify the effectiveness of CorrNet.

### How2Sign dataset

1. Download and extract the [How2Sign dataset](https://how2sign.github.io/) including the RGB front video clips and aligned annotations.

2. Place the video clips under:
```

dataset/how2sign/train\_rgb\_front\_clips/raw\_videos
dataset/how2sign/val\_rgb\_front\_clips/raw\_videos
dataset/how2sign/test\_rgb\_front\_clips/raw\_videos

```
and the CSV annotations under:
```

dataset/how2sign/how2sign\_realigned\_train.csv
dataset/how2sign/how2sign\_realigned\_val.csv
dataset/how2sign/how2sign\_realigned\_test.csv

````

3. Run the following command to preprocess How2Sign and generate the gloss dictionary:
```bash
cd ./preprocess
python dataset_preprocess-how2sign.py --dataset how2sign --dataset-root ../dataset/how2sign
````

Add `--process-image --multiprocessing` if you want to extract and resize frames from video clips.

The `dataset_preprocess-how2sign.py` script handles the How2Sign-specific structure and formats.

### PHOENIX2014 dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments based on phoenix-2014.v3.tar.gz.

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phoenix2014`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python dataset_preprocess.py --process-image --multiprocessing
   ```

### PHOENIX2014-T dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/PHOENIX-2014-T-release-v3/PHOENIX-2014-T ./dataset/phoenix2014-T`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python dataset_preprocess-T.py --process-image --multiprocessing
   ```

If you get an error like ```IndexError: list index out of range``` on the PHOENIX2014-T dataset, you may refer to [this issue](https://github.com/hulianyuyy/CorrNet/issues/10#issuecomment-1660363025) to tackle the problem.


### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model, run the command below:

`python main.py --config ./config/baseline.yaml --device your_device`

Note that you can choose the target dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml.
 
For CSL-Daily dataset, You may choose to reduce the lr by half from 0.0001 to 0.00005, change the lr deacying rate (gamma in the 'optimizer.py') from 0.2 to 0.5, and disable the temporal resampling strategy (comment line 121 in dataloader_video.py).

<!-- ### Visualizations
For Grad-CAM visualization, you can replace the resnet.py under "./modules" with the resnet.py under "./weight_map_generation", and then run ```python generate_cam.py``` with your own hyperparameters.

### Test with one video input
Except performing inference on datasets, we provide a `test_one_video.py` to perform inference with only one video input. An example command is 

`python test_one_video.py --model_path /path_to_pretrained_weights --video_path /path_to_your_video --device your_device`

The `video_path` can be the path to a video file or a dir contains extracted images from a video.

Acceptable paramters:
- `model_path`, the path to pretrained weights.
- `video_path`, the path to a video file or a dir contains extracted images from a video.
- `device`, which device to run inference, default=0.
- `language`, the target sign language, default='phoenix', choices=['phoenix', 'csl'].
- `max_frames_num`, the max input frames sampled from an input video, default=360.

-->