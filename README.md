# Reinforced Tree Search

RTS is a Deep Reinforcement Learning agent capable of increase performance of algorithms applied on high resolution images. Instead of using a conventional sliding windows, RTS perform a tree search like by successive zoom in subpart of the image. RTS learn to find the most effective path with the higher probabilities of finding objects.

<img src="images/rts_visualisation.png">
On the image above RTS we can see that RTS has learn to find object on coastal area. We can also we the successive zooms realized by the algorithm.

RTS can improve time of search in high resolution images.

It has been tested on two problematics:
- finding a object has quickly has possible.
- finding all object in the images has quickly has possible.

RTS perform extremely well on the first and good in the second case.

<img src="images/best_so_far.gif" width="200" height="200">

here is a visualization of the agent trying to find an object, we can see that he has learn to search on coastal area.

<img src="images/16.jpg.gif">

In this second example RTS try to find all the sea lion in the image. This image come from the dataset NOAA: https://www.kaggle.com/competitions/noaa-fisheries-steller-sea-lion-population-count/overview

# Demo

in the noaa folder you can find fully implemented demo on the NOAA dataset.

**Caution:** RTS is, in is current state, not ready to be used. It is a proof of concept.
If you want to use it anyway, you can follow the installation and usage instructions.

# Installation instructions

clone this repository:

```
git clone https://github.com/UncleBen420/RTS-Reinforced-Tree-Search-.git
```

install dependencies using pip install:

```
pip install -r requirements.txt
```

Your are now able to use RTS.

# Usage instructions

## Training

Before using RTS, you need to train it on your data. To do this you need a high resolution images dataset.

```
python3 train.py -tr path/to/training/data -e 300 -plt -ts path/to/testing/data

```

here are the argument you can use to custom the training:

```
-tr, --train_path: The path to the training data.
-ts, --eval_path: The path to the evaluation data.
-o, --results_path: The path where the results and weights will be saved.
-e, --episodes: duration of the training in episode.
-eg, --epsilon: epsilon factor of the e-greedy function.
-a, --learning_rate: learning of the Q-Net.
-g, --gamma: discount factor of the discounted rewards.
-lrg, --lr_gamma: every 100 episodes the learning rate drop by this factor.
-res, --min_resolution: resolution at which, the agent stop zooming.
-rt, --real_time_monitor: enable a server that allow you to see the training in real time.
-plt, --plot_metric: return different metrics about evaluation and training.
-tl, --transfer_learning: path to existing weights

```

The data used for training and testing must be in this format:
```
Folder
  |-img
  |-bboxes
```

the image contained in the img folder must have a file matching their name but in .txt format in the folder bboxes.
Those file contains all objects in the corresponding image given:
the label, the x of the left-upmost point, the y of the left-upmost point, the width and the height:

here is an exemple of a file containing multiple objects:

```
0 0 0 17 32
0 25 0 18 28
0 49 0 13 43
0 76 0 21 44
0 107 0 22 34
0 65 14 10 53
3 0 33 18 11
3 33 38 15 29
3 79 56 49 17
0 0 68 43 44
0 52 86 50 18
3 16 116 46 24
```

 RTS does not care about the class. It search every objects so if you want to only search one class of object, you should delete unnecessary lines.

 ## Inference

 to use RTS in your code, simple copy paste the needed weights in the rts/weights folder.
 Then copy the rts folder to your project folder.
 Finally you can import rts.

 If you want to use rts with YOLO v5n, you can simply paste YOLO weights in the rts/weights folder.

 then your can use directly the rts.py file for inference:

 ```
 python3 rts/rts.py -img path/to/the/image -max 100

 ```

 this command will perform a search of rts with YOLO bounded to it.
 You can use the following arguments to custom the search.

 ```
-img, --image_path: the path to the image that will be analyzed
-max, --max_actions_allowed: the maximal number of actions allowed to be perform by RTS
-yolo, --yolo_weights: the name of the yolo weights file (default: yolo_weights.pt)
-rts, --rts_weights: the name of the rts weights file (default: weights_rts.pt)
 ```

 For inference and bounding with a custom model directly in your code. Please refer to the NOAA demo iPython file.
