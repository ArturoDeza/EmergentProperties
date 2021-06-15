# Emergent Properties of Foveated Perceptual Systems

<img src="https://github.com/ArturoDeza/EmergentProperties/blob/main/Summary_Figure.png" width="1024">

Code + Data for paper Emergent Properties of Foveated Perceptual Systems


| Reference-Net | Foveation-Texture-Net | Uniform-Net | Foveation-Blur-Net |  
| :-: | :-: | :-: | :-: |
| <img src="https://github.com/ArturoDeza/EmergentProperties/blob/main/Reference-Net_Evolution.gif" width="128"> | <img src="https://github.com/ArturoDeza/EmergentProperties/blob/main/Foveation-Texture-Net_Evolution.gif" width="128"> | <img src="https://github.com/ArturoDeza/EmergentProperties/blob/main/Uniform-Net_Evolution.gif" width="128"> | <img src="https://github.com/ArturoDeza/EmergentProperties/blob/main/Foveation-Blur-Net_Evolution.gif" width="128"> |


### All Networks
Contains the PyTorch checkpoints to monitor the evolultion of each one of the randomly initialized networks (paired across sytems) for each perceptual system: Reference-Net, Foveation-Texture-Net, Uniform-Net, Foveation-Blur-Net > Subdivided into Network model (AlexNet or ResNet18) > subdivided by 10. Sanity check observation: Epoch 0 across ALL systems per run should be the same (e.g. all weights of epoch 0 networks are the same for Reference-Net, Foveation-Texture-Net, Uniform-Net and Foveation-Blur-Net for run ID 1, but these are different for run id 2; Same within run, different across runs)

### Dataset Files
Contains the Training, Validation and Testing partition of the images.

### Training
Contains the main train_MiniPlaces.py file that is used to train all perceptual systems reported in the paper. Code is flexible to be used in other training schemes as well.

### Filters
Contains code to Visualize the evolution of the learned filters and create animated .gifs as above.

### Generalization
Contains code to reproduce the Generalization Experiments and a pipeline to integrate further experiments as well.

### Robustness to Occlusion
Contains code to reproduce the Robustness to Occlusion Experiments (Glaucoma + Scotoma) and a pipeline to integrate further experiments as well. We also have the Left2Right and Top2Bottom conditions coded here as well. This folder also contains the occluded stimuli used in our experiments.

### Spatial Frequency
Contains code to reproduce the Spatial Frequency Sensitivity Experiments (High Pass and Low Pass). We also have the High Pass and Low Pass (GRAY) controls where the stimuli are transformed to grayscale to test if these differences are due to color processing (they are not). This folder also contains the occluded stimuli used in our experiments.

### Window Cue Conflict
Contains code to reproduce the Window Cue Conflict Experiment. This folder also contains the occluded stimuli used in our experiments.

### Square Uniform Cue Conflict
Contains code to reproduce the Square Uniform Cue Conflict Experiment -- which is an additional control of the Window Cue Conflict experiment where the conflicting classes are uniformly sampled (vs regularly mis-matched), and also have a square window with a finer sampling step (similar to occlusion). This folder also contains the occluded stimuli used in our experiments.

### Render Loss Function
Contains code to render the individual and aggregate losses (Training + Validation) as a function of epoch across all runs and systems.

### Render Learning Dynamics
Contains code to render the individual and aggregate Validation Accuracy as a function of epoch across all runs and systems.
