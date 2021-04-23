# Emergent Properties of Foveated Perceptual Systems
Code + Data for paper Emergent Properties of Foveated Perceptual Systems



| Reference-Net | Foveation-Texture-Net | Uniform-Net | Foveation-Blur-Net |  
| :-: | :-: | :-: | :-: |
| <img src="https://github.com/ArturoDeza/EmergentProperties/blob/main/Reference-Net_Evolution.gif" width="128"> | <img src="https://github.com/ArturoDeza/EmergentProperties/blob/main/Foveation-Texture-Net_Evolution.gif" width="128"> | <img src="https://github.com/ArturoDeza/EmergentProperties/blob/main/Uniform-Net_Evolution.gif" width="128"> | <img src="https://github.com/ArturoDeza/EmergentProperties/blob/main/Foveation-Blur-Net_Evolution.gif" width="128"> |


### All Networks
Contains the PyTorch checkpoints to monitor the evolultion of each one of the randomly initialized networks (paired across sytems) for each perceptual system: Reference-Net, Foveation-Texture-Net, Uniform-Net, Foveation-Blur-Net > Subdivided into Network model (AlexNet or ResNet18) > subdivided by 10. Sanity check observation: Epoch 0 across ALL systems per run should be the same (e.g. all weights of epoch 0 networks are the same for Reference-Net, Foveation-Texture-Net, Uniform-Net and Foveation-Blur-Net for run ID 1, but these are different for run id 2; Same within run, different across runs)

### Dataset Files
Contains the Training, Validation and Testing partition of the images.

### Filters
