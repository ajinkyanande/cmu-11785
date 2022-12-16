To run the entire pipeline : python hw2p2.py

Experimentation :

    01) Epochs       : 120              : Number of epochs were kept large with Cosine Annealing learning rate scheduler in order to train for close to minimum learning rate for atleast 15 epochs.

    02) Batch Size   : 256              : Maximum that fits the hardware.

    03) Weight Decay : 1e-3             : Values 1e-4 and 1e-5 were tried as well but he model seem to overfit because training accuracy reached 99.5% with validation accuracy roughly 80%.

    04) Scheduler    : Cosine Annealing : Starts with 0.1 and is decayed till is 1e-5 for 120 epochs.

    05) Optimizer    : SGD              : SGD with momentum 0.9 and larger weight decay of 1e-3 was used for training.

    06) Network Weights Initialization : Network Weights were initialized with previously trained ConvNext with Weight Decay = 1e-4. (cosann_convnext is the wandb run used for weight initialization and cosann_convnext_final is the final submission run).
    
    07) Image Transforms : Following image transforms were used. The choices were made such that the transforms represent reality (e.g. RandomVertical was not used because inverted image of human is hard to come by) :
                               RandomPerspective(distortion_scale=0.2)
                               RandomRotation(degrees=10)
                               GaussianBlur(kernel_size=3)
                               RandomAdjustSharpness(sharpness_factor=1.5)
                               RandomAutocontrast()
                               RandomHorizontalFlip()
                               ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    08) Recognition Network  : ResNet34 and ConvNext were tried. ConvNext did better by just 1% for Face Recognition task but performed much better with Face Verification task. ConvNext with output size = 7000 (number of classes in the dataset) was finally used.

    09) Verification Network : ConvNext trained for Face Recognition task was used to extract the known and unknown image features. Cosine Similarity was used as metric with the extracted features for the task.

    10) WandB Runs : wandb project is made public and can be found here : https://wandb.ai/ajinkyanande111/hw2p2
