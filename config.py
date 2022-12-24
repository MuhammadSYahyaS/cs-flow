'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''

# device settings
device = 'cuda'  # 'cuda' or 'cpu'

# data settings
dataset_path = "data_BTAD/images"  # parent directory of datasets
# class_name = "01"  # dataset subdirectory
# modelname = "01"  # export evaluations/logs with this name
# class_name = "02"  # dataset subdirectory
# modelname = "02"  # export evaluations/logs with this name
class_name = "03"  # dataset subdirectory
modelname = "03"  # export evaluations/logs with this name
pre_extracted = True  # were feature preextracted with extract_features?

img_size = (768, 768)  # image size of highest scale, others are //2, //4
assert img_size[0] % 128 == 0 and img_size[1] % 128 == 0, "image width/height should be a multiple of 128"

img_dims = [3] + list(img_size)

# transformation settings
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# network hyperparameters
n_scales = 3  # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
clamp = 3  # clamping parameter
max_grad_norm = 1e0  # clamp gradients to this norm
n_coupling_blocks = 4  # higher = more flexible = more unstable
fc_internal = 1024  # * 4 # number of neurons in hidden layers of s-t-networks
lr_init = 2e-4  # inital learning rate
use_gamma = True

extractor = "effnetB5"  # pretrained feature extractor model
n_feat = {"effnetB5": 304, "resnet34": 256, "ssd_mobilenet_v3_large": 160}[extractor]  # dependend from feature extractor
map_size = (img_size[0] // 12, img_size[1] // 12)

# dataloader parameters
batch_size = 1 # testing
# batch_size = 8  # actual batch size is this value multiplied by n_transforms(_test) # effnetB5 / ssd_mobilenet_v3_large
# batch_size = 4  # actual batch size is this value multiplied by n_transforms(_test) # resnet34
kernel_sizes = [3] * (n_coupling_blocks - 1) + [5]

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 40  # total epochs = meta_epochs * sub_epochs
sub_epochs = 6  # evaluate after this number of epochs

# output settings
verbose = True
hide_tqdm_bar = False
save_model = True
