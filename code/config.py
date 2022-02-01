import easydict
from easydict import EasyDict as edict

config = edict()

config.path = edict()
config.path.train_image_dir = '../data/train/'
config.path.val_image_dir = '../data/val/'

config.model = edict()
config.model.name = 'igpt_s'
config.model.img_size = 32

config.train = edict()

