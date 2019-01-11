from fcn8s_tensorflow import FCN8s
from data_generator.batch_generator import BatchGenerator
from helpers.visualization_utils import print_segmentation_onto_image, create_video_from_images
from cityscapesscripts.helpers.labels import TRAINIDS_TO_COLORS_DICT, TRAINIDS_TO_RGBA_DICT

from math import ceil
import time
import matplotlib.pyplot as plt

# TODO: Set the paths to the images.
train_images = '../datasets/data/leftImg8bit/train/'
val_images = '../datasets/data/leftImg8bit/val/'
test_images = '../datasets/data/leftImg8bit/test/'

# TODO: Set the paths to the ground truth images.
train_gt = '../datasets/data/gtFine/train/'
val_gt = '../datasets/data/gtFine/val/'

# Put the paths to the datasets in lists, because that's what `BatchGenerator` requires as input.
train_image_dirs = [train_images]
train_ground_truth_dirs = [train_gt]
val_image_dirs = [val_images]
val_ground_truth_dirs = [val_gt]

num_classes = 21 # TODO: Set the number of segmentation classes.

train_dataset = BatchGenerator(image_dirs=train_image_dirs,
                               image_file_extension='png',
                               ground_truth_dirs=train_ground_truth_dirs,
                               image_name_split_separator='leftImg8bit',
                               ground_truth_suffix='gtFine_labelIds',
                               check_existence=True,
                               num_classes=num_classes)

val_dataset = BatchGenerator(image_dirs=val_image_dirs,
                             image_file_extension='png',
                             ground_truth_dirs=val_ground_truth_dirs,
                             image_name_split_separator='leftImg8bit',
                             ground_truth_suffix='gtFine_labelIds',
                             check_existence=True,
                             num_classes=num_classes)

num_train_images = train_dataset.get_num_files()
num_val_images = val_dataset.get_num_files()

print("Size of training dataset: ", num_train_images, " images")
print("Size of validation dataset: ", num_val_images, " images")


# TODO: Set the batch size. I'll use the same batch size for both generators here.
batch_size = 4

train_generator = train_dataset.generate(batch_size=batch_size,
                                         convert_colors_to_ids=False,
                                         convert_ids_to_ids=False,
                                         convert_to_one_hot=True,
                                         void_class_id=None,
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         brightness=False,
                                         flip=0.5,
                                         translate=False,
                                         scale=False,
                                         gray=False,
                                         to_disk=False,
                                         shuffle=True)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     convert_colors_to_ids=False,
                                     convert_ids_to_ids=False,
                                     convert_to_one_hot=True,
                                     void_class_id=None,
                                     random_crop=False,
                                     crop=False,
                                     resize=False,
                                     brightness=False,
                                     flip=False,
                                     translate=False,
                                     scale=False,
                                     gray=False,
                                     to_disk=False,
                                     shuffle=True)

# Print out some diagnostics to make sure that our batches aren't empty and it doesn't take forever to generate them.
start_time = time.time()
images, gt_images = next(train_generator)
print('Time to generate one batch: {:.3f} seconds'.format(time.time() - start_time))
print('Number of images generated:' , len(images))
print('Number of ground truth images generated:' , len(gt_images))

# Generate batches from the train_generator where the ground truth does not get converted to one-hot
# so that we can plot it as images.
example_generator = train_dataset.generate(batch_size=batch_size,
                                           convert_to_one_hot=False)

# Generate a batch.
example_images, example_gt_images = next(example_generator)

i = 0 # Select which sample from the batch to display below.

figure, cells = plt.subplots(1, 2, figsize=(16,8))
cells[0].imshow(example_images[i])
cells[1].imshow(example_gt_images[i])

plt.figure(figsize=(16, 8))
plt.imshow(example_gt_images[i])