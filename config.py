

image_width=768
image_height=768
target_size=768
gaussian_heatmap_size=1024
start_position_region=0.15
size_of_heatmap_region=1-start_position_region*2
start_position_affinity=0.15
size_of_heatmap_affinity=1-start_position_affinity*2
height_of_box=64.0
expand_small_box=5
batch_size_synthtext=1
batch_size_word=3
epochs_end=2000
nb_epochs_change_lr=20
prob_syn=0.4 #0.6
path_saved = "/media/aimenext/Newdisk/cuongdx/ufj/models"
synth_data = "/home/aimenext/cuongdx/craft/data/hoyu/SynthText"
# OWN_DATA=["/home/aimenext/cuongdx/craft/data/license/all_1216"]
word_data = ["/home/aimenext/cuongdx/craft/data/hoyu/all",
            # "/home/aimenext/cuongdx/craft/data/hoyu/craft_1907",
            # "/home/aimenext/cuongdx/craft/data/hoyu/craft_2508"
             "/home/aimenext/cuongdx/craft/data/ufj/word",
             "/home/aimenext/cuongdx/craft/data/ufj/word_0120",
             "/home/aimenext/cuongdx/craft/data/ufj/word_0120_duplicate"
            ]
char_data=["/home/aimenext/cuongdx/craft/data/ufj/char"]
# PRE_TRAINED="/home/ubuntu/cuongdx/craft/models/pretrained"
