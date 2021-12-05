# -*- coding: utf-8 -*-

# Code for paper:
# [Title]  - "GCL"
# [Author] - Junbin Zhang
# [Github] - https://github.com/

import numpy as np
import os
from easydict import EasyDict as edict

cfg = edict()

#########################################################################################
# import gc
# 删除list以节省内存
# del feature_list
# gc.collect()

# 使用Numpy保存特征为.npy格式，以节省存储空间和提高读写速度
# with open(os.path.join(base_path, publish_path, './features.npy'), 'wb') as f:
#     np.save(f, feat_arr)
#########################################################################################
cfg.GPU_id = '0'  # '-1'
cfg.device = 'cuda:0'      # 'cpu'
cfg.seed = 100
cfg.mode = 'train'
cfg.model = 'GCN'     # ', GAT, GraphSage'
cfg.hidden_dim = 128    #
cfg.num_layers = 2
cfg.optimizer = 'adam'       # sgd, rmsprop or adagrad
cfg.weight_decay = 5e-4
cfg.dropout = 0.0
cfg.lr = 0.004
# cfg.lr = '[0.0001]*100'  # '[0.0001]*6000'
# cfg.num_iters = len(eval(cfg.lr))
cfg.optimizer_scheduler = 'step'     # 'cos'
# 'step'
cfg.optimizer_decay_step = 0
cfg.optimizer_decay_rate = 0
# 'cos'
cfg.optimizer_restart = 0
cfg.epochs = 200
cfg.batch_size = [8, 1, 1]
cfg.embNode2Vec = True

cfg.dataset = "gtea"  # breakfast, , 50salads,
cfg.num_classes = 0
cfg.num_node_features = 2048   # + 128
cfg.DATA_PATH = '/home/cpslabzjb/zjb/projects/zjb/PGCL/data/'
cfg.GT_PATH = os.path.join(cfg.DATA_PATH, '/gt.json')
cfg.best_model_path = '/home/cpslabzjb/zjb/projects/zjb/PGCL'

cfg.model_dir = 'models'
cfg.result_dir = 'results'
cfg.label2index_dict = {}
cfg.index2label_dict = {}
cfg.split_ratio = [0.8, 0.1, 0.1]



'''
cfg.CLASS_DICT = {"SIL": 0, "take_knife": 1, "cut_bun": 2, "take_butter": 3, "smear_butter": 4, "take_topping": 5,
                  "put_toppingOnTop": 6, "put_bunTogether": 7, "take_plate": 8, "pour_cereals": 9, "pour_milk": 10,
                  "take_bowl": 11, "stir_cereals": 12, "butter_pan": 13, "crack_egg": 14, "fry_egg": 15,
                  "put_egg2plate": 16, "add_saltnpepper": 17, "pour_oil": 18, "take_eggs": 19, "cut_orange": 20,
                  "squeeze_orange": 21, "take_glass": 22, "pour_juice": 23, "take_squeezer": 24, "stirfry_egg": 25,
                  "stir_egg": 26, "pour_egg2pan": 27, "take_cup": 28, "pour_coffee": 29, "pour_sugar": 30,
                  "stir_coffee": 31, "spoon_sugar": 32, "cut_fruit": 33, "put_fruit2bowl": 34, "peel_fruit": 35,
                  "stir_fruit": 36, "add_teabag": 37, "pour_water": 38, "stir_tea": 39, "spoon_powder": 40,
                  "stir_milk": 41, "pour_flour": 42, "stir_dough": 43, "pour_dough2pan": 44, "fry_pancake": 45,
                  "put_pancake2plate": 46, "spoon_flour": 47}

cfg.CLASS_DICT = {"garbage": 0, "reach_bag": 1, "reach_bread": 2, "carry_bread": 3, "carry_bag": 4, "reach_knife": 5,
                  "carry_knife": 6, "cut_bread": 7, "reach_butter": 8, "carry_butter": 9, "liftopen_lidButter": 10,
                  "scoop_butter": 11, "smear_butter": 12, "pressclose_lidButter": 13, "move": 14, "turn": 15,
                  "reach_topping": 16, "carry_topping": 17, "liftopen_lidTopping": 18, "reach_toppingpiece": 19,
                  "carry_toppingpiece": 20, "pressclose_lidTopping": 21, "reach_fridge": 22, "open_fridge": 23,
                  "close_fridge": 24, "reach_cuttingboard": 25, "carry_cuttingboard": 26, "reach_lidButter": 27,
                  "carry_lidButter": 28, "reach_drawer": 29, "pull_drawer": 30, "push_drawer": 31, "walk": 32,
                  "reach_cabinet": 33, "carry_plate": 34, "shift": 35, "reach_cloth": 36, "wipe_cloth": 37,
                  "reach_cereal": 38, "carry_cereal": 39, "reach_plate": 40, "open_cabinet": 41, "close_cabinet": 42,
                  "reach_fork": 43, "carry_fork": 44, "open_cereal": 45, "pour_cereal": 46, "close_cereal": 47,
                  "screwopen_capMilk": 48, "pour_milk": 49, "screwclose_capMilk": 50, "carry_milk": 51, "wait": 52,
                  "reach_milk": 53, "reach_bowl": 54, "carry_bowl": 55, "hold_bowl": 56, "carry_spoon": 57,
                  "stir_cereal": 58, "reach_spoon": 59, "carry_sugar": 60, "screwopen_capSugar": 61,
                  "carry_capSugar": 62, "scoop_sugar": 63, "reach_sugar": 64, "screwclose_capSugar": 65,
                  "carry_capMilk": 66, "reach_capMilk": 67, "pour_sugar": 68, "reach_handle": 69, "carry_handle": 70,
                  "reach_eggcarton": 71, "carry_eggcarton": 72, "liftopen_eggcarton": 73, "reach_egg": 74,
                  "carry_egg": 75, "pressclose_eggcarton": 76, "crack_egg": 77, "fry_egg": 78, "reach_salt": 79,
                  "carry_salt": 80, "pour_salt": 81, "transfer_egg2plate": 82, "reach_spatula": 83, "carry_spatula": 84,
                  "reach_oil": 85, "screwopen_capOil": 86, "pour_oil": 87, "screwclose_capOil": 88, "carry_pan": 89,
                  "hold_egg": 90, "transfer_egg2pan": 91, "reach_pepper": 92, "grind_pepper": 93, "carry_pepper": 94,
                  "reach_cup": 95, "carry_cup": 96, "pickup_egg": 97, "carry_oil": 98, "screwopen_capPepper": 99,
                  "twistonOff_faucet": 100, "wash": 101, "turnonOff_stove": 102, "reach_object": 103,
                  "reach_stove": 104, "carry_cloth": 105, "screwopen_capSalt": 106, "melt_butter": 107,
                  "transfer_butter2pan": 108, "screwclose_capPepper": 109, "pour_pepper": 110, "reach_pan": 111,
                  "screwclose_capSalt": 112, "hold_handle": 113, "hold_pan": 114, "carry_capSalt": 115, "wipe": 116,
                  "reach_orange": 117, "carry_orange": 118, "cut_orange": 119, "reach_juicer": 120, "carry_juicer": 121,
                  "squeeze_orange": 122, "pour_juice": 123, "hold_orange": 124, "hold_cup": 125, "hold_spatula": 126,
                  "scramble_egg": 127, "reach_faucet": 128, "carry_whisk": 129, "reach_whisk": 130, "carry_coffee": 131,
                  "pour_coffee": 132, "reach_coffee": 133, "hold_coffee": 134, "hold_milk": 135, "stir_coffee": 136,
                  "peel_fruit": 137, "cut_fruit": 138, "reach_fruit": 139, "carry_fruit": 140, "hold_fruit": 141,
                  "reach_peeler": 142, "carry_peeler": 143, "transfer_fruit2bowl": 144, "stir_salad": 145,
                  "open_teabag": 146, "carry_teabag": 147, "carry_kettle": 148, "pour_water": 149, "reach_teabox": 150,
                  "openClose_teabox": 151, "reach_teabag": 152, "reach_kettle": 153, "carry_teabox": 154,
                  "carry_chocolatepowder": 155, "reach_chocolatepowder": 156, "liftopen_lidChoco": 157,
                  "carry_lidChoco": 158, "scoop_chocolatepowder": 159, "stir_chocolate": 160, "reach_lidChoco": 161,
                  "pressclose_lidChoco": 162, "carry_flour": 163, "scoop_flour": 164, "transfer_flour2bowl": 165,
                  "pressclose_lidFlour": 166, "stir_pancake": 167, "cook_pancake": 168, "reach_flour": 169,
                  "liftopen_lidFlour": 170, "pour_flour": 171, "pour_pancake": 172, "transfer_pancake2plate": 173,
                  "carry_lidFlour": 174, "reach_lidFlour": 175, "flip_pancake": 176, "reach_sink": 177}
'''
####################################################################################

cfg.MODAL = 'all'

cfg.NUM_WORKERS = 8
cfg.LAMBDA = 0.01
cfg.R_EASY = 5
cfg.R_HARD = 20
cfg.m = 3
cfg.M = 6
cfg.TEST_FREQ = 100
cfg.PRINT_FREQ = 20
cfg.CLASS_THRESH = 0.2
cfg.NMS_THRESH = 0.6
cfg.CAS_THRESH = np.arange(0.0, 0.25, 0.025)
cfg.ANESS_THRESH = np.arange(0.1, 0.925, 0.025)
cfg.TIOU_THRESH = np.linspace(0.1, 0.7, 7)
cfg.UP_SCALE = 24
cfg.FEATS_FPS = 25
cfg.NUM_SEGMENTS = 750
