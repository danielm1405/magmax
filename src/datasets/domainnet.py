import os
import os.path
import json
import hashlib

import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url



class _DomainNet:
    DOMAINS = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    CLASSES = ['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cell_phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint_can', 'paintbrush', 'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 'river', 'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

    @classmethod
    def get_complementary_domains(cls, domains):
        return [x for x in cls.DOMAINS if x not in domains]
    
    @classmethod
    def get_complementary_classes(cls, classes):
        return [x for x in cls.CLASSES if x not in classes]

    @classmethod    
    def get_md5(cls, subset_config):
        return hashlib.md5(json.dumps(subset_config).encode('utf-8')).hexdigest()

    def __init__(self, root, train=True, transform=None, target_transform=None, subset_config=None):
        self.fpath = os.path.join(os.path.abspath(root),
                                  'domainnet',
                                  'train' if train else 'test')
        
        if not subset_config:
            subset_config = {'domains': self.DOMAINS, 'classes': self.CLASSES}
        assert 'domains' in subset_config and 'classes' in subset_config    
        print(f"Using subset config: {subset_config}")
        
        id = self.get_md5(subset_config)
        new_fpath = os.path.join(os.path.abspath(root),
                                    'domainnet_subsets',
                                    id,
                                    'train' if train else 'test')
        if not os.path.exists(new_fpath):
            print("Creating a subset from scratch")
            for domain in subset_config['domains']:
                for cls in subset_config['classes']:                    
                    source_cls_path = os.path.join(self.fpath, domain, cls)
                    if not os.path.exists(source_cls_path):
                        print(f"Path {source_cls_path} does not exist, skipping!")
                        continue
                    
                    new_cls_path = os.path.join(new_fpath, cls)
                    os.makedirs(new_cls_path, exist_ok=True)
                    
                    for img_name in os.listdir(source_cls_path):
                        os.symlink(os.path.join(source_cls_path, img_name),
                                   os.path.join(new_cls_path, img_name))

            with open(os.path.join(new_fpath, 'config.json'), 'w') as config_file:
                json.dump(subset_config, config_file)
        else:
            print("Using subset that already exists")
            
        self.fpath = new_fpath
        self.data = datasets.ImageFolder(self.fpath, transform=transform)


class DomainNet:
    BASE_CLASS = _DomainNet
    
    default_class_order = [76, 44, 168, 324, 291, 274, 118, 223, 113, 269, 279, 105, 6, 30, 248, 56, 230, 101, 148, 143, 307, 33, 325, 19, 221, 71, 161, 130, 200, 306, 94, 195, 224, 320, 340, 253, 37, 158, 189, 201, 97, 150, 18, 159, 311, 334, 266, 321, 284, 203, 182, 67, 214, 136, 267, 296, 4, 41, 226, 11, 283, 126, 210, 193, 132, 167, 72, 16, 63, 7, 241, 251, 282, 254, 149, 218, 204, 28, 302, 222, 103, 231, 96, 106, 146, 121, 270, 119, 74, 145, 120, 38, 252, 114, 256, 140, 225, 22, 124, 185, 322, 215, 264, 42, 23, 292, 265, 174, 313, 129, 285, 69, 43, 34, 301, 219, 257, 21, 229, 48, 234, 309, 66, 310, 24, 202, 261, 68, 271, 176, 316, 186, 196, 315, 319, 134, 54, 60, 70, 288, 194, 245, 268, 147, 337, 281, 326, 235, 47, 187, 243, 213, 181, 255, 212, 92, 77, 294, 17, 49, 331, 144, 197, 190, 155, 117, 323, 53, 50, 341, 191, 178, 308, 100, 61, 237, 83, 0, 1, 14, 205, 238, 40, 81, 183, 220, 112, 98, 135, 31, 35, 276, 172, 258, 125, 286, 295, 343, 25, 184, 65, 32, 26, 151, 84, 206, 10, 39, 318, 91, 13, 87, 99, 277, 240, 62, 156, 262, 188, 211, 166, 317, 247, 328, 289, 217, 169, 232, 330, 246, 173, 46, 51, 244, 228, 208, 64, 152, 249, 179, 198, 55, 20, 15, 128, 154, 303, 116, 171, 278, 59, 73, 45, 180, 177, 314, 110, 312, 141, 36, 27, 163, 175, 339, 95, 78, 259, 162, 142, 272, 57, 233, 260, 2, 137, 209, 102, 300, 216, 304, 3, 227, 89, 104, 52, 263, 239, 90, 93, 170, 133, 80, 336, 297, 8, 250, 79, 82, 108, 138, 280, 88, 192, 333, 344, 275, 298, 327, 157, 9, 85, 287, 335, 127, 111, 290, 299, 29, 242, 329, 293, 199, 236, 58, 75, 86, 5, 165, 273, 107, 153, 123, 332, 305, 160, 207, 139, 338, 122, 342, 131, 164, 12, 115, 109]
    default_domain_order = [0, 5, 4, 1, 3, 2]
        
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 subset_config=None):
        self.train_dataset = _DomainNet(
            location,
            train=True,
            transform=preprocess,
            target_transform=None,
            subset_config=subset_config,
        ).data
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = _DomainNet(
            location,
            train=False,
            transform=preprocess,
            target_transform=None,
            subset_config=subset_config,
        ).data
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        self.classnames = [c.replace('_', ' ') for c in list(self.train_dataset.class_to_idx.keys())]
