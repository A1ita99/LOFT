from random import choice
from string import ascii_uppercase
import os
from configs import global_config, paths_config
import wandb
from training.coaches.single_id_coach import SingleIDCoach





from data_loader import get_loader

def run_LOFT(run_name='', use_wandb=False, use_multi_id_training=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    if use_wandb:
        run = wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=global_config.run_name)
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)


    #data_loader settings
    celeba_image_dir = "./test_images"
    attr_path = "./test_images/CelebAMask-HQ-attribute-anno_test.txt"
    selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    data_loader = get_loader(celeba_image_dir, attr_path, selected_attrs, num_workers=1)


    coach = SingleIDCoach(data_loader, use_wandb)
    coach.train()

    return global_config.run_name



if __name__ == '__main__':
    run_LOFT(run_name='', use_wandb=False, use_multi_id_training=False)
