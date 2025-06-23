import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
import cv2
from PIL import Image
def save_image(path, image):
  """Saves an image to disk.

  NOTE: The input image (if colorful) is assumed to be with `RGB` channel order
  and pixel range [0, 255].

  Args:
    path: Path to save the image to.
    image: Image to save.
  """
  if image is None:
    return

  assert len(image.shape) == 3 and image.shape[2] in [1, 3]
  cv2.imwrite(path, image[:, :, ::-1])


class SingleIDCoach(BaseCoach):
    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self):
        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)
        use_ball_holder = True

        # StarGAN settings
        c_dim = 5
        selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        import sys
        sys.path.append('../../StarGAN/')
        from StarGAN import stargan
        starG = stargan.Generator(conv_dim=64, c_dim=5, repeat_num=6)
        G_path = paths_config.stargan_weight_path
        starG.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        starG.to(global_config.device)

        # process images
        for image, c_org, filename in tqdm(self.data_loader):
            from torchvision import transforms
            trans_resize_tool = transforms.Resize((128, 128))
            trans_resize_tool_256 = transforms.Resize((256, 256))
            image_name = filename[0].split('.')[0]
            print(image_name)
            image=image.to(global_config.device)
            save_path=paths_config.output_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            ori_image = image.clone().detach()
            ori_image = (ori_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
            ori_image_256 = Image.fromarray(ori_image, mode='RGB').resize((256, 256))
            ori_image_256.save(save_path+f"/{image_name}_ori_image_256.png")

            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)
            label = []
            for i in range(c_dim):
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
                label.append(c_trg.to(global_config.device))



            self.restart_training()


            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None
            if hyperparameters.use_last_w_pivots:
                w_pivot = self.load_inversions(w_path_dir, image_name)
            elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                w_pivot = self.calc_inversions(image, image_name)
            w_pivot = w_pivot.to(global_config.device)

            # ALO
            z = w_pivot.clone().detach().cuda()
            z.requires_grad = True
            z_optimizer = torch.optim.Adam([z], lr=1e-2)
            for i in tqdm(range(hyperparameters.alo_steps)):
                z = z.detach().requires_grad_()
                x_rec = self.forward(z)
                x_rec = x_rec.to(global_config.device)
                loss_adv = 0.0
                out_recs = []
                out_oris = []

                for c_trg in label[:]:
                    out_rec = starG(trans_resize_tool(x_rec), c_trg)
                    out_ori = starG(trans_resize_tool(image), c_trg)
                    out_recs.append(out_rec)
                    out_oris.append(out_ori)
                    loss, l2_loss_val, loss_lpips = self.calc_loss(trans_resize_tool_256(x_rec), trans_resize_tool_256(image), image_name,
                                                             self.G, use_ball_holder, z)
                    loss_adv +=loss+torch.mean((trans_resize_tool(image) - out_rec) ** 2)
                loss_adv=-loss_adv*hyperparameters.pt_att_lambda
                z_optimizer.zero_grad()
                loss_adv.backward()
                z_optimizer.step()

            # Fine-tuning
            real_images_batch = image.to(global_config.device)
            temp_z=z.detach().clone()
            for i in tqdm(range(hyperparameters.ft_steps)):
                loss_adv = 0.0

                generated_images = self.forward(temp_z)
                generated_images=generated_images.to(global_config.device)

                out_recs_ = []
                out_oris_ = []
                for c_trg in label:
                    out_rec = starG(trans_resize_tool(generated_images), c_trg)
                    out_ori = starG(trans_resize_tool(real_images_batch), c_trg)
                    out_recs_.append(out_rec)
                    out_oris_.append(out_ori)
                    loss_adv += torch.mean((trans_resize_tool(real_images_batch) - out_rec) ** 2)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, z)
                loss_adv=loss_adv * hyperparameters.pt_att_lambda
                loss1=loss+loss_adv
                self.optimizer.zero_grad()
                loss1.backward()
                self.optimizer.step()




                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0
                global_config.training_step += 1

                #save images
                if(i==hyperparameters.ft_steps-1):
                    rec_image = generated_images.clone().detach()
                    rec_image = (rec_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
                    rec_image_256 = Image.fromarray(rec_image, mode='RGB').resize((256, 256))
                    rec_image_256.save(save_path + f"/{image_name}_rec_image_256.png")

            self.image_counter += 1
            # save stargan output images
            stargan_output_path=paths_config.stargan_output_path
            if not os.path.exists(stargan_output_path):
                os.mkdir(stargan_output_path)
            stargan_results_=[]
            for num in range(len(out_recs_)):
                stargan_results_.append(0.5 * 255. * (out_recs_[num][0] + 1).detach().cpu().numpy().transpose(1, 2, 0))
            for num in range(len(out_oris_)):
                stargan_results_.append(0.5 * 255. * (out_oris_[num][0] + 1).detach().cpu().numpy().transpose(1, 2, 0))

            for num in range(len(selected_attrs)):
                save_image(stargan_output_path+f'/{image_name}_rec_{num}.png', stargan_results_[num])
                save_image(stargan_output_path+f'/{image_name}_ori_{num}.png',stargan_results_[num + len(selected_attrs)])