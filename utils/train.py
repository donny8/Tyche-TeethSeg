__author__ = "Dohyun Kim <donny8.kim@gmail.com>"
__all__ = ['set_seed', 'DiceLoss', 'HindsightLoss', 'loss_handler', 'evaluate_testset', 'arch_curve_inference']

import time
import random
from PIL import Image

from tyche import *
from .tools import *

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class DiceLoss(tnn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        intersection = (y_pred_flat * y_true_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (y_pred_flat.sum() + y_true_flat.sum() + self.smooth)
        
        return dice_coeff

class HindsightLoss(tnn.Module):
    def __init__(self, num_outputs, base_loss_fn):
        super(HindsightLoss, self).__init__()
        self.num_outputs = num_outputs  
        self.base_loss_fn = base_loss_fn
    
    def forward(self, predictions, target):
        """
        predictions: [batch_size, num_outputs, channels, height, width]
        target: [batch_size, channels, height, width]
        """
        batch_size = predictions.size(0)
        
        all_losses = torch.zeros((batch_size, self.num_outputs)).to(predictions.device)
        
        for i in range(self.num_outputs):
            all_losses[:, i] = self.base_loss_fn(predictions[:, i], target)
        if(batch_size != 1):
            all_losses = torch.mean(all_losses, dim=0)
            min_idx = torch.argmin(all_losses).item()
            hindsight_losses = all_losses.min()
        else:
            min_idx = torch.argmin(all_losses,dim=1).item()
            hindsight_losses = all_losses.min(dim=1)[0]  # 배치 내에서 최소 loss 선택
        
        return hindsight_losses.mean(), min_idx


def loss_handler(outputs, target, useLOGDice, useBCELoss, dice_fn, bce_fn, device:"cpu"):
    loss = torch.zeros(1).to(device)
    loss_dice = torch.zeros(1).to(device)
    loss_bce = torch.zeros(1).to(device)
    if(useLOGDice):
        dice_val, min_idx = dice_fn(outputs, target)
        loss_dice = -torch.log(dice_val)
    else:
        dice_val, min_idx = dice_fn(outputs, target)
        loss_dice = 1-dice_val

    if(useBCELoss):
        loss_bce = bce_fn(outputs[:,min_idx:min_idx+1,:], target)

    loss = loss_dice + loss_bce
    return loss, loss_dice, loss_bce


def evaluate_testset(test_items:{}, result_dir:str, config_dict:dict, infer_kit:list, device, show_plot=True, save_predict_label=False):

    K = infer_kit[3]
    exp_name = config_dict["exp_name"]
    memory_usages, inference_times = np.empty(0), np.empty(0)

    for item in test_items:
        input_image, predict_label, _ = arch_curve_inference(item['image'], infer_kit, config_dict, device)

        memory_usage = predict_label.dtype.metadata['Memory Usage']
        inference_time = predict_label.dtype.metadata['Inference Time']
        memory_usages = np.append(memory_usages, memory_usage)
        inference_times = np.append(inference_times, inference_time)

        ## Show
        if show_plot:
            label = tio.ScalarImage(item['label']).data.to(torch.float32).squeeze()

            fig, axs = plt.subplots(2,K+1, figsize=(4*K,3))
            axs[0,0].imshow(label)
            axs[1,0].imshow(input_image)
            if(K>1):
                for i in range(K):
                    axs[0,i+1].imshow(predict_label[i])
                    axs[1,i+1].imshow(input_image)
                    axs[1,i+1].imshow(predict_label[i], alpha=0.5)
            else:
                axs[0,1].imshow(predict_label)
                axs[1,1].imshow(input_image)
                axs[1,1].imshow(predict_label, alpha=0.5)


            fig.suptitle('CaseID {} | pred time {}s, memory {}MB'.format(item['case_id'], 
                                                                        round(inference_time, 2), 
                                                                        round(memory_usage, 2)))
            fig.tight_layout()

            mkdir(f'./results/preds')
            mkdir(f'./results/{exp_name}/preds')
            plt.savefig(f'./results/{exp_name}/preds/{item["case_id"]}.png')
            plt.savefig(f'./results/preds/{item["case_id"]}_{exp_name}.png')
            plt.close()

        ## Save as PNG
        if save_predict_label:
            prediced_label_path = os.path.join(result_dir, '{}_pred.png'.format(item['case_id']))
            pil_img = Image.fromarray(predict_label)
            pil_img.save(prediced_label_path)


    print('Avg memory usage: ', np.mean(memory_usages), 'MB')
    print('Avg inference time: ', np.mean(inference_times), 's')


def arch_curve_inference(image_path, infer_kit, config_dict, device='cpu'):
    ## Get config
    [support_images, support_labels, noise_image, K] = infer_kit

    checkpoint_path = config_dict['checkpoint_path']

#    model = TysegXC(encoder_blocks=[16, 32, 64, 64], decoder_blocks=[16, 32, 64, 64]).to(device)
    model = TysegXC(encoder_blocks=[16, 16, 16, 16], decoder_blocks=[16, 16, 16, 16]).to(device)

    model_dict = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(model_dict)
    model.eval()


    # Load image
    image = tio.ScalarImage(image_path).data.to(torch.float32).squeeze()
    
    # Normalizatin
    image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
    image = image.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
    image = repeat(image, 'batch 1 1 width height -> batch K 1 width height', K=K)

    # Prediction
    t = time.time()
    memory_uasge_before = print_memory_usage('Before sliding_window_inference', 2)

    outputs = model(support_images=support_images.to(device), support_labels=support_labels.to(device), target_image=image, noise_image=noise_image.to(device))
    pred_label = torch.sigmoid(outputs).squeeze().detach().cpu().numpy()
    pred_label[pred_label < 0.4] = 0
    
    inference_time = time.time()-t
    memory_uasge_after = print_memory_usage('After sliding_window_inference', 2)
    memory_usage = memory_uasge_after-memory_uasge_before

    ## Finalize
    metadata = {'Inference Time':inference_time,
                'Memory Usage':memory_usage}
   
    dt = np.dtype(np.float64, metadata=metadata)
    pred_label = np.array(pred_label*255, dtype=dt)

    return image[0][0].detach().cpu().numpy().squeeze(), pred_label, model
