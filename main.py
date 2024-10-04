from tyche import *
from utils.data import *
from utils.train import *
from utils.tools import *


import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--MX', action='store_true', help='Whether to use max-projected image for augmentation')
parser.add_argument('--BCE', action='store_true', help='Whether to use BCE along with DICE loss for training')
parser.add_argument('--LOG', action='store_true', help='Whether to use negative log dice loss for training')
parser.add_argument('--train', action='store_true', help='Whether to train a new model')
parser.add_argument('--test', action='store_true', help='Whether to evaluate a pretrained model')

parser.add_argument('--support', type=str, default='normal', choices=['normal', 'mix', 'noisy'], help='Which support set to choose')
parser.add_argument('--K', type=int, default=1, help='The number of stochastic input')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()



num_workers = 1
in_channels = 1
out_channels = 1
learning_rate = 1e-4
val_every = 5

model_name = 'tyche'
dataset_dir = './data/done_img'
labels_dir = './data/done_label'



if __name__ == "__main__":

    K = args.K
    seed = args.seed
    device = args.device
    batch_size = args.batch
    max_epochs = args.epochs

    useMAXAug = args.MX
    useBCELoss = args.BCE
    useLOGDice = args.LOG
    onlyTrain = args.train
    onlyTest = args.test
    support = args.support

    set_seed(seed)

    ## =========================================================  Step 0: Datasets  ========================================================= ##
    case_ids = list()
    for file_name in os.listdir(labels_dir):
        name, exe = file_name.split('.')
        name_splits = name.split('_')
        if exe == 'png':
            case_ids.append(name_splits[0])

    exp_name = f'ArchCurveSeg_{model_name}_D{len(case_ids)}_Ep{max_epochs}_B{batch_size}_K{K}_S{support}'
    if(useMAXAug):
        exp_name += '_MXAug'
    if(useBCELoss):
        exp_name += '_BCE'
    if(useLOGDice):
        exp_name += '_LOG'


    result_dir = './results/{}'.format(exp_name)
    writer = SummaryWriter(log_dir=result_dir)
    checkpoints_path = os.path.join(result_dir, 'checkpoints')
    mkdir(result_dir)
    mkdir(checkpoints_path)


    ## ===================================================  Step 1: Prepare the Training Materials  =================================================== ##
    # Load the dataloaders
    loaders = load_dataloaders(case_ids, dataset_dir, batch_size, num_workers,  support, useMAXAug=useMAXAug)

    # Load the model
    model = TysegXC(encoder_blocks=[8, 16, 32, 64], decoder_blocks=[8, 16, 32, 64]).to(device)


    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = int(max_epochs/3), gamma=0.7)
    dice_fn = DiceLoss()
    dice_fn_min = HindsightLoss(num_outputs=K, base_loss_fn=dice_fn)
    bce_fn = BCEWithLogitsLoss()


    ## ======================================================  Step 2: Model Train & Val  ====================================================== ##
    support_data = next(iter(loaders[3])) # supp_loader
    support_images, support_labels = support_data['image'].unsqueeze(0).to(device), support_data['label'].unsqueeze(0).to(device)
    noise_image_fixed = torch.randn(1, K, 1, 512, 512)
    infer_kit = [support_images, support_labels, noise_image_fixed, K]
    np.savez(f'./results/{exp_name}/support.npz', support_images = support_images, support_labels=support_labels, noise_image = noise_image_fixed, K=K)
    support_images_tr = repeat(support_images, '1 S 1 width height -> batch S 1 width height', batch=batch_size)            
    support_labels_tr = repeat(support_labels, '1 S 1 width height -> batch S 1 width height', batch=batch_size)            

    if(onlyTrain):
        best_val = {
                'value': float('-inf'),
                'epoch': -1
                }
        train_losses = []
        train_dice = []
        train_bce = []
        val_dice = []
        
        ## ======================================================  Train Phase  ====================================================== ##
        for i, epoch in tqdm(enumerate(range(max_epochs)), total=max_epochs, desc='Train epoch', ascii = ' ='):

            model.train()
            epoch_losses = np.empty(0)
            epoch_dice = np.empty(0)
            epoch_bce = np.empty(0)
            for batch_data in tqdm(loaders[0]):
                if batch_data['label'].max() == 0:
                    continue

                data, target = batch_data['image'].unsqueeze(1), batch_data['label'].unsqueeze(1)
                data = repeat(data, 'batch 1 1 width height -> batch K 1 width height', K=K)
                if device=='cuda':
                    data, target = data.to(device), target.to(device)

                noise_image = torch.randn(batch_size, K, 1, 512, 512).to(device)
                outputs = model(support_images=support_images_tr, support_labels=support_labels_tr, target_image=data, noise_image=noise_image)
                loss, loss_dice, loss_bce = loss_handler(outputs, target/target.max(), useLOGDice, useBCELoss, dice_fn_min, bce_fn, device)
                    

                if(useMAXAug):
                    aug = batch_data['imagemax'].unsqueeze(1)
                    aug = repeat(aug, 'batch 1 1 width height -> batch K 1 width height', K=K)
                    if device == 'cuda':
                        aug = aug.to(device)
                    outputs_aug = model(support_images=support_images_tr, support_labels=support_labels_tr, target_image=aug, noise_image=noise_image)
                    loss_aug, loss_dice_aug, loss_bce_aug = loss_handler(outputs_aug, target/target.max(), useLOGDice, useBCELoss, dice_fn_min, bce_fn, device)

                    loss += loss_aug
                    loss_dice += loss_dice_aug
                    loss_bce += loss_bce_aug

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_losses = np.append(epoch_losses, loss.item())
                epoch_dice = np.append(epoch_dice, loss_dice.item())
                epoch_bce = np.append(epoch_bce, loss_bce.item())

                del data, target, outputs

            train_losses.append(np.mean(epoch_losses))
            train_dice.append(np.mean(epoch_dice))
            train_bce.append(np.mean(epoch_bce))
            
            writer.add_scalar('Train/Train Loss', np.mean(epoch_losses), epoch)


            ## ======================================================  Validate Phase  ====================================================== ##
            val_loss = 0
            if epoch % val_every == 0: 
                model.eval()
                with torch.no_grad():
                    for val_idx, val_batch_data in tqdm(enumerate(loaders[1]), total=len(loaders[1]), desc=f'Validation'): # val_loader

                        if val_batch_data['label'].max() == 0:
                            continue

                        val_images, val_target = val_batch_data['image'].unsqueeze(1), val_batch_data['label'].unsqueeze(1)
                        val_images = repeat(val_images, 'batch 1 1 width height -> batch K 1 width height', K=K)
                        if device=='cuda':
                            val_images, val_target = val_images.to(device), val_target.to(device)

                        noise_image = torch.randn(1, K, 1, 512, 512).to(device)
                        val_outputs = model(support_images=support_images, support_labels=support_labels, target_image=val_images, noise_image=noise_image)
                        min_dice, min_idx = dice_fn_min(val_outputs, val_target/val_target.max())
                        val_loss += min_dice.item()
                        if val_idx == 0:
                            input_image = val_images.squeeze()
                            predict_image = val_outputs.squeeze()
                            label_image = val_target.squeeze()
                            if(K==1):
                                concat_image = torch.rot90(torch.concat((input_image, label_image, predict_image))).unsqueeze(0)
                            else:
                                concat_image = torch.rot90(torch.concat((input_image[min_idx], label_image, predict_image[min_idx]))).unsqueeze(0)

                            writer.add_image('Valication/Prediction', concat_image, epoch)

                            del predict_image, label_image, concat_image

                    dice_score = val_loss / len(loaders[1])
                    val_dice.append(dice_score)
                    
                    writer.add_scalar('Valication/Val Mean Dice', dice_score, epoch)
                    print('Epoch: ', epoch, 'Val Mean Dice: ', dice_score)

                    if dice_score > best_val['value']:
                        best_val['value'] = dice_score
                        best_val['epoch'] = epoch

                        torch.save(model.state_dict(), os.path.join(checkpoints_path, 'best.pth'))
                        writer.add_scalar('Valication/Best Val Dice', dice_score, epoch)

            scheduler.step()
            torch.save(model.state_dict(), os.path.join(checkpoints_path, 'last.pth'))

        save_loss_plot(train_losses, 'Train_Loss', exp_name, './results')
        save_loss_plot(train_dice, 'Train_DICE', exp_name, './results')
        save_loss_plot(train_bce, 'Train_BCE', exp_name, './results')
        save_loss_plot(val_dice, 'Val_DICE', exp_name, './results')


    ## ======================================================  Step 3: Model Test  ====================================================== ##
    if(onlyTest):
        result_dir = './results/{}'.format(exp_name)
        config_dict = {
                    'seed':seed,
                    'exp_name':exp_name,
                    'checkpoint_path':os.path.join(result_dir, 'checkpoints', 'best.pth'),
                        }        
        evaluate_testset(loaders[2], result_dir, config_dict, infer_kit, device, show_plot=True, save_predict_label=False)