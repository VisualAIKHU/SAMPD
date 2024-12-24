import os
import time
import shutil
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from datasets import KAISTPed
from inference import val_epoch, save_results
from model import MultiBoxLoss
from utils import utils
from utils.evaluation_script import evaluate
from train_utils import create_teacher_student
from train import train_epoch

torch.backends.cudnn.benchmark = False

utils.set_seed(seed=9)

def main():
    """Train and validate a model"""

    args = config.args
    train_conf = config.train
    epochs = train_conf.epochs
    phase = "Multispectral"
    
    original_GT_example_txt = f"./imgAvgs/{config.args.dataset_type}_img_rgb_avg_{config.args.MP}.txt"
    Edited_GT_example_txt = f"./imgAvgs/{config.args.dataset_type}_img_rgb_avg_{config.args.MP}_Edit.txt"
    if os.path.exists(Edited_GT_example_txt):
        os.remove(Edited_GT_example_txt)

    with open(original_GT_example_txt, 'r', encoding='utf-8') as infile:
        content = infile.read()
    
    with open(Edited_GT_example_txt, 'w', encoding='utf-8') as outfile:
        outfile.write(content)
    
    # Initialize model or load checkpoint
    s_model, optimizer, optim_scheduler, s_epochs, \
    t_model,       _,                 _,        _ = create_teacher_student()

    # Move to default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:    inference_func = t_model.detect_objects_cuda
    except: inference_func = t_model.module.detect_objects_cuda

    s_model = s_model.to(device)
    s_model = nn.DataParallel(s_model)

    t_model = t_model.to(device)
    t_model = nn.DataParallel(t_model)
    t_model.eval()

    original_GT_example_path = f"{config.args.path.DB_ROOT}{config.args.dataset_type}_gt_samples_{config.args.MP}/"
    Edited_GT_example_path = f"{config.args.path.DB_ROOT}{config.args.dataset_type}_gt_samples_{config.args.MP}_Edit/"
    print(f'\n\n\nCopied: {original_GT_example_path} to {Edited_GT_example_path}\n\n\n')
    print(f"This process will take a few minutes...")
    start_time = time.time()
    if os.path.exists(Edited_GT_example_path):
        print(f"Edited_GT_example_path exists. Removing it...")
        shutil.rmtree(Edited_GT_example_path)
        print(f"Done!")
    shutil.copytree(original_GT_example_path, Edited_GT_example_path)
    print(f"Time: {time.time() - start_time}")

    criterion = MultiBoxLoss(priors_cxcy=s_model.module.priors_cxcy).to(device)

    train_dataset = KAISTPed(args, condition='train')
    train_loader = DataLoader(train_dataset, batch_size=train_conf.batch_size, shuffle=True,
                              num_workers=config.dataset.workers,
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True)  # note that we're passing the collate function here

    test_dataset = KAISTPed(args, condition='test')
    test_batch_size = args["test"].eval_batch_size * torch.cuda.device_count()
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                             num_workers=config.dataset.workers,
                             collate_fn=test_dataset.collate_fn,
                             pin_memory=True)
    
    # Set job directory
    if args.exp_time is None:
        args.exp_time = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    
    exp_name = ('_' + args.exp_name) if args.exp_name else '_'
    jobs_dir = os.path.join('jobs', args.exp_time + exp_name)
    os.makedirs(jobs_dir, exist_ok=True)
    args.jobs_dir = jobs_dir

    # Make logger
    logger = utils.make_logger(args)

    # Epochs
    kwargs = {'grad_clip': args['train'].grad_clip, 'print_freq': args['train'].print_freq}
    for epoch in range(s_epochs, epochs):
        # One epoch's training
        logger.info('#' * 20 + f' << Epoch {epoch:3d} >> ' + '#' * 20)
        train_loss = train_epoch(s_model=s_model,
                                 t_model=t_model,
                                 dataloader=train_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 logger=logger,
                                 epoch=epoch,
                                 inference_func=inference_func,
                                 **kwargs)

        optim_scheduler.step()

        # Save checkpoint
        utils.save_checkpoint(epoch, s_model.module, optimizer, train_loss, jobs_dir, "student")
        utils.save_checkpoint(epoch, t_model.module,      None,       None, jobs_dir, "teacher")
        
        if epoch >= 0:
            result_filename = os.path.join(jobs_dir, f'Epoch{epoch:03d}_test_det.txt')
            results = val_epoch(s_model, test_loader, config.test.input_size, inference_func, min_score=0.1)

            save_results(results, result_filename)
            
            evaluate(config.PATH.JSON_GT_FILE, result_filename, phase)
            
if __name__ == '__main__':
    main()
