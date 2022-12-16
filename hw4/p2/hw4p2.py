import os
import gc
from glob import glob

import numpy as np
import torch
import torchaudio
import pandas as pd

from tqdm import tqdm

import wandb
os.environ['WANDB_SILENT'] = 'True'
wandb.login(key='5ac46cad74707129efc6a4fec92ceb503f6e882f')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from params import *
from utils import *
from data import *
from model import *


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE)


def train(dataloader, model, criterion, optimizer, scaler, tf_rate):

    model.train()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, desc='train')

    running_loss = 0.0
    running_perplexity = 0.0

    for i, (x, lx, y, ly) in enumerate(dataloader):

        gc.collect()
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.cuda.amp.autocast():

            predictions, attention_plot = model(x, lx, y, tf_rate)
            loss = criterion(predictions.flatten(end_dim=1), y.flatten()).flatten()

            _, transcript_max_seq_len = y.shape

            range_tensor = torch.arange(transcript_max_seq_len).unsqueeze(0)
            ly = ly.unsqueeze(1)

            padding_mask = (range_tensor < ly).flatten().to(DEVICE)
            masked_loss = torch.sum(loss * padding_mask) / torch.sum(padding_mask)
            perplexity = torch.exp(masked_loss)

            running_loss += masked_loss.item()
            running_perplexity += perplexity.item()

        scaler.scale(masked_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_bar.set_postfix(loss=running_loss/(i+1),
                              perplexity=running_perplexity/(i+1),
                              lr=float(optimizer.param_groups[0]['lr']),
                              tf_rate=tf_rate)
        batch_bar.update()

        del x, y, lx, ly

    running_loss /= len(dataloader)
    running_perplexity /= len(dataloader)

    batch_bar.close()

    return running_loss, running_perplexity, attention_plot


def validate(dataloader, model, criterion):

    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='val')

    running_lev_dist = 0.0

    for i, (x, lx, y, ly) in enumerate(dataloader):

        gc.collect()
        torch.cuda.empty_cache()

        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.inference_mode():
            predictions, _ = model(x, lx)

        greedy_predictions = torch.argmax(predictions, dim=2)
        running_lev_dist += calc_edit_distance(greedy_predictions, y, ly, print_example=True)

        batch_bar.set_postfix(dist=running_lev_dist/(i+1))
        batch_bar.update()

        del x, lx, y, ly

    batch_bar.close()

    running_lev_dist /= len(dataloader)

    # return running_lev_dist, running_loss, running_perplexity
    return running_lev_dist


def experiments(train_dataloader, val_loader, model, criterion, optimizer, lr_scheduler, scaler):

    best_lev_dist = float('inf')
    tf_rate = 0.1

    for epoch in range(0, CONFIG['epochs']):

        print('\nepoch:', epoch+1, '/', CONFIG['epochs'])

        (running_loss, running_perplexity,
         attention_plot) = train(train_dataloader, model, criterion, optimizer, scaler, tf_rate)

        valid_dist = validate(val_loader, model, criterion)

        plot_attention(epoch+1, attention_plot)

        wandb.log({'running_loss': running_loss, 'running_perplexity': running_perplexity,
                   'valid_dist': valid_dist, 'learning_rate': optimizer.param_groups[0]['lr'],
                   'tf_rate': tf_rate})

        lr_scheduler.step()
        # tf_scheduler.step()

        if valid_dist <= best_lev_dist:
            best_lev_dist = valid_dist
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': valid_dist,
                        'epoch': epoch}, 'checkpoint_'+ATTEMPT_ID+'.pth')

            # wandb model save
            wandb.save('checkpoint_'+ATTEMPT_ID+'.pth')


def testing(test_dataloader, model):

    model.eval()
    batch_bar = tqdm(total=len(test_dataloader), dynamic_ncols=True, leave=False, position=0, desc='test') 

    predictions = []

    for i, (x, lx) in enumerate(test_dataloader):

        gc.collect()
        torch.cuda.empty_cache()

        x = x.to(DEVICE)

        # forward prop
        with torch.inference_mode():
            pred, _ = model(x, lx)

        greedy_predictions = torch.argmax(pred, dim=2)
        batch_preds = make_output(greedy_predictions)

        predictions += batch_preds

        batch_bar.update()

        del x, lx

    return predictions


def main():

    train_loader, val_loader, test_loader = get_loaders()

    model = get_model()
    model = model.to(DEVICE)

    sanity(train_loader, model)

    # load model with best validation accuracy
    checkpoint = torch.load('checkpoint_3.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['cosann_start_lr'], amsgrad=True, weight_decay=5e-6)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=CONFIG['epochs'],
                                                              eta_min=CONFIG['cosann_min_lr'])

    run = wandb.init(name='attempt_'+ATTEMPT_ID, reinit=True, project='hw4p2', config=CONFIG)

    experiments(train_loader, val_loader, model, criterion, optimizer, lr_scheduler, scaler)
    predictions = testing(test_loader, model)

    run.finish()

    df = pd.read_csv(SUBMISSION_FILE)
    df.label = predictions

    df.to_csv('submission.csv', index=False)
    os.system('kaggle competitions submit -c 11-785-f22-hw4p2 -f submission.csv -m "first"')


if __name__ == '__main__':

    main()
