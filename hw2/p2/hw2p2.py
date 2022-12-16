import os
import gc
import glob

import numpy as np
import pandas as pd
import torch
import torchvision
from torchsummary import summary

import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from params import *
from data import *
from model import *


os.environ['WANDB_SILENT'] = 'True'
wandb.login(key='5ac46cad74707129efc6a4fec92ceb503f6e882f')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_classification(dataloader, model, optimizer, criterion, scaler) :
    
    # train mode
    model.train()

    # tqdm progress bar 
    batch_bar = tqdm(total=len(dataloader),
                     dynamic_ncols=True, leave=True,
                     position=0, desc='train', ncols=50) 
    
    # initialize metrics
    num_correct = 0
    total_loss = 0

    for (images, labels) in dataloader:
        
        # initialize optimier gradients to zero
        optimizer.zero_grad()

        # batch images and labels to DEVICE
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # forward prop
        with torch.cuda.amp.autocast() :
            outputs = model(images)
            loss = criterion(outputs, labels)

        # metrics update
        num_correct += int((torch.argmax(outputs, axis=1)==labels).sum())
        total_loss  += float(loss.item())

        # tqdm update
        batch_bar.set_postfix(lr='{:.04f}'.format(float(optimizer.param_groups[0]['lr'])),
                              num_correct=num_correct)
        batch_bar.update()
        
        # back prop with mixed precision
        scaler.scale(loss).backward()

        # scaler update
        scaler.step(optimizer)
        scaler.update()

    # close tqdm bar
    batch_bar.close()

    # update metrics
    acc = 100 * num_correct / (CONFIG['batch_size'] * len(dataloader))
    total_loss = float(total_loss / len(dataloader))

    return acc, total_loss

def validate_classification(dataloader, model, criterion):

    # eval mode
    model.eval()

    # tqdm progress bar
    batch_bar = tqdm(total=len(dataloader),
                     dynamic_ncols=True, leave=True,
                     position=0, desc='val  ', ncols=3)

    # initialize metrics
    num_correct = 0.0
    total_loss = 0.0

    for (images, labels) in dataloader:
        
        # batch images and labels to DEVICE
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # forward prop
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # metrics update
        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss  += float(loss.item())

        # tqdm update
        batch_bar.set_postfix(num_correct=num_correct)
        batch_bar.update()
    
    # close tqdm bar
    batch_bar.close()

    # update metrics
    acc = 100 * num_correct / (CONFIG['batch_size'] * len(dataloader))
    total_loss = float(total_loss / len(dataloader))

    return acc, total_loss

def experimentation_classification(train_loader, val_loader, model, optimizer, criterion, scheduler, scaler):
    
    # initialize metrics
    best_train_loss = 0.0
    best_train_acc  = 0.0
    best_val_loss   = 0.0
    best_val_acc    = 0.0

    for epoch in range(CONFIG['epochs']):
        
        # epoch start
        print('\nepoch:         ', epoch+1, '/', CONFIG['epochs'])
        print('learning rate: ', optimizer.param_groups[0]['lr'])

        # train for one epoch
        train_acc, train_loss = train_classification(dataloader=train_loader, model=model,
                                                     optimizer=optimizer, criterion=criterion,
                                                     scaler=scaler)
        
        # validation for one epoch
        val_acc, val_loss = validate_classification(dataloader=val_loader, model=model,
                                                    criterion=criterion)

        # print train results for one epoch
        print('train loss:    ', train_loss)
        print('train acc:     ', train_acc)
        print('val loss:      ', val_loss)
        print('val acc:       ', val_acc)

        # wandb update
        wandb.log({'train_loss': train_loss, 'train_accuracy': train_acc,
                   'validation_accuracy': val_acc, 'validation_loss': val_loss,
                   'learning_Rate': optimizer.param_groups[0]['lr']})
        
        # save if val accuracy is best
        if val_acc >= best_val_acc:
            # results for best validation accuracy
            best_train_loss = train_loss
            best_train_acc  = train_acc
            best_val_loss   = val_loss
            best_val_acc    = val_acc
            
            # torch model save
            print('saving model')
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_acc': val_acc, 
                        'epoch': epoch}, 'checkpoint.pth')
            
            # wandb model save
            wandb.save('checkpoint.pth')
        
        # update learning rate
        scheduler.step()

    return model, best_train_loss, best_train_acc, best_val_loss, best_val_acc

def test_classification(dataloader, model):

    # eval mode
    model.eval()

    # tqdm progress bar
    batch_bar = tqdm(total=len(dataloader),
                     dynamic_ncols=True, position=0,
                     leave=True, desc='test ', ncols=3)
    
    # log test predictions
    test_results = []
    
    for (images) in dataloader:

        # batch images to DEVICE
        images = images.to(DEVICE)

        # forward prop
        with torch.inference_mode():
            outputs = model(images)

        # final output is max of predictions
        outputs = torch.argmax(outputs, axis=1).detach().cpu().numpy().tolist()
        test_results.extend(outputs)
        
        # tqdm update
        batch_bar.update()
    
    # close tqdm bar
    batch_bar.close()
    
    return test_results


def validate_verification(unknown_images, known_paths, known_images, model, similarity, mode='val'): 

    unknown_feats, known_feats = [], []

    batch_bar = tqdm(total=len(unknown_images) // CONFIG['batch_size'], dynamic_ncols=True, position=0, leave=False, desc=mode)
    model.eval()

    for i in range(0, unknown_images.shape[0], CONFIG['batch_size']):

        unknown_batch = unknown_images[i:i+CONFIG['batch_size']]
        
        with torch.no_grad():
            unknown_feat = model(unknown_batch.float().to(DEVICE), return_res_out=True)         

        unknown_feats.append(unknown_feat)
        batch_bar.update()
    
    batch_bar.close()
    
    batch_bar = tqdm(total=len(known_images) // CONFIG['batch_size'], dynamic_ncols=True, position=0, leave=False, desc=mode)
    
    for i in range(0, known_images.shape[0], CONFIG['batch_size']):

        known_batch = known_images[i:i+CONFIG['batch_size']] 

        with torch.no_grad():
              known_feat = model(known_batch.float().to(DEVICE), return_res_out=True)
          
        known_feats.append(known_feat)
        batch_bar.update()

    batch_bar.close()

    # concatenate all the batches
    unknown_feats = torch.cat(unknown_feats, dim=0)
    known_feats = torch.cat(known_feats, dim=0)

    similarity_values = torch.stack([similarity(unknown_feats, known_feature) for known_feature in known_feats])
    predictions = similarity_values.argmax(0).cpu().numpy()

    # map argmax indices to identity strings
    pred_id_strings = [known_paths[i] for i in predictions]
    
    if mode == 'val':

        true_ids = pd.read_csv('/content/data/verification/dev_identities.csv')['label'].tolist()
        accuracy = accuracy_score(pred_id_strings, true_ids)
        
        print('verification accuracy = {}'.format(accuracy))
    
    return pred_id_strings


def main():

    ### FACE CLASSIFICATION ###

    train_dataset, val_dataset, test_dataset = get_datasets()
    
    train_loader, val_loader, test_loader = get_loaders(train_dataset,
                                                        val_dataset,
                                                        test_dataset)

    model = get_ConvNext(in_channels=3, num_classes=len(train_dataset.classes)).to(DEVICE)
    summary(model, (3, 224, 224))

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=CONFIG['cross_entropy_label_smoothing'])
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=CONFIG['cosann_start_lr'],
                                momentum=CONFIG['sgd_momentum'],
                                weight_decay=CONFIG['sgd_weight_decay'])

    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=CONFIG['epochs'],
                                                           eta_min=CONFIG['cosann_min_lr'])

    run = wandb.init(name='cosann_convnext_final', reinit=True, project='hw2p2', CONFIG=CONFIG)

    model, train_loss, train_acc, val_loss, val_acc = experimentation_classification(train_loader=train_loader,
                                                                                     val_loader=val_loader,
                                                                                     model=model,
                                                                                     optimizer=optimizer,
                                                                                     criterion=criterion,
                                                                                     scheduler=scheduler,
                                                                                     scaler=scaler)
    
    print('trained model results')
    print('final train loss: ', train_loss)
    print('final train accuracy: ', train_acc)
    print('final validation loss: ', val_loss)
    print('final validation accuracy: ', val_acc)

    run.finish()

    # load model with best validation accuracy
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = test_classification(dataloader=test_loader, model=model)

    with open('classification_submission.csv', 'w+') as f:
        f.write('id,label\n')
        for i in range(len(test_dataset)):
            f.write("{},{}\n".format(str(i).zfill(6) + '.jpg', test_results[i]))

    # submit classification results to kaggle
    os.system('kaggle competitions submit -c 11-785-f22-hw2p2-classification -f classification_submission.csv -m "Test Submission"')


    ### FACE VERIFICATION ###

    known_paths = [i.split('/')[-2] for i in sorted(glob.glob(KNOWN_REGEX))] 
    known_images = [Image.open(p) for p in tqdm(sorted(glob.glob(KNOWN_REGEX)))]
    known_images  = torch.stack([FACE_VERIFICATION_TRANSFORMS(y) for y in known_images])

    unknown_test_images = [Image.open(p) for p in tqdm(sorted(glob.glob(UNKNOWN_TEST_REGEX)))]
    unknown_images = torch.stack([FACE_VERIFICATION_TRANSFORMS(x) for x in unknown_test_images])

    # face embeddings similarity metric
    similarity_metric = torch.nn.CosineSimilarity(dim= 1, eps= 1e-6)
    pred_id_strings = validate_verification(unknown_images, known_paths, known_images, model, similarity_metric, mode='test')

    with open('verification_submission.csv', 'w+') as f:
        f.write("id,label\n")
        for i in range(len(pred_id_strings)):
            f.write("{},{}\n".format(i, pred_id_strings[i]))
    
    # submit verification results to kaggle
    os.system('kaggle competitions submit -c 11-785-f22-hw2p2-verification -f verification_submission.csv -m "Test Submission"')


if __name__ == '__main__':

    main()
