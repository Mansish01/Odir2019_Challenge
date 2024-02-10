from torch.utils.data import Dataset
import torch.nn as nn 
import torch
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt  
from uuid import uuid4
import os 
from datetime import datetime 
from torch.utils.tensorboard import SummaryWriter
from src.dataloader import train_dataloader , val_dataloader
from models.models import Dense161Model
#3. Training 

if  __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    
    device = torch.device('cuda')
    dt= datetime.now()
    
    format_dt = dt.strftime("%Y-%m-%d-%H-%M-%S")
    #create folder
    folder_name = f"run-{format_dt}"
    os.mkdir(f"artifacts/{folder_name}")
    
    
    writer = SummaryWriter(log_dir = f"artifacts/{folder_name}/tensorboard_logs")
    
    
    #load the model
    model= Dense161Model(num_labels=8).to(device)
    

    
    
    #3. Train
    
    LR= 0.001
    EPOCHS = 20
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    epochwise_train_losses = []
    epochwise_val_losses = []
    epochwise_val_acc= []
    epochwise_train_acc= []

    best_val_accuracy = 0
    limit = 0.01
    avg_acc = 0
    avg_acc_next= 0
    MIN_EPOCHS = 6
    for epoch in range(EPOCHS):
        train_running_loss  = 0
        val_running_loss = 0
        train_running_accuracy = 0
        val_running_accuracy = 0
        
        
        model.train()   #change into training mode
        for images , labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            model_out = model(images)
            model_out = F.log_softmax(model_out , dim =1)
            loss = criterion(model_out, labels)
            train_running_loss += loss.item()* images.size(0)
            loss.backward()
            optimizer.step()
            # print("loss",loss)
            # values, indices = torch.max(model_out, dim =1)
        
        # for images , labels in train_dataloader:
        #     images, labels = images.to(device), labels.to(device)
        #     model_out = model(images)
        #     model_out = F.log_softmax(model_out , dim =1)
        #     loss = criterion(model_out, labels)
        #     train_running_loss += loss.item()* images.size(0)
           
            #find acuracy           
            preds = torch.argmax(model_out, dim=1)
            acc = (preds== labels).float().mean()
            train_running_accuracy += acc.item() 
              
        model.eval()  #change into validation mode
        for images , labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            model_out = model(images)
           
            model_out = F.log_softmax(model_out , dim =1)
            loss = criterion(model_out, labels)
            val_running_loss += loss.item()* images.size(0)
            
            
            #find acuracy           
            preds = torch.argmax(model_out, dim=1)
            acc = (preds== labels).float().mean()
            val_running_accuracy += acc.item()
            
            
              
        
            
        
        avg_train_loss = train_running_loss/ len(train_dataloader)
        avg_val_loss = val_running_loss/ len(val_dataloader)
        
        avg_train_acc = train_running_accuracy / len(train_dataloader)
        avg_val_acc = val_running_accuracy / len(val_dataloader) #data loader length gives the number of total batches

        #log to tensorboard
        writer.add_scalar("loss/train" , avg_train_loss, epoch)
        writer.add_scalar("loss/val" , avg_val_loss, epoch)
        writer.add_scalar("acc/train" , avg_train_acc, epoch)
        writer.add_scalar("acc/val" , avg_val_acc, epoch)
        
        if avg_val_acc > best_val_accuracy:
             best_val_accuracy = avg_val_acc
             epoch_no = epoch
             
             checkpoint_name_best = f"artifacts/best_model.pth"
             checkpoint_best = {
                "epoch" : epoch, 
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(), 
                "train_loss": avg_train_loss, 
                "val_loss": avg_val_loss, 
                "train_acc": avg_train_acc, 
                "val_acc" : avg_val_acc}
            
        torch.save(checkpoint_best ,  checkpoint_name_best)
             
             
        
       
        epochwise_train_losses.append(avg_train_loss)
        epochwise_val_losses.append(avg_val_loss)
        epochwise_val_acc.append(avg_val_acc)
        epochwise_train_acc.append(avg_train_acc)
        
        # if epoch >= MIN_EPOCHS:
        #     avg_acc = np.mean(epochwise_val_acc[-5:])
        #     avg_acc_next = np.mean(epochwise_val_acc[-6:-1])

        #     difference = avg_acc - avg_acc_next
        #     print(f"Difference: {difference:.3f}")
            
        #     if( difference <= limit):
        #         print("the last epoch was", epoch)
        #         break
        
        
       
        print(f"epoch {epoch} train loss: {avg_train_loss:.3f} \t val loss : {avg_val_loss:.3f} \t train accuracy : {avg_train_acc:.3f} \t val accuracy : {avg_val_acc:.3f}")

        # checkpoint_name = f"artifacts/{folder_name}/ckpt-{model.__class__.__name__}-val={avg_val_acc:.3f}-epoch={epoch}"
        # checkpoint = {
        #     "epoch" : epoch, 
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict" : optimizer.state_dict(), 
        #     "train_loss": avg_train_loss, 
        #     "val_loss": avg_val_loss, 
        #     "train_acc": avg_train_acc, 
        #     "val_acc" : avg_val_acc
            
        # }
        
        # torch.save(checkpoint ,  checkpoint_name)
    
    print(f" the best val acc is {best_val_accuracy} from epoch {epoch_no}")
                
    #save the model

    # torch.save(model, "artifacts/first_model.pth")

    fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    x_axis_values = np.arange(0, EPOCHS, 1)

    ax1.plot(epochwise_train_losses , label= "train loss")
    ax1.plot(epochwise_val_losses , label="val loss")
    ax1.set_title("train vs validation loss")
    ax1.legend()

    ax2.plot(epochwise_train_acc , label= "train accuracy")
    ax2.plot(epochwise_val_acc , label="val loss")
    ax2.set_title("train vs validation accuracy")
    ax2.legend()
    plt.show()


    # print(epochwise_train_losses)
    # print(epochwise_val_loss)
        
        
        
        
        
        