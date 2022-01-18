import torch                  
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.optim as optim
from os.path import join, isfile, exists
import os
import logging
import argparse
import csv
import datetime
import numpy as np
from PIL import Image
from progress.bar import Bar
import sklearn.metrics
from loadDataset_sil_pose import Dataset as loadDataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

_LOGGER = logging.getLogger(__name__)


def main(data_dir, learning_rate, batch_size, epochs, mode, name):
    """
    Train and test CNN for classifying view angles from 3D poses and silhouettes.
    param: data_dir - input data directory
    param: learning_rate - learning rate for CNN
    param: batch_size - size of batch to use
    param: epochs - number of epochs to train for
    param: mode - train, test or resume.
    param: name - name of the run.
    """
    _LOGGER.info("Learning Rate: %.10f", learning_rate)
    _LOGGER.info("Batch Size: %d", batch_size)
    _LOGGER.info("Epochs: %d", epochs)

    train_data = join(data_dir, "train/")

    val_data = join(data_dir, "val/")
	
    test_data = join(data_dir, "test/")

    model_dir = join("models", name)
    
    if not exists(model_dir):
        os.makedirs(model_dir)

    #CUDA timer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
	
    use_gpu = torch.cuda.is_available()
	
    if(use_gpu):
        _LOGGER.info("Using GPU.") 
    else:
        _LOGGER.info("Using CPU.") 

    model = Net().cuda()
	
	
    #mean, std = [0,0]
    #mean,std = [0.14080575],[0.3444886]#100k dataset
    mean,std = [0.14054182],[0.3442364]#128k dataset

    writer = SummaryWriter('runs/' + name)

    if mode == 'train':
		
        print ("Mean :", mean)
        print ("Std :", std)
        train_dataloader = load_data(train_data,batch_size, mean, std)
		
        val_dataloader = load_data(val_data, batch_size, mean, std)

        _LOGGER.info("Train batches: %d",  len(train_dataloader))
    #	_LOGGER.info("Gallery train batches: %d", len(train_gal_dataloader))

		
        _LOGGER.info("Val batches: %d",  len(val_dataloader))
	#	_LOGGER.info("Gallery val batches: %d", len(val_gal_dataloader))
		

        _LOGGER.info("Starting Training... ")
	
        #SGD optimiser, momentum=0.9, weight_decay=0.0005
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, 
                weight_decay=0.0005)
		
		
        #Initialise early stopping
        early_stopping = EarlyStopping(5)
        total_time_ms = 0.0
		

        for epoch in range(epochs):
		
            start.record()
					
            _LOGGER.info("Training Epoch: %d", epoch+1)
			
            _LOGGER.info("Learning Rate: %.10f \n", learning_rate )
			
			
            #Update learning rate
            if epoch+1 > 1 and epoch+1  % 5000 == 0:
                print ("Updating Learning Rate")
                learning_rate = learning_rate/10
                print ("New learning rate = ", learning_rate)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
			
			
            #TRAIN
            t_loss, t_acc = train(model, optimizer, train_dataloader, epoch+1, batch_size, use_gpu)#, writer)		
            
            # Write train loss to tensorboard
            writer.add_scalar('training loss',
                            np.mean(t_loss),
                            epoch)

            # Write train accuracy to tensorboard
            writer.add_scalar('training acc',
                    np.mean(t_acc),
                    epoch)


            # Save every 5th epoch
            if (epoch+1) > 0 and (epoch+1) % 5 == 0:
				
                torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, model_dir + "/model_"+ str(epoch+1)+ ".pth")

                _LOGGER.info("Saving Model.")


            _LOGGER.info("Performing Validation... ")
			
            #VALIDATION TESTING
            v_loss, v_acc = validate(model, val_dataloader, batch_size, epoch+1, use_gpu,early_stopping, optimizer)
	        #Write validation loss to tensorboard
            writer.add_scalar('val loss',
                            np.mean(v_loss),
                            epoch)

            #Write validation accuracy to tensorboard
            writer.add_scalar('val acc',
                    np.mean(v_acc),
                    epoch)


            if early_stopping.early_stop:
                print("Early stopping.")
                break

            
            # Calc time taken
            end.record()
            torch.cuda.synchronize()
            time_ms = start.elapsed_time(end)
            epoch_time = datetime.timedelta(milliseconds=time_ms)
            _LOGGER.info("Epoch Elapsed Time: %s" % str(epoch_time))

            total_time_ms = total_time_ms + time_ms
            total_time = datetime.timedelta(milliseconds=total_time_ms)
            _LOGGER.info("Total Elapsed Time: %s" % str(total_time))
			
            # Calc time remaining
            epoch_remain = (epochs - (epoch + 1))
            _LOGGER.info("Epochs remaining: %d " % epoch_remain)
            est_time_remain_ms = epoch_remain * (total_time_ms/(epoch+1))
            est_time_remain = datetime.timedelta(milliseconds=est_time_remain_ms)
            _LOGGER.info("Estimated Time Remaining: %s" % str(est_time_remain) + "\n")

        _LOGGER.info("Training Complete.")
		
        # Save final model
        torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, "model.pth")

        _LOGGER.info("Model Saved.")

		
        _LOGGER.info("Total Elapsed Time: %s" % str(total_time))
		

    elif mode == 'test':
		
        _LOGGER.info("Starting Testing... ") 
        model = Net().cuda()
        model.load_state_dict(torch.load("model.pth")['state_dict'])
        test_dataloader = load_data(test_data, batch_size, mean, std)
				
        
        #writer = SummaryWriter("runs/" + name)

        #get some random images
        dataiter = iter(test_dataloader)
        images,labels = dataiter.next()

        #create grid of images
        img_grid = torchvision.utils.make_grid(images)

        #show images
        matplotlib_imshow(img_grid, one_channel=True)
        
        writer.add_image('test_images', img_grid)
        
        #show net graph
        net = Net()
        writer.add_graph(net, images)
        writer.close()

        # TEST
        test(model, test_dataloader, batch_size, use_gpu)
		
    elif mode == 'resume':
		
        model = Net().cuda()
        model_data = torch.load('model.pth')
        model.load_state_dict(model_data['state_dict'])
		
        print ("Mean :", mean)
        print ("Std :", std)
		
        train_dataloader = load_data(train_data,batch_size, mean, std)
		
        val_dataloader = load_data(val_data, batch_size, mean, std)

        _LOGGER.info("Train batches: %d",  len(train_dataloader))
			
        _LOGGER.info("Val batches: %d",  len(val_dataloader))

        start_epoch = model_data['epoch']

        _LOGGER.info("Resuming Training from epoch %d" % start_epoch)
	
        #SGD optimizer 
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                weight_decay=0.0005)
        
        optimizer.load_state_dict(model_data['optimizer'])
	
        early_stopping = EarlyStopping(5)
		
        total_time_ms = 0.0
        
        es_epoch = start_epoch + epochs

        for epoch in range(start_epoch, start_epoch + epochs):
			
            start.record()
			
            _LOGGER.info("Training Epoch: %d", epoch+1)
			
            _LOGGER.info("Learning Rate: %.10f", learning_rate)
			
			
            #Update learning rate
            if epoch > 1 and epoch % 5000 == 0:
                learning_rate = learning_rate/10
                for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate
			
            #TRANING
            t_loss, t_acc = train(model, optimizer, train_dataloader, epoch, batch_size, use_gpu)

            #Write train loss to tensorboard
            writer.add_scalar('training loss',
                            np.mean(t_loss),
                            epoch)
            
            #Write train accuracy to tensorboard
            writer.add_scalar('training acc',
                    np.mean(t_acc),
                    epoch)

            #Save every 5th epoch
            if (epoch+1) % 5 == 0:
				
                torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, model_dir + "model_"+ str(epoch+1)+ ".pth")
            _LOGGER.info("Performing Validation... ")

            #VALIDATION TESTING
            v_loss, v_acc = validate(model, val_dataloader, batch_size, epoch+1, use_gpu,early_stopping, optimizer)
            #Write validation loss to tensorboard
            writer.add_scalar('val loss',
                            np.mean(v_loss),
                            epoch)

            #Write validation accuracy to tensorboard
            writer.add_scalar('val acc',
                    np.mean(v_acc),
                    epoch)



            if early_stopping.early_stop:
                print("Early stopping.")
                es_epoch = (epoch+1)
                break
		
            #Calc time taken
            end.record()
            torch.cuda.synchronize()
            time_ms = start.elapsed_time(end)
            epoch_time = datetime.timedelta(milliseconds=time_ms)
            _LOGGER.info("Epoch Elapsed Time: %s" % str(epoch_time))

            total_time_ms = total_time_ms + time_ms
            total_time = datetime.timedelta(milliseconds=total_time_ms)
            _LOGGER.info("Total Elapsed Time: %s" % str(total_time))
			
            #Calc time remaining
            epoch_remain = (epochs - (epoch + 1 - start_epoch))
            _LOGGER.info("Epochs remaining: %d " % epoch_remain)
            est_time_remain_ms = epoch_remain * (total_time_ms/(epoch+1-start_epoch))
            est_time_remain = datetime.timedelta(milliseconds=est_time_remain_ms)
            _LOGGER.info("Estimated Time Remaining: %s" % str(est_time_remain) + "\n")

        _LOGGER.info("Training Complete.")
			
        #Save final model
        torch.save({
        'epoch': es_epoch,
        'state_dict': model.state_dict(),
       'optimizer': optimizer.state_dict(),
        }, "model.pth")

        _LOGGER.info("Model Saved.")

        _LOGGER.info("Total Elapsed Time: %s" % str(total_time))



def load_data(data_dir, batch_size, mean, std):
    """
    Load the probe and gallery data and perform data transforms on data.
    Return: pro_dataloader, gal_dataloader, pro_sizes, gal_sizes 
    """
	
    #Data transformations
    data_transforms = transforms.Compose([
			
        #transforms.ToPILImage(),	
			
        transforms.Resize((126,126)),
        transforms.Grayscale(1),
        transforms.ToTensor(),		
        transforms.Normalize(mean,std)
        #transforms.Normalize((mean[0],mean[1]),(std[0],std[1])),
			
        ]) 
	
    #PROBE
    dataset = loadDataset(data_dir, data_transforms)
#   test_data = datasets.ImageFolder(data_dir)
#   print ("DATA TYPE: " , type(test_data))
    dataloader = torch.utils.data.DataLoader(dataset,
    batch_size=batch_size,shuffle=True, num_workers=4, drop_last=True) 

	
    return dataloader


def calc_acc(y_pred, y, total):
    """
    Calculate the accuracy of a prediction.
    return:  acc, n_correct
    """
    
    n_correct = 0

    pred = y_pred.argmax(dim=1)
    #print ("PRED: ", pred) 

    n_correct += (pred == y.view_as(pred)).sum().item()

    acc = 100 * n_correct / total
    
    return acc, n_correct


def train(model, optimizer, dataloader, epoch, batch_size, use_gpu,):
    """
    Perform the training of the network.
    Return: loss_vals, acc_vals.
    """
		
    model.train()
	
    loss_values = []	
    acc_values = []
    #loss function.

    loss_fn = torch.nn.CrossEntropyLoss()
    #loss_fn = torch.nn.BCEWithLogitsLoss()
	
    with Bar('  Training', max=len(dataloader), suffix ='%(index)d/%(max)d - %(eta)ds\r') as bar:

        for i,data in enumerate(dataloader):
		
            iteration = i + 1
		
            if epoch > 0:
			
                iteration = (i + (epoch*len(dataloader)+1))
		
            x,y = data        
            #x, x1, y = data

            y = y.view(x.size(0), -1)

            if use_gpu:
                x = Variable(x.cuda())
                #	x1 = Variable(x1.cuda())
                y = Variable(y.cuda())

            # Forward pass.
            y_pred = model(x)#,x1)
            #print (torch.max(torch.sigmoid(y_pred), 1)[0].view(y.size()))
            loss = loss_fn(y_pred, torch.max(y.long(), 1)[0])
            #loss = loss_fn(y_pred, y.float())
            #print (y_pred)
            loss_values.append(loss.item())
            #_LOGGER.info("Iteration: %d, Loss: %.20f", iteration, loss.item())
		
		
            # Zero the gradients before running the backward pass.
            optimizer.zero_grad()

            # Backward pass.
            loss.backward()
		
            optimizer.step()
            train_acc, correct = calc_acc(y_pred,y,batch_size)	
            acc_values.append(train_acc)
				
			
            bar.next()
	    
	
            if i+1 == len(dataloader):

                print ("\n")
                _LOGGER.info("Train Loss: %.5f" % np.mean(loss_values))
                _LOGGER.info("Train Accuracy: %.1f %%" % np.mean(acc_values))
			

    bar.finish()		

    return loss_values, acc_values

def validate(model, dataloader, batch_size, epoch,use_gpu, early_stopping, optimizer):
    """
    Perform validation tests on the model.
    Return: loss_vals, acc_vals
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    #loss_fn = torch.nn.BCEWithLogitsLoss()

    #torch.no_grad()
    model.eval()
    loss_values = []
    acc_values = []
    val_loss = 0


    with Bar('  Validation Testing:', max=len(dataloader), suffix ='%(index)d/%(max)d - %(eta)ds\r') as bar, torch.no_grad():
        for i,data in enumerate(dataloader):
		
            iteration = i + 1

            if epoch > 1:
			
                iteration = (i + (epoch*len(dataloader)+1))
		
            #print ("Validation Iteration %d" % iteration)
        
            x, y = data
            #x, x1, y = data

            y = y.view(x.size(0), -1)
	
            if use_gpu:
                x = Variable(x.cuda())
                #	x1 = Variable(x1.cuda())
                y = Variable(y.cuda())

                # Forward pass.
            y_pred = model(x)#,x1)
		
            val_loss = loss_fn(y_pred, torch.max(y.long(), 1)[0])
            #val_loss = loss_fn(y_pred, y.float())
            loss_values.append(val_loss.item())
            val_acc, correct = calc_acc(torch.softmax(y_pred,dim=1),y,batch_size)
            #val_acc, correct = calc_acc(torch.sigmoid(y_pred),y,batch_size)
            acc_values.append(val_acc)
			
            bar.next()
            

            if i + 1 == len(dataloader):

                print ("\n")
                _LOGGER.info("Validation Loss: %.5f" % np.mean(loss_values))
                _LOGGER.info("Validation Accuracy: %.1f %%" % np.mean(acc_values) + "\n")
	        
                early_stopping(np.mean(loss_values), model, optimizer, epoch)
	

    bar.finish()

    return loss_values,acc_values


def test(model, dataloader, batch_size, use_gpu):
    """
    Perform testing on the model.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    #loss_fn = torch.nn.BCEWithLogitsLoss()

    #torch.no_grad()
    model.eval()
    acc_values = []
    n_correct, test_loss = 0, 0
    #preds = []
    #labels = []
    topk_acc = []
    
    confusion_matrix = torch.zeros(11,11)


    with Bar('  Final Testing:', max=len(dataloader), suffix ='%(index)d/%(max)d - %(eta)ds\r') as bar, torch.no_grad():
        for i, data in enumerate(dataloader):
		
            iteration = i + 1
			
            x,y = data
            #	x, x1, y = data


            y = y.view(x.size(0), -1)

            if use_gpu:
                x = Variable(x.cuda())
                #	x1 = Variable(x1.cuda())
                y = Variable(y.cuda())


            # Forward pass.
            y_pred = model(x)#,x1)
            #preds = preds + y_pred.tolist()
            #labels = labels + y.tolist()
            test_loss = loss_fn(y_pred, torch.max(y.long(), 1)[0])
            #test_loss = loss_fn(y_pred, y.float())
            test_acc, correct = calc_acc(torch.softmax(y_pred,dim=1),y,batch_size)
            acc_values.append(test_acc)
	    
            _,preds = torch.max(y_pred,1)
            for t, p in zip(y.view(-1),preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1


            bar.next()

            #print ("Top 1: ", topk[0].item())
			
            if i + 1 == len(dataloader):
                print ("\n")
                print ("Test Accuracy: %.1f %%" % np.mean(acc_values))
                #EER = calc_eer(preds,labels)
                #print ("EER: %.8f" % EER)

                #print ("Lengths:",len(preds),len(labels))
                #print (preds)
                #preds.clear()
                #labels.clear() 
                acc_values.clear()
	
    bar.finish()

    for x in range(11):
        print (["{0:0.0f}".format(i) for i in confusion_matrix.tolist()[x]])
        print ("\n")
    cm_class_accs = (confusion_matrix.diag()/confusion_matrix.sum(1)*100).tolist()
    print (["{0:0.2f}".format(i) for i in cm_class_accs])
    #print (len(cm_class_accs))
    final_acc = sum(cm_class_accs)/len(cm_class_accs)
    print ("\n")
    print ("Average Accuracy: {0:0.2f}%".format(final_acc))
    print ("\n")


def matplotlib_imshow(img, one_channel=False):
    """
    Helper function to show image
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5 #unnormalise
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)))


class EarlyStopping:
	
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7):
        """
        Args:
        patience (int): How long to wait after last time validation loss improved.
        Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement. 
        Default: False
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, optimizer, epoch):
	
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,optimizer, epoch)
        elif score < self.best_score:
            self.counter += 1
            _LOGGER.info("Early stopping counter: %d out of %d", self.counter, self.patience)
            #self.best_score = score
            #self.val_loss_min = val_loss
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,optimizer, epoch)
            self.counter = 0
            #self.val_loss_min = val_loss

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        '''Saves model when validation loss decrease.'''
        _LOGGER.info("Early stopping: Validation Loss Decreased from %.6f to %.6f, Saving Model...",
        self.val_loss_min, val_loss)
        self.val_loss_min = val_loss
		
        #torch.save(model.state_dict(), 'checkpoint.pth')
        torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, 'checkpoint.pth')
		




#MODEL
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.fc_1 = nn.Linear(in_features = 112896, out_features = 11, bias = True)

        self.dropout_fc = nn.Dropout(p=0.5)
        #for m in self.modules():
			
        #    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                #print("Updating weights")
        #        nn.init.normal_(m.weight, mean = 0, std = 0.01)
        #        nn.init.constant_(m.bias, 0)
   
    def forward(self, x):
        conv1    	= self.conv1(x)
        relu1   	= F.relu(conv1)
        #drop1           = self.dropout_conv(relu1)
        norm1           = F.local_response_norm(relu1, size=5, alpha=1.0, beta=0.75, k=1.0)
        pool1           = F.max_pool2d(norm1, kernel_size=(2, 2), stride=(2, 2), padding=(0,), ceil_mode=True)
        conv2           = self.conv2(pool1)
        relu2           = F.relu(conv2)
        #drop2           = self.dropout_conv(relu2)
        norm2           = F.local_response_norm(relu2, size=5, alpha=1.0, beta=0.75, k=1.0)
        pool2           = F.max_pool2d(norm2, kernel_size=(2, 2), stride=(2, 2), padding=(0,), ceil_mode=True)
        conv3           = self.conv3(pool2)
        relu3 		= F.relu(conv3)
        #norm3           = F.local_response_norm(relu3, size=5, alpha=1.0, beta=0.75, k=1.0)
        #pool3           = F.max_pool2d(relu3, kernel_size=(2, 2), stride=(2, 2), padding=(0,), ceil_mode=True)
        drop3           = self.dropout_fc(relu3)
        #dropout        = F.dropout(input = relu3, p = 0.5, training = self.training, inplace = True)
        #fc_0            = pool3.view(pool3.size(0),-1) 
        fc_0            = drop3.view(drop3.size(0), -1)
        fc_1            = self.fc_1(fc_0)
        #drop            = self.dropout_fc(fc_1)
        #fc_2            = self.fc_2(drop)
        return fc_1
	

    
		
if __name__=='__main__':
    
    logging.basicConfig(level=logging.INFO)	
	
    parser = argparse.ArgumentParser(description="Train gait CNN.")
	
    parser.add_argument(
        'data_dir',
        help="Directory of the data.") 
	
    parser.add_argument(
        '--learning_rate',
        default=0.0001,
        type=float,
        help="Learning rate for the model. Default: 0.0001.")   

    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help="Batch size to use. Default: 64.") 

    parser.add_argument(
        '--epochs',
        default=1,
        type=int,
        help="Number of epochs to train for. Default: 1.") 
	
    parser.add_argument(
        '--mode',
        default='train',
        choices=['train','test', 'resume'],
        help="Mode of operation, train or test. Default: train.")	   
     
    parser.add_argument(
        '--name',
        default='temp_train',
        help="Name of the folder to store trained models. Default: models/temp_train.")	   
   
    args = parser.parse_args()

    main(args.data_dir, args.learning_rate, args.batch_size, args.epochs, args.mode, args.name)
