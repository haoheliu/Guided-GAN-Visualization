from __future__ import print_function
import torch    
import torch.nn as nn
import torchvision.datasets as dset
from collections import OrderedDict
import torchvision.transforms as transforms
import torch.utils.data
import torch.utils.model_zoo as model_zoo
from PIL import Image
import math
import argparse
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision
import pickle
import matplotlib.pyplot as plt

from corelation import corelation


device = 'cuda:0'
#Total number of iter we want to generate fake picture we want, the more iter, more chance to generate more fake images
max_iter = 6000
#threshold = 0.6
#How many fake picture we want for each digit
num_generate = 3000
#In what distribution we sample noise vector
noise_distribution = 'normal' # option: 'uniform','normal'

###########
ngpu = 1
nz = 2
ngf = 64
ndf = 64
nc = 1 # By default, could be modified using command line
###########

#Mnist: so image size is 28
classifier_input_imagesize = 28

#Depend on how we train our generator
if(noise_distribution == 'normal'):
    random_collection = torch.randn(max_iter,1,nz,1,1)
elif(noise_distribution == 'uniform'):
    random_collection = torch.from_numpy(np.random.uniform(-1,1,(max_iter,1,nz,1,1)))
    random_collection = random_collection.type(torch.FloatTensor)


def generator_visulization( resume = './data/pretrained_generator/netG_epoch_58_mnist_2attri_m.pth',
                            model_dict = './data/pretrained_classifier/CNN_mnist.pth',
                            ):
    #Save the coordination of labeled noise vectors
    coor_dict = {}
    for i in range(0,10):
        coor_dict[i] = []
    #Save the score of labeled noise vectors
    score_dict = {}
    for each in range(0,10):
        score_dict[each] =  {}

    transform=transforms.Compose([
                                   transforms.Resize(classifier_input_imagesize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                               ])
    #Generator model
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
	        # input is Z, going into a convolution
	        #But I don't quite understand why input size to the 'forward' function should be (n,nz,1,1)
	        nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
	        nn.BatchNorm2d(ngf * 8),
	        nn.ReLU(True),
	        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
	        nn.BatchNorm2d(ngf * 4),
	        nn.ReLU(True),
	        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
	        nn.BatchNorm2d(ngf * 2),
	        nn.ReLU(True),
	        nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
	        nn.BatchNorm2d(ngf),
	        nn.ReLU(True),
	        nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
	        nn.Tanh()
            )

        def forward(self, inputs):
            if inputs.is_cuda and self.ngpu > 1:
	            output = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
            else:
	            output = self.main(inputs)
            return output

    #Classifier model
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
	        nn.Conv2d(
	            in_channels=1,              # input height
	            out_channels=16,            # n_filters
	            kernel_size=5,              # filter size
	            stride=1,                   # filter movement/step
	            padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
	        ),                              # output shape (16, 28, 28)
	        nn.ReLU(),                      # activation
	        nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
            )
            self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
	        nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
	        nn.ReLU(),                      # activation
	        nn.MaxPool2d(2),                # output shape (32, 7, 7)
            )
            self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
            output = self.out(x)
            return output, x    # return x for visualization

    #load pre-trained model
    print('>> Resuming from Classifier checkpoint..')
    model = CNN().cuda()
    model.load_state_dict(torch.load(model_dict)) 
    print('>>Resume checkpoint finished')

    print('>> Resuming from Generator checkpoint..')
    G = Generator(ngpu).to(device)
    G.load_state_dict(torch.load(resume))
    print('>>Resume checkpoint finished')

    #Classifier the fake result of Generator
    def classify(G_result):
        vutils.save_image(G_result.detach(),
                               './data/temp.png',
                                normalize=True,
                                nrow = 1)
        img_test = np.array(Image.open('./data/temp.png'))
        ndarray_convert_img= Image.fromarray(G_result[0,0,:,:].cpu().detach().numpy())
        pic = transform(ndarray_convert_img)
        temp = torch.zeros(1,1,classifier_input_imagesize,classifier_input_imagesize)
        temp[0,0,:,:] = pic
        temp = temp.cuda()
        result,_ = model(temp)     
        result = result[0].cpu().detach().numpy().tolist()
        index = result.index(max(result))  #The regonition result for the generated picture
        img_standard = np.array(Image.open('./data/standerd_pic/'+str(index)+'_standerd.png'))
        score = corelation(img_test,img_standard)
        return index,score

    #Construct the basic data if we don't have
    dic_name  = './data/coor_dict/coor_dict_mnist'+str(max_iter)+'_'+resume.split('/')[-1]+'_'+str(nz)+'.pkl'
    if(not os.path.exists(dic_name)):
        print(dic_name+' not Found. Start construction...')
        for cnt in range(0,max_iter):
            noise = random_collection[cnt]
            noise = noise.cuda()
            G_result = G(noise)
            index,score = classify(G_result)
            score_dict[index][score] = [noise[0,0,0,0].item(),noise[0,1,0,0].item()]

        for each in range(0,10):
            order = sorted(score_dict[each].keys())
            cnt = num_generate
            for i in range(0,cnt):
                try:
                    coor_dict[each].append(score_dict[each][order[-i-1]])
                except:
                    #print('Not enough :'+str(each))
                    #print('You need '+str(num_generate)+" pictures of :"+str(each))
                    #print('But we only have:'+str(i)+' '+str(each))
                    break

        print('Finished')
        print('Saving dictionary: '+dic_name)
        f = open(dic_name,'wb')
        pickle.dump(coor_dict,f)
        f.close()

    else:
        print('Load data from :'+dic_name)
        f = open(dic_name,'rb')
        coor_dict = pickle.load(f)
        f.close()

    #Evaluate of generator
    def evaluate(k1 = 0,k2 = 0,k3 = 30,pic_num = 40,criteria = 0):

        angle = np.linspace(0,2*math.pi,pic_num)
        
        noise_x = np.sin(angle)
        noise_y = np.cos(angle)

        squence = []
        col_score = []
    
        def score(sq,cs): 
            num_classes = len(set(sq))
            num_divided  = 0
            now = sq[0]
            for i in range(1,len(sq)):
                if(sq[i] == sq[i-1]):
                    continue
                else:
                    num_divided+=1
            score = k1*(1/num_classes)+k2*(num_divided/num_classes)+k3*(1/num_classes)*pic_num/(sum(cs)-criteria)
            return score

        for each in zip(noise_x,noise_y):
            noise = torch.zeros(1,nz,1,1)
            noise[0,0,0,0] = each[0]  #We can try to switch these two coo to see what will happen
            noise[0,1,0,0] = each[1]
            G_result = G(noise.cuda())
            index,col = classify(G_result)
            col_score.append(col)
            squence.append(index)
        
        result = score(squence,col_score)
        return result

    #Plot the distribution figure  of noise vectors
    def distribution_plot(rou = 0.4,pic_num = 40,xticks = [-4,4],yticks = [-4,4],img_size = 0.05,plot_size = 5):

        fig = plt.figure()                              
        ax0 = plt.gca() 
        ax0.xaxis.set_ticks_position('bottom')   
        ax0.yaxis.set_ticks_position('left')          

        ax0.spines['bottom'].set_position(('data', 0))   
        ax0.spines['left'].set_position(('data', 0))
        ax0.spines['bottom'].set_linewidth(1.5)   
        ax0.spines['left'].set_linewidth(1.5)

        plt.yticks([0.5,1.0,1.5,2.0,2.5,3.0])

        for each in coor_dict[0]:
            sc0 = plt.scatter(each[0],each[1],s = plot_size,c = 'black',marker = '1')
        for each in coor_dict[1]:
            sc1 = plt.scatter(each[0],each[1],s = plot_size,c = 'blue',marker = '.')
        for each in coor_dict[2]:
            sc2 = plt.scatter(each[0],each[1],s = plot_size,c = 'red',marker = 'v')
        for each in coor_dict[3]:
            sc3 = plt.scatter(each[0],each[1],s = plot_size,c = 'green',marker = '*')
        for each in coor_dict[4]:
            sc4 = plt.scatter(each[0],each[1],s = plot_size,c = 'purple',marker = '+')
        for each in coor_dict[5]:
            sc5 = plt.scatter(each[0],each[1],s = plot_size,c = 'orange',marker = 'D')
        for each in coor_dict[6]:
            sc6 = plt.scatter(each[0],each[1],s = plot_size-1,c = 'yellow',marker = 'd')
        for each in coor_dict[7]:
            sc7 = plt.scatter(each[0],each[1],s = plot_size-2,c = 'gray',marker = 'P')
        for each in coor_dict[8]:
            sc8 = plt.scatter(each[0],each[1],s = plot_size,c = 'brown',marker = 'x')
        for each in coor_dict[9]:
            sc9 = plt.scatter(each[0],each[1],s = plot_size,c = 'blue',marker = 'X')
        plt.legend(['4'])
        plt.title('Most significant noise')

        path =  './data/pretrained_generator/image/image_'+str(max_iter)+'_'+resume.split('/')[-1]+'_'+str(nz)+'.png'
        plt.savefig(path)
        plt.close()
            
    #Plot the distribution of noise vectors with fake numbers
    def num_plot(rou = 0.4,pic_num = 40,xticks = [-4,4],yticks = [-4,4],img_size = 0.05,plot_size = 5):

        sc0,sc1,sc2,sc3,sc4,sc5,sc6,sc7,sc8,sc9 = [None]*10
        fig = plt.figure(figsize = (10,10))
        plt.xlabel('Dimension 0')
        plt.ylabel('Dimension 1')                                

        ax = fig.add_axes([0,0,1,1])
        ax0 = plt.gca()                                            

        ax0.xaxis.set_ticks_position('bottom')   
        ax0.yaxis.set_ticks_position('left')          

        ax0.spines['bottom'].set_position(('data', 0))   
        ax0.spines['left'].set_position(('data', 0))
        ax0.spines['bottom'].set_linewidth(1.5)   
        ax0.spines['left'].set_linewidth(1.5)

        plt.axis('equal')
        plt.tick_params(labelsize=15)
        plt.xticks([-2,-1.-0.5,0.5,1,2])
        plt.yticks([-2,-1,-0.5,0.5,1,2])
        plt.xlim(xticks)
        plt.ylim(yticks)


        for each in coor_dict[0]:
            sc0 = plt.scatter(each[0],each[1],s = plot_size,c = 'black',marker = '1')
        for each in coor_dict[1]:
            sc1 = plt.scatter(each[0],each[1],s = plot_size,c = 'blue',marker = '.')
        for each in coor_dict[2]:
            sc2 = plt.scatter(each[0],each[1],s = plot_size,c = 'red',marker = 'v')
        for each in coor_dict[3]:
            sc3 = plt.scatter(each[0],each[1],s = plot_size,c = 'green',marker = '*')
        for each in coor_dict[4]:
            sc4 = plt.scatter(each[0],each[1],s = plot_size,c = 'purple',marker = '+')
        for each in coor_dict[5]:
            sc5 = plt.scatter(each[0],each[1],s = plot_size,c = 'orange',marker = 'D')
        for each in coor_dict[6]:
            sc6 = plt.scatter(each[0],each[1],s = plot_size-1,c = 'yellow',marker = 'd')
        for each in coor_dict[7]:
            sc7 = plt.scatter(each[0],each[1],s = plot_size-2,c = 'gray',marker = 'P')
        for each in coor_dict[8]:
            sc8 = plt.scatter(each[0],each[1],s = plot_size,c = 'brown',marker = 'x')
        for each in coor_dict[9]:
            sc9 = plt.scatter(each[0],each[1],s = plot_size,c = 'blue',marker = 'X')
        
        plt.legend(handles = [sc0,sc1,sc2,sc3,sc4,sc5,sc6,sc7,sc8,sc9], labels = ['0', '1','2','3','4','5','6','7','8','9'], loc = 'best')
        angle = np.linspace(0,2*math.pi,pic_num)

        noise_x = np.sin(angle)
        noise_y = np.cos(angle)
        x = 0.5+np.sin(angle)*rou
        y = 0.5+np.cos(angle)*rou
        
        
        for each in zip(x,y,noise_x,noise_y):
            noise = torch.zeros(1,nz,1,1)
            noise[0,0,0,0] = each[2]  #We can try to switch these two coo to see what will happen
            noise[0,1,0,0] = each[3]
            G_result = G(noise.cuda())
            ax1 = fig.add_axes([each[0]-0.5*img_size,each[1]-0.5*img_size,img_size,img_size])
            ax.arrow(each[2]*2,each[3]*2,each[2]*0.62,each[3]*0.62,head_width = 0.03,head_length = 0.1,fc = 'k',ec = 'k')
            ax.scatter(each[2]*2,each[3]*2,s = plot_size+3,c = 'black')
            ax1.imshow(G_result[0,0,:,:].cpu().detach().numpy(),cmap = 'gray')
            plt.xticks([])
            plt.yticks([])


        path =  './data/pretrained_generator/image/image_'+str(max_iter)+'_'+resume.split('/')[-1]+'_'+str(nz)+'.png'
        plt.savefig(path)
        plt.close()

    #Specify a noise vector and generate fake image
    def generate_test(coo,fname = 'test_generate.png'):
        noise = torch.zeros(1,nz,1,1)
        noise[0,0,0,0] = coo[0]
        noise[0,1,0,0] = coo[1]
        result = G(noise.cuda())
        plt.figure()
        plt.imshow(result[0,0,:,:].cpu().detach().numpy(),cmap = 'gray')
        plt.show()
        vutils.save_image(result.detach(),
                               './data/result/'+fname,
                                normalize=True,
                                nrow = 1)
        
    #Generate a specific number
    def generate_number(num):
        if(len(coor_dict[num]) == 0):
            print('Error!')
            return None
        rand = random.randint(0,len(coor_dict[num])-1)
        return coor_dict[num][rand]
        
        result = G(noise.cuda())
        vutils.save_image(result.detach(),
                               './data/result/'+str(num)+'.png',
                                normalize=True,
                                nrow = 1)

    #Generate a handwriting string
    def generate_string(string):
        lis = list(str(string))
        noise = torch.zeros(len(lis),nz,1,1)
        for step,each in enumerate(lis):
            if(each == '.'):
                continue
            temp = generate_number(int(each))
            if(temp == None):
                continue
            noise[step,0,0,0] = temp[0]
            noise[step,1,0,0] = temp[1]
        result = G(noise.cuda())
        vutils.save_image(result.detach(),
                               './data/result/'+'stri4ng.png',
                                normalize=True,
                                nrow = 8)
    #Main function
    def main():
        num_plot()

    main()


if __name__ == '__main__':
    cnt = 0
    root = './data/pretrained_generator/group'
    lis = os.listdir('./data/pretrained_generator/group')
    if(len(lis) == 0):
        raise ValueError('Error: Not a single pretrained generator found, please refer to : for more details')
    else:
        print(str(len(lis))+' generators found, start analysising...')
    for each in lis:
        cnt+=1      
        path = root+'/'+each
        generator_visulization(resume = path)
        print('Number:',str(cnt),'finished!')


