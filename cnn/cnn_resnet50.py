# Author: Chen Shi
# CNN 动物识别

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import time
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import font
from tkinter import filedialog

### Parameters ###
# epoch size
epochs = 2
# batch size
bs = 32
# set dataset path
dataset_path = './dataset/cnn_dataset'
# set train and valid directory paths
train_directory = dataset_path + '/train'
valid_directory = dataset_path + '/valid'
test_directory = dataset_path + '/test'

CATEGORY_NAMES = []
# traverse train dir to get all classes 
for dir_name in os.listdir(f'{dataset_path}/train'):
        CATEGORY_NAMES.append(str(dir_name))
# number of classes
# should be updated after get CATEGORY_NAMES
num_classes = len(CATEGORY_NAMES)

### End ###

'''
Function Class
'''
class CNNSystem:
    def __init__(self):
        # self.apply_transforms()
        # self.load_data()
        # self.load_model()
        # self.training()
        pass

    '''
    Applying Transforms to the Data
    '''
    def apply_transforms(self):
        self.image_transforms = { 
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }

    '''
    Load the Data
    '''
    def load_data(self):
        # first, apply transforms of data
        self.apply_transforms()

        # Load Data from folders
        self.data = {
            'train': datasets.ImageFolder(root=train_directory, transform=self.image_transforms['train']),
            'valid': datasets.ImageFolder(root=valid_directory, transform=self.image_transforms['valid']),
            'test': datasets.ImageFolder(root=test_directory, transform=self.image_transforms['test'])
        }

        # Size of Data, to be used for calculating Average Loss and Accuracy
        self.train_data_size = len(self.data['train'])
        self.valid_data_size = len(self.data['valid'])
        self.test_data_size = len(self.data['test'])

        # Create iterators for the Data loaded using DataLoader module
        self.train_data_loader = torch.utils.data.DataLoader(self.data['train'], batch_size=bs, shuffle=True)
        self.valid_data_loader = torch.utils.data.DataLoader(self.data['valid'], batch_size=bs, shuffle=True)
        self.test_data_loader = torch.utils.data.DataLoader(self.data['test'], batch_size=bs, shuffle=True)
 
    '''
    Load pretrained ResNet50 Model
    '''
    def load_model(self):
        self.model = torchvision.models.resnet50(pretrained=True)

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Change the final layer of ResNet50 Model for Transfer Learning
        fc_inputs = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10), 
            nn.LogSoftmax(dim=1) # For using NLLLoss()
        )

        # Define Optimizer and Loss Function
        self.loss_criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    '''
    Training
    '''
    def training(self):
        history = []
        for epoch in range(epochs):
            epoch_start = time.time()
            print("Epoch: {}/{}".format(epoch+1, epochs))

            # Set to training mode
            self.model.train()

            # Loss and Accuracy within the epoch
            train_loss = 0.0
            train_acc = 0.0

            valid_loss = 0.0
            valid_acc = 0.0

            for i, (inputs, labels) in enumerate(self.train_data_loader):
            
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Clean existing gradients
                self.optimizer.zero_grad()

                # Forward pass - compute outputs on input data using the model
                outputs = self.model(inputs)

                # Compute loss
                loss = self.loss_criterion(outputs, labels)

                # Backpropagate the gradients
                loss.backward()

                # Update the parameters
                self.optimizer.step()

                # Compute the total loss for the batch and add it to train_loss
                train_loss += loss.item() * inputs.size(0)

                # Compute the accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to train_acc
                train_acc += acc.item() * inputs.size(0)

                print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

            # Validation - No gradient tracking needed
            with torch.no_grad():
            
                # Set to evaluation mode
                self.model.eval()

                # Validation loop
                for j, (inputs, labels) in enumerate(self.valid_data_loader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass - compute outputs on input data using the model
                    outputs = self.model(inputs)

                    # Compute loss
                    loss = self.loss_criterion(outputs, labels)

                    # Compute the total loss for the batch and add it to valid_loss
                    valid_loss += loss.item() * inputs.size(0)

                    # Calculate validation accuracy
                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))

                    # Convert correct_counts to float and then compute the mean
                    acc = torch.mean(correct_counts.type(torch.FloatTensor))

                    # Compute total accuracy in the whole batch and add to valid_acc
                    valid_acc += acc.item() * inputs.size(0)

                    print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

            # Find average training loss and training accuracy
            avg_train_loss = train_loss / self.train_data_size 
            avg_train_acc = train_acc/float(self.train_data_size)

            # Find average training loss and training accuracy
            avg_valid_loss = valid_loss / self.valid_data_size 
            avg_valid_acc = valid_acc/float(self.valid_data_size)

            history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

            epoch_end = time.time()

            print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        return history

    '''
    predict label of an image
    '''
    def predict(self, test_image_name: str):
        transform = self.image_transforms['test']
    
        test_image = Image.open(test_image_name)
        # plt.imshow(test_image)

        test_image_tensor = transform(test_image)
    
        if torch.cuda.is_available():
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
        else:
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

        with torch.no_grad():
            self.model.eval()
            # Model outputs log probabilities
            out = self.model(test_image_tensor)
            ps = torch.exp(out)
            topk, topclass = ps.topk(1, dim=1)
            # print("Predicted class:", CATEGORY_NAMES[topclass.cpu().numpy()[0][0]])
            predict_result = CATEGORY_NAMES[topclass.cpu().numpy()[0][0]]
            return predict_result

'''
UI Class
单线程，存在阻塞
'''
class MainWindow:
    def __init__(self):
        # main window
        self.root = tk.Tk()

        self.root.title("ResNet50 CNN 动物识别系统")

        # frame
        self.top_frame = tk.Frame(self.root, width=750, height=150)
        self.left_frame = tk.Frame(self.root, width=200, height=200)
        self.main_frame = tk.Frame(self.root, width=525, height=200,
                                   highlightbackground="black", highlightthickness=1, bd=1)
        self.bottom_frame = tk.Frame(self.root, width=750, height=150)
        self.top_frame.grid_propagate(0)
        self.top_frame.grid(row=0, column=0, columnspan=3, padx=5, pady=5)
        self.left_frame.grid_propagate(0)
        self.left_frame.grid(row=1, column=0, padx=5, pady=5)
        self.main_frame.grid_propagate(0)
        self.main_frame.grid(row=1, column=1, padx=5, pady=5)
        # self.bottom_frame.grid_propagate(0)
        self.bottom_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

        # instruction labels in top_frame
        self.label0 = tk.Label(self.top_frame, text = "执行流程：", anchor='w', 
                                        font = font.Font(size=14, family='Microsoft YaHei'), width = 40)
        self.label1 = tk.Label(self.top_frame, text = "1. 加载数据集（裁剪）", anchor='w',  
                                        font = font.Font(size=12, family='Microsoft YaHei'), width = 40)
        self.label2 = tk.Label(self.top_frame, text = "2. 加载模型", anchor='w', 
                                        font = font.Font(size=12, family='Microsoft YaHei'), width = 40)
        self.label3 = tk.Label(self.top_frame, text = "3. 开始训练", anchor='w', 
                                        font = font.Font(size=12, family='Microsoft YaHei'), width = 40)
        self.label4 = tk.Label(self.top_frame, text = "4. 输入待分类图片", anchor='w', 
                                        font = font.Font(size=12, family='Microsoft YaHei'), width = 40)
        self.label0.grid(row = 0)
        self.label1.grid(row = 1)
        self.label2.grid(row = 2)
        self.label3.grid(row = 3)
        self.label4.grid(row = 4)

        # operation buttons in left_frame
        self.button_load_data = tk.Button(self.left_frame, text = "加载数据集", 
                                        font = font.Font(size=12, family='Microsoft YaHei'), width=15,
                                        command = self.cnn_load_data)
        self.button_load_model = tk.Button(self.left_frame, text = "加载模型", 
                                        font = font.Font(size=12, family='Microsoft YaHei'), width=15,
                                        command = self.cnn_load_model)
        self.button_start_training = tk.Button(self.left_frame, text = "开始训练", 
                                        font = font.Font(size=12, family='Microsoft YaHei'), width=15,
                                        command = self.cnn_start_training)
        self.button_open_image = tk.Button(self.left_frame, text = "打开图片", 
                                        font = font.Font(size=12, family='Microsoft YaHei'), width=15,
                                        command = self.open_image)
        self.button_clear_text_result = tk.Button(self.left_frame, text = "清空输出", 
                                        font = font.Font(size=12, family='Microsoft YaHei'), width=15,
                                        command = self.clear_text_result)
        self.button_load_data.grid(row = 0, column = 0)
        self.button_load_model.grid(row = 1, column = 0)
        self.button_start_training.grid(row = 2, column = 0)
        self.button_open_image.grid(row = 3, column = 0)
        self.button_clear_text_result.grid(row = 4, column = 0)
        
        # image viewer in main_frame
        self.label_image = tk.Label(self.main_frame)
        self.label_image.grid(row = 0)

        # output text in bottom_frame
        self.text_result = tk.Text(self.bottom_frame, width = 80, height = 10, 
                                    font = font.Font(size=12, family='Consolas'))
        self.text_result.grid(row = 0)
        # vertical scrollbar
        self.scroll_vertical = tk.Scrollbar(self.bottom_frame)
        self.scroll_vertical.grid(row = 0, column = 1, sticky = 'ns')
        self.text_result.config(yscrollcommand = self.scroll_vertical.set)
        self.scroll_vertical.config(command = self.text_result.yview)
        # horizental scrollbar
        self.scroll_horizental = tk.Scrollbar(self.bottom_frame, orient = tk.HORIZONTAL)
        self.scroll_horizental.grid(row=1, column=0, sticky='ew')
        self.text_result.config(xscrollcommand = self.scroll_horizental.set)
        self.scroll_horizental.config(command=self.text_result.xview)

        # 创建产生式系统类
        self.cnn_sys = CNNSystem()

        self.root.mainloop()

    '''
    加载数据集
    '''
    def cnn_load_data(self):
        self.cnn_sys.load_data()

        self.print_to_text_result("Loading data completed")
        # print categories
        self.print_to_text_result(f"Category nums: {num_classes}")
        self.print_to_text_result(f"Categories: {str(CATEGORY_NAMES)}")

    '''
    加载模型
    '''
    def cnn_load_model(self):
        self.cnn_sys.load_model()
        
        self.print_to_text_result("Loading ResNet50 model completed")

    '''
    执行训练
    '''
    def cnn_start_training(self):
        history = self.cnn_sys.training()
        
        self.print_to_text_result("Training completed.")
        self.print_to_text_result("===== History Training output =====")
        # print history
        for i in range(len(history)):
            self.print_to_text_result(f"Epoch {i}:")
            self.print_to_text_result(f"\tavg_train_loss: {history[i][0]}\n\tavg_valid_loss: {history[i][1]}\n\tavg_train_cc: {history[i][2]}\n\tavg_valid_acc: {history[i][3]}")
        self.print_to_text_result("=============== End ===============")

    '''
    执行预测
    '''
    def cnn_predict(self, image_name: str):
        res = self.cnn_sys.predict(image_name)

        self.print_to_text_result(f"Predict animal: {res}")

    '''
    resize images
    '''
    def __resize(self, w, h, w_box, h_box, pil_image):
        ''' 
        resize a pil_image object so it will fit into 
        a box of size w_box times h_box, but retain aspect ratio 
        对一个 pil_image 对象进行缩放，让它在一个矩形框内，还能保持比例 
        '''  
        f1 = 1.0 * w_box / w
        f2 = 1.0 * h_box / h  
        factor = min([f1, f2])  
        #print(f1, f2, factor) # test  
        # use best down-sizing filter  
        width = int(w * factor)  
        height = int(h * factor)  
        return pil_image.resize((width, height))

    '''
    打开图片、显示至 label_image
    '''
    def open_image(self):
        # open file dialog and select an image
        image_file_path = filedialog.askopenfilename(
            title = "选择图片",
            filetypes = [("Image Files", "*.jpg *.jpeg *.png *.gif *.bmp")]
        )
    
        if image_file_path:
            image = Image.open(image_file_path)
            image_w, image_h = image.size
            image_resize = self.__resize(image_w, image_h, 360, 180, image)
            # change to imagetk
            image_tk = ImageTk.PhotoImage(image_resize)

            # 显示到 label_image
            self.label_image.config(image = image_tk)
            self.label_image.image = image_tk

            # clear text_result
            # self.clear_text_result()
            # 打印图片的文件路径
            self.print_to_text_result(f"Image file path: {image_file_path}")

            # 执行动物识别
            self.cnn_predict(image_file_path)

    '''
    清空 text_result
    '''
    def clear_text_result(self):
        self.text_result.delete("1.0", "end")

    '''
    打印至 text_result
    '''
    def print_to_text_result(self, print_str: str):
        self.text_result.insert(tk.END, f"{print_str}\n")
        self.text_result.see(tk.END)

if __name__ == "__main__":
    main_window = MainWindow()