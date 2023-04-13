import torch 
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import  transforms
import wandb
from PIL import ImageFile, Image, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import timm
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model_classify(nn.Module):
    def __init__(self):
        super(Model_classify, self).__init__()
        self.backbone = timm.create_model('efficientnet_b2', pretrained=True, in_chans = 3, bn_eps=0.001)
        self.backbone.classifier = nn.Linear(in_features=1408, out_features=7, bias=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    
    
class data_loader(Dataset):
    def __init__(self, txt_path, transform=None):
        self.txt_path = txt_path
        self.transform = transform
        with open(txt_path, 'r') as f:
            self.lines = f.read().splitlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_path = self.lines[idx].split(' ')[0] + ' ' + self.lines[idx].split(' ')[1]
        label = int(self.lines[idx].split(' ')[2]) - 1
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')

        if self.transform :
            img = self.transform(img)
        return img, label   


transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Run_model():
    def __init__(self, model_classify, train_loader, val_loader, num_epochs, optimizer, criterion):
        self.model_classify = model_classify.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.best_f1 = 0

    def train(self):
        print("Training in epoch {}".format(self.num_epochs))
        self.model_classify.train()
        total_loss = 0
        total_correct = 0
        total = 0
        preds, targetss = [], []

        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model_classify(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            total_correct += predicted.eq(targets).sum().item()
            preds += predicted.tolist()
            targetss += targets.tolist()

        train_loss = total_loss / len(self.train_loader)
        train_acc = total_correct / total
        train_f1 = f1_score(targetss, preds, average='macro')
        train_recall = recall_score(targetss, preds, average='macro')
        train_precision = precision_score(targetss, preds, average='macro')

        print('Train Loss: {:.4f} | Train Acc: {:.4f} | Train F1: {:.4f} | Train Recall: {:.4f} | Train Precision: {:.4f}'.format(train_loss, train_acc, train_f1, train_recall, train_precision))

        wandb.log({'Train Loss': train_loss, 
                    'Train Acc': train_acc, 
                    'Train F1': train_f1, 
                    'Train Recall': train_recall, 
                    'Train Precision': train_precision})


    def val(self):
        print("Validating in epoch {}".format(self.num_epochs))
        self.model_classify.eval()
        total_loss = 0
        total_correct = 0
        total = 0
        preds, targetss = [], []

        for batch_idx, (inputs, targets) in enumerate(tqdm(self.val_loader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)          
            with torch.no_grad():
                outputs = self.model_classify(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                total_correct += predicted.eq(targets).sum().item()
                preds += predicted.tolist()
                targetss += targets.tolist()
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = total_correct / total
        val_f1 = f1_score(targetss, preds, average='macro')
        val_recall = recall_score(targetss, preds, average='macro')
        val_precision = precision_score(targetss, preds, average='macro')

        print('Val Loss: {:.4f} | Val Acc: {:.4f} | Val F1: {:.4f} | Val Recall: {:.4f} | Val Precision: {:.4f}'.format(val_loss, val_acc, val_f1, val_recall, val_precision))

        wandb.log({
            'Val Loss': val_loss,
            'Val Acc': val_acc,
            'Val F1': val_f1,
            'Val Recall': val_recall,
            'Val Precision': val_precision
        })

        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            torch.save(self.model_classify.state_dict(), 'checkpoint_efficientnet_b2.pt')
            torch.jit.save(torch.jit.script(self.model_classify), 'checkpoint_efficientnet_b2.pt')
            print('Best model saved')
            
            
            
def config_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def input_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--path_train', type=str, default="")
    parser.add_argument('--path_val', type=str, default="")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wd', type=float, default=0.0001)


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = input_params()
    config_seed(args.seed)

    train_dataset = data_loader(txt_path= args.path_train, transform=transforms)
    val_dataset = data_loader(txt_path= args.path_val, transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size= args.batch_size, shuffle=True, num_workers=4)

    model = Model_classify()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    wandb.init(project="Emotion", name="Efficientnet_b2")
    wandb.watch(model)

    run_model = Run_model(model, train_loader, val_loader, args.epochs, optimizer , criterion)

    for epoch in range(args.epochs):
        run_model.num_epochs = epoch
        run_model.train()
        run_model.val()