from public.parseArgs import ParseArgs
from models.resnet import resnet18AndMultiHeadAttention, ResNet
import torch
import os
from utils.imageDataLoader import ImageDataset
import torchvision.transforms as transforms
import pandas as pd

class Finetune():
    def __init__(self, args:ParseArgs) -> None:
        self.args = args
        self.device, self.device_ids = self.setup_device(n_gpu_use=self.args.gpu)
        self.model = self.loadModel()
        self.lossFunction = torch.nn.MSELoss()

        if not os.path.exists(self.args.save_model_dir):
            os.makedirs(self.args.save_model_dir)
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)

        self.img_transformer = [transforms.CenterCrop(args.image_size), transforms.ToTensor()]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        
        self.train_dataset = ImageDataset(datas=self.loadDataInfos("train"), datadir=os.path.join(self.args.datadir, self.args.dataset), img_transformer=transforms.Compose(self.img_transformer), normalize=self.normalize)
        self.val_dataset = ImageDataset(datas=self.loadDataInfos("val"), datadir=os.path.join(self.args.datadir, self.args.dataset), img_transformer=transforms.Compose(self.img_transformer), normalize=self.normalize)
        self.test_dataset = ImageDataset(datas=self.loadDataInfos("test"), datadir=os.path.join(self.args.datadir, self.args.dataset), img_transformer=transforms.Compose(self.img_transformer), normalize=self.normalize)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.args.batch,
                                                   shuffle=True,
                                                   num_workers=self.args.worker,
                                                   pin_memory=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                                    batch_size=self.args.batch,
                                                    shuffle=False,
                                                    num_workers=self.args.worker,
                                                    pin_memory=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.args.batch,
                                                    shuffle=False,
                                                    num_workers=self.args.worker,
                                                    pin_memory=True)
        self.model = self.model.cuda()
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        self.optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=10 ** self.args.weight_decay,
        )

    def loadDataInfos(self, dataType:str) -> list[dict]:
        if dataType == "train":
            dataPathcsv = os.path.join(self.args.datadir, self.args.dataset, "{}_train.csv".format(self.args.dataset))
            dataPathxlsx = os.path.join(self.args.datadir, self.args.dataset, "{}_train.xlsx".format(self.args.dataset))
        elif dataType == "test":
            dataPathcsv = os.path.join(self.args.datadir, self.args.dataset, "{}_test.csv".format(self.args.dataset))
            dataPathxlsx = os.path.join(self.args.datadir, self.args.dataset, "{}_test.xlsx".format(self.args.dataset))
        elif dataType == "val":
            dataPathcsv = os.path.join(self.args.datadir, self.args.dataset, "{}_val.csv".format(self.args.dataset))
            dataPathxlsx = os.path.join(self.args.datadir, self.args.dataset, "{}_val.xlsx".format(self.args.dataset))
        else:
            raise ValueError("Unsupported data type:{}".format(dataType))
        
        if os.path.exists(dataPathcsv):
            dfData = pd.read_csv(dataPathcsv)
        elif os.path.exists(dataPathxlsx):
            dfData = pd.read_excel(dataPathxlsx)
        else:
            raise ValueError("Data does not exist:{}".format(os.path.join(self.args.datadir, self.args.dataset, "{}_test".format(self.args.dataset))))
        dataDict = dfData.to_dict(orient='records')
        return dataDict

    def loadModel(self) -> ResNet:
        if self.args.model_type == "resnet18AndMultiHeadAttention":
            model = resnet18AndMultiHeadAttention(self.args.num_classes)
        else:
            raise ValueError("Unsupported model network:{}".format(self.args.model_type))

        if self.args.resume is not None:
            checkpoint = torch.load(self.args.resume, weights_only=False)
            ckp_keys = list(checkpoint['model_state_dict'])
            cur_keys = list(model.state_dict())
            model_sd = model.state_dict()
            for ckp_key in ckp_keys:
                model_sd[ckp_key] = checkpoint['model_state_dict'][ckp_key]
            model.load_state_dict(model_sd)
        return model
    
    def loadLoss(self) -> torch.nn.MSELoss:
        if self.args.loss_function == "mse":
            criterion = torch.nn.MSELoss()
        else:
            raise Exception("param {} is not supported.".format(self.args.loss_function))
        return criterion

    def setup_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        for step, data in enumerate(self.train_dataloader):
            images, labels, dataInfos = data
            images = images.to(self.device)
            labels = labels.to(self.device)
            pred = self.model(images)
            labels = labels.view(pred.shape).to(torch.float64)
            loss = self.lossFunction(pred.double(), labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(loss)
    
    def evaluate_train(self):
        pass

    def evaluate_val(self):
        pass

    def evaluate_test(self):
        pass

    def run(self):
        for ep in range(self.args.start_epoch, self.args.epoch):
            # 训练
            self.train()
            # 评估
            self.evaluate_train()
            self.evaluate_val()
            self.evaluate_test()
            # 计算PPV
            # 根据策略计算数据
            # 根据策略保存模型
            # 保存日志
            pass