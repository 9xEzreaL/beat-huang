import torch
import torch.nn as nn
import os
import argparse
import datetime
import json
from os.path import join
import torch.utils.data as data
from util.Metric import Metric
from util.helpers import Progressbar, add_scalar_dict
from tensorboardX import SummaryWriter
from torchsampler import ImbalancedDatasetSampler
from network.vision_model import hardnet, efficientnet, densenet_201, arc_efficientnet, Resnext, Resnest, my_densenet
import torch.optim as optim
from util.sam import SAM
from util.smooth_crossentropy import smooth_crossentropy
from util.by_pass import enable_running_stats, disable_running_stats
from util.configuration import dataset1, dataset2, dataset3, blood_type, save_logs, split, total_num


"""
Train a classfier model and save it so that you can use this classfier to test your generated data.
"""

attrs_default = [
     'blast', 'promyelo', 'myelo', 'meta', 'band', 'seg'
]


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')

    parser.add_argument('--img_size', dest='img_size', type=int, default=256)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='# of epochs')
    parser.add_argument('--bs_per_gpu', dest='batch_size_per_gpu', type=int, default=20) # training batch size
    parser.add_argument('--lr', dest='lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.9)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--net', dest='net', default='vgg')
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    parser.add_argument('--ckpt', dest='ckpt', default=None)
    parser.add_argument('--num_classes', dest='num_classes', type=int, default=6)



    return parser.parse_args(args)



class Classifier:
    def __init__(self, args, net):
        self.args = args
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.model = self.network_map(net)(num_classes=args.num_classes, use_meta=False)
        self.model.train()
        self.model.cuda()

        if self.multi_gpu:
            self.model = nn.DataParallel(self.model)

        # base_optim = optim.SGD
        # self.optim_model = SAM(self.model.parameters(), base_optim, lr=args.lr, momentum=0.9)
        self.optim_model = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9)
        if args.fine_tune:
            self.load(args.fine_tune, args.ckpt)


    def set_lr(self, lr):
        for g in self.optim_model.param_groups:
            g['lr'] = lr

    def train_model(self, img, label, metric): #(self, img, label) [0., 0., 0., 0., 1., 0.]
        for p in self.model.parameters():
            p.requires_grad = True
        # enable_running_stats(self.model)
        self.optim_model.zero_grad()
        pred = self.model(img)
        label = label.type(torch.int64)
        loss = smooth_crossentropy(pred, label, smoothing=0.1)
        loss.mean().backward()
        self.optim_model.step()

        #
        # dc_loss = smooth_crossentropy(pred, label, smoothing=0.1)
        # dc_loss.mean().backward()
        # self.optim_model.first_step(zero_grad=True) # this is optimizer sam
        #
        # disable_running_stats(self.model)
        # smooth_crossentropy(self.model(img), label, smoothing=0.1).mean().backward()
        # self.optim_model.second_step(zero_grad=True)

        _, predicted = pred.max(1)
        metric.update(predicted, label)
        acc = metric.accuracy()
        f1 = metric.f1()

        errD = {
            'd_loss': loss.mean().item()
        }
        return errD, acc, f1

    def eval_model(self, img, label, metric): #(self, img, label) [0., 0., 0., 0., 1., 0.]
        with torch.no_grad():
            pred = self.model(img)
        label = label.type(torch.float)

        _, predicted = pred.max(1)
        # _, targets = label.max(1)
        metric.update(predicted, label)
        acc = metric.accuracy()
        f1 = metric.f1()
        each_f1 = metric.f1(each_cls=True)

        return acc, f1, each_f1

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, path):
        states = {
            'model': self.model.state_dict(),
            'optim_model': self.optim_model.state_dict(),
        }
        torch.save(states, path)

    def load(self, fine_tune=False, ckpt=None):
        if fine_tune:
            states = torch.load(ckpt)
            self.model.load_state_dict(states['model'])
            for module in self.model.modules():
                # print(module)
                if isinstance(module, nn.BatchNorm2d):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()

    def network_map(self, net):
        network_mapping = {
            'densenet201': densenet_201,
            'efficientnet': efficientnet,
            'hardnet': hardnet,
            'arc_meta_efficientnet': arc_efficientnet,
            'resnext': Resnext,
            'resnest': Resnest,
            'my_densenet121': my_densenet,
        }
        return network_mapping[net]

if __name__=='__main__':
    args = parse()

    if args.ckpt is not None:
        args.fine_tune=True
        args.ckpt = os.path.join(save_logs, args.ckpt)
    else:
        args.fine_tune=False

    args.lr_base = args.lr
    args.n_attrs = len(args.attrs)
    args.betas = (args.beta1, args.beta2)

    os.makedirs(join(save_logs, args.experiment_name), exist_ok=True)
    os.makedirs(join(save_logs, args.experiment_name, 'checkpoint'), exist_ok=True)
    writer = SummaryWriter(join(save_logs, args.experiment_name, 'summary'))

    with open(join(save_logs, args.experiment_name, 'setting.txt'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

    num_gpu = torch.cuda.device_count()

    classifier = Classifier(args, net=args.net)
    progressbar = Progressbar()

    from loader.dataloader import BeatDataset
    train_dataset = BeatDataset(dataset1, dataset2, dataset3, blood_type, split, mode='train')
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size_per_gpu * num_gpu,
                                       num_workers=10, drop_last=True, sampler=ImbalancedDatasetSampler(train_dataset))

    eval_dataset = BeatDataset(dataset1, dataset2, dataset3, blood_type, split, mode='eval')
    eval_dataloader = data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size_per_gpu * num_gpu,
                                       num_workers=10, drop_last=True, shuffle=False)

    print('Training images:', len(train_dataset))
    print('Eval images:', len(eval_dataset))
    print('--------- Good Luck Beat Huang GO GO GO ----------')


    it = 0
    it_per_epoch = len(train_dataset) // (args.batch_size_per_gpu * num_gpu)
    for epoch in range(args.epochs):
        lr = args.lr_base * (0.9 ** epoch)
        classifier.set_lr(lr)
        classifier.train()
        writer.add_scalar('LR/learning_rate', lr, it + 1)
        metric_tr = Metric(num_classes=args.num_classes)
        metric_ev = Metric(num_classes=args.num_classes)
        for img, label, id in progressbar(train_dataloader):
            img = img.cuda()
            label = label.cuda()

            label = label.type(torch.float)
            img = img.type(torch.float)

            errD, acc, f1 = classifier.train_model(img, label, metric_tr)
            it += 1
            progressbar.say(epoch=epoch, d_loss=errD['d_loss'], acc=acc.item(), f1=f1.item())

        classifier.eval()
        for img, label, id in progressbar(eval_dataloader):
            img = img.cuda()
            label = label.cuda()

            img = img.type(torch.float)
            label = label.type(torch.float)

            acc, f1, each_f1 = classifier.eval_model(img, label, metric_ev)
            it += 1
            progressbar.say(epoch=epoch, acc=acc.item(), f1=f1.item())

            zipped = zip(attrs_default, list([x.item() for x in each_f1]))
        print(list(zipped))
        
        classifier.save(os.path.join(
            save_logs, args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
        ))


# training code
# CUDA_VISIBLE_DEVICES=3 python main.py --net densenet121 --experiment_name first
#　CUDA_VISIBLE_DEVICES=3 python main.py --net my_densenet121 --experiment_name my_densenet --bs_per_gpu 40
# fine tune code
# CUDA_VISIBLE_DEVICES=3 python main_first.py --net meta_densenet --experiment_name meta_densenet_sam_optim_512_1028_1030 --lr 0.0005 --ckpt meta_densenet_sam_optim_512_1028/checkpoint/weights.29.pth --gpu --batch_size 15