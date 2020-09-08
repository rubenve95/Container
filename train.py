import torch
import numpy as np
import random
import argparse
import os
import sys
import time
from datetime import datetime
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm

from project_code.models.deepLab.deeplab import DeepLab
from project_code.loss import get_loss_function
from project_code.metrics.metrics import Metrics
from project_code.saver.saver import ModelSaver, ImgSaver, ArraySaver
from project_code.utils.utils import pretty_print, save_string
from project_code.optimizer import get_optimizer
from project_code.dataloader.get_loaders import LoaderGetter
from project_code.dataloader.utils import decode_seg_map_sequence
from homography.EM import EM

class Trainer():
    def __init__(self, options):
        random_seed = options['training']['random_seed']
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.options =  options
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logdir = os.path.join('runs', 'normal')#datetime.now().strftime('%Y-%m-%d_%H:%M'))
        self.writer = SummaryWriter(self.logdir)
        options_string = pretty_print(options)
        print(options_string)
        save_string(options_string, self.logdir)

        self.model = DeepLab(options).to(self.device)
        #print(model)

        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        self.model_saver = ModelSaver(options)
        self.model, self.epoch, self.best_acc, self.scheduler, self.optimizer = self.model_saver.load_model(self.model)
        self.optimizer = get_optimizer(options, self.model.parameters()) if self.optimizer is None else self.optimizer
        self.scheduler = StepLR(self.optimizer, step_size=1, 
                        gamma=options['training']['scheduler']['gamma']) if self.scheduler is None else self.scheduler
        self.criterion = get_loss_function()
        self.evaluator = Metrics(options['data']['num_classes'])

        #print("Total number of parameters:", sum(p.numel() for p in self.model.parameters())/1e6, "M")

        self.validation_folder = 'MVI_4627.MP4'
        self.loader_getter = LoaderGetter(options, specific_folders={'val': self.validation_folder})
        self.img_saver = ImgSaver()
        self.step = int(self.epoch*self.loader_getter.get_size('train')/options['training']['batch_size'])

    def validate(self):
        self.model.eval()
        self.evaluator.reset()
        evaluator2 = Metrics(options['data']['num_classes'])
        evaluator_vp = Metrics(options['data']['num_classes'])
        evaluator_vp.reset()
        post = EM(options['data']['root'], self.validation_folder)
        array_saver = ArraySaver()
        val_loader = self.loader_getter('val')
        loss = []
        #out_list = []
        #av_length = 5
        accuracy, accuracy_post, accuracy_vp = [], [], []
        names = []
        with torch.no_grad():
            for b,batch in tqdm(enumerate(val_loader)):
                evaluator2.reset()
                self.evaluator.reset()
                evaluator_vp.reset()

                name = batch['name'][0]

                # if name not in ['687.png', '205.png', '867.png', '1008.png', '777.png']:
                #     continue

                image = batch['image'].to(self.device)
                gt = batch['ground_truth'].to(self.device)

                skip_model = True

                if skip_model:
                    with open(os.path.join('data/results', 'segmentation_probs', self.validation_folder, name + '.npy'), 'rb') as f:
                        pred = np.load(f)
                else:
                    out = self.model(image)
                    loss.append(self.criterion(out, gt).item())

                    pred = out.data.cpu().numpy()
                    array_saver(pred, batch['name'][0] + '.npy', folder=os.path.join('segmentation_probs', self.validation_folder))

                gt = gt.cpu().numpy()
                pred0 = np.argmax(pred, axis=1)
                self.evaluator.add_batch(gt, pred0)
                pred0 = decode_seg_map_sequence(pred0)
                # decoded_gt = decode_seg_map_sequence(gt)
                self.img_saver(pred0[0], batch['name'][0], folder=os.path.join('segmentation_raw', self.validation_folder))

                start_time = time.time()
                skip_post = False

                if skip_post:
                    with open(os.path.join('data/results', 'polygons', self.validation_folder, name), 'rb') as f:
                        polygons = np.load(f)
                        face_labels = np.load(f)               
                else:
                    polygons,face_labels = post.EM(pred[0])
                    array_saver(polygons, batch['name'][0] + '.npy', folder=os.path.join('polygons', self.validation_folder), array1=face_labels)
                pred_post = post.get_segmentation_output(polygons, face_labels)
                pred_post = np.expand_dims(pred_post, axis=0)
                evaluator2.add_batch(gt, pred_post)
                #evaluator2.add_time(time.time() - start_time)

                #or i,o in enumerate(out):
                pred_post = decode_seg_map_sequence(pred_post)
                self.img_saver(pred_post[0], batch['name'][0], folder=os.path.join('segmentation_post', self.validation_folder))

                t2 = time.time()

                print('post time', t2 - start_time)

                skip_vp = False

                if skip_vp:
                    with open(os.path.join('data/results', 'polygons_vp', self.validation_folder, name), 'rb') as f:
                        new_polygons = np.load(f)
                        face_labels = np.load(f) 
                else:
                    new_polygons = post.opt_vp(polygons, face_labels, pred[0])
                pred_post = post.get_segmentation_output(new_polygons, face_labels)
                pred_post = np.expand_dims(pred_post, axis=0)
                evaluator_vp.add_batch(gt, pred_post)
                evaluator_vp.add_time(time.time() - start_time)

                #or i,o in enumerate(out):
                pred_post = decode_seg_map_sequence(pred_post)

                #for i,p in enumerate(pred1):
                #     #nb = b*options['training']['batch_size'] + i
                self.img_saver(pred_post[0], batch['name'][0], folder=os.path.join('segmentation_vp', self.validation_folder))
                array_saver(new_polygons, batch['name'][0] + '.npy', folder=os.path.join('polygons_vp', self.validation_folder), array1=face_labels)

                print('vp time', t2 - time.time())

                accuracy.append(self.evaluator.Pixel_Accuracy())
                accuracy_post.append(evaluator2.Pixel_Accuracy())
                accuracy_vp.append(evaluator_vp.Pixel_Accuracy())
                #self.best_acc = max(accuracy, self.best_acc)
                # print_string = 'VALIDATION: Step {} Loss {:.2f} Accuracy {:.2f}%'.format(self.step,
                #                 sum(loss)/len(loss), accuracy*100)
                #print(print_string)
                #avg_time = evaluator_vp.get_average_time()
                print("Name {} First Accuracy {:.2f} Post Accuracy {:.2f} VP Accuracy {:.2f}".format(name ,accuracy[-1]*100, accuracy_post[-1]*100, accuracy_vp[-1]*100))
                names.append(name)
                if b % 10 == 0:
                    print("Overall scores: First Accuracy {:.4f} Post Accuracy {:.4f} VP Accuracy {:.4f}".format(sum(accuracy)/len(accuracy)*100, 
                                                                                    sum(accuracy_post)/len(accuracy_post)*100, sum(accuracy_vp)/len(accuracy_vp)*100))
        #save_string(print_string, self.logdir)

        print("Overall scores: First Accuracy {:.4f} Post Accuracy {:.4f} VP Accuracy {:.4f}".format(sum(accuracy)/len(accuracy)*100, 
                                                                                    sum(accuracy_post)/len(accuracy_post)*100, sum(accuracy_vp)/len(accuracy_vp)*100))
        self.evaluator.reset()

        acc_performance = [vp - post for post,vp in zip(accuracy_post, accuracy_vp)]
        names = [n for a,n in sorted(zip(acc_performance, names))]
        acc_performance = sorted(acc_performance)
        for name,p in zip(names, acc_performance):
            print(name, p)

    def train(self):
        while True:
            print()
            print("Start epoch {}".format(self.epoch))

            train_loader = self.loader_getter('train')
            self.evaluator.reset()

            for b,batch in enumerate(train_loader):
                if len(batch['image']) != self.options["training"]["batch_size"]:#is this even necessary?
                    break
                start_time = time.time()

                image = batch['image'].to(self.device)
                gt = batch['ground_truth'].to(self.device)

                self.model.train()

                # Predict with model
                out = self.model(image)
                loss = self.criterion(out, gt)

                # Do backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Add batch to evaluator
                pred = out.data.cpu().numpy()
                gt = gt.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                self.evaluator.add_batch(gt, pred)
                self.evaluator.add_time(time.time() - start_time)

                # pred = decode_seg_map_sequence(pred)
                # for i,p in enumerate(pred):
                #     nb = b*options['training']['batch_size'] + i
                #     self.img_saver(p.permute(2,1,0), batch['name'][i] + '.png', folder='train')

                self.step += 1
                if self.step % self.options['training']['print_interval'] == 0:
                    accuracy = self.evaluator.Pixel_Accuracy()
                    avg_time = self.evaluator.get_average_time()/self.options['training']['batch_size']
                    print_string = 'Step {} Time per image {:.2f} Loss {:.2f} Accuracy {:.2f}%'.format(self.step,
                         avg_time, loss, accuracy*100)
                    print(print_string)
                    save_string(print_string,self.logdir)

                #if self.step % self.options['training']['val_interval'] == 0:
                #     self.model_saver.save_model(self.model, self.epoch)

            self.scheduler.step()

            self.model_saver.save_model(self.model, self.epoch, self.best_acc, self.scheduler, self.optimizer)
            #self.validate()

            self.epoch += 1
            if self.epoch == self.options['training']['total_epochs']:
                print('Last epoch reached')
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="yo",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    args.config = os.path.join("configs", args.config + ".yml")

    with open(args.config) as fp:
        options = yaml.load(fp)

    if options["gpu_id"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(options["gpu_id"])

    trainer = Trainer(options)
    trainer.validate()