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

        self.validation_folder = 'MVI_3015.MP4'
        self.loader_getter = LoaderGetter(options, specific_folders={'val': self.validation_folder})
        self.img_saver = ImgSaver()
        self.step = int(self.epoch*self.loader_getter.get_size('train')/options['training']['batch_size'])

    def validate(self):
        self.model.eval()
        self.evaluator.reset()
        evaluator2 = Metrics(options['data']['num_classes'])
        evaluator2.reset()
        post = EM(options['data']['root'], self.validation_folder)
        val_loader = self.loader_getter('val')
        loss = []
        #out_list = []
        #av_length = 5
        with torch.no_grad():
            for b,batch in tqdm(enumerate(val_loader)):
                if len(batch['image']) != self.options["training"]["batch_size"]:
                    break
                if b < 44:
                    continue
                image = batch['image'].to(self.device)
                gt = batch['ground_truth'].to(self.device)

                out = self.model(image)
                loss.append(self.criterion(out, gt).item())

                pred = out.data.cpu().numpy()
                array_saver = ArraySaver()
                array_saver(pred, batch['name'][0] + '.npy', folder=os.path.join('segmentation_probs', self.validation_folder))

                gt = gt.cpu().numpy()
                pred0 = np.argmax(pred, axis=1)
                self.evaluator.add_batch(gt, pred0)
                pred0 = decode_seg_map_sequence(pred0)
                # decoded_gt = decode_seg_map_sequence(gt)
                self.img_saver(pred0[0], batch['name'][0], folder=os.path.join('segmentation_raw', self.validation_folder))

                start_time = time.time()
                polygons,face_labels = post.EM(pred[0])
                pred_post = post.get_segmentation_output(polygons, face_labels)
                pred_post = np.expand_dims(pred_post, axis=0)
                evaluator2.add_batch(gt, pred_post)
                evaluator2.add_time(time.time() - start_time)

                #or i,o in enumerate(out):
                pred_post = decode_seg_map_sequence(pred_post)

                #for i,p in enumerate(pred1):
                #     #nb = b*options['training']['batch_size'] + i
                self.img_saver(pred_post[0], batch['name'][0], folder=os.path.join('segmentation_post', self.validation_folder))
                array_saver(polygons, batch['name'][0] + '.npy', folder=os.path.join('polygons', self.validation_folder), array1=face_labels)

                # if b == 50:
                #     break

                # out_list.append(out)
                # #print(batch['name'])
                # if len(out_list) > av_length:
                #     out_list.pop(0)
                #     #av_out = torch.mean(torch.stack(out_list),0)
                #     av_out = (3*out_list[2] + 2*(out_list[1] + out_list[3]) + (out_list[0] + out_list[4]))/14
                #     av_pred = av_out.data.cpu().numpy()
                #     av_pred = np.argmax(av_pred, axis=1)
                #     av_pred = decode_seg_map_sequence(av_pred)
                #     self.img_saver(av_pred[0], str(b-av_length+1) + '.png', img2=pred[0], folder='results_MVI_3027.MP4')
        #writer.add_scalar('validation return', returns_average, step)


                accuracy = self.evaluator.Pixel_Accuracy()
                accuracy2 = evaluator2.Pixel_Accuracy()
                self.best_acc = max(accuracy, self.best_acc)
                print_string = 'VALIDATION: Step {} Loss {:.2f} Accuracy {:.2f}%'.format(self.step,
                                sum(loss)/len(loss), accuracy*100)
                print(print_string)
                avg_time = evaluator2.get_average_time()
                print("other accuracy", accuracy2*100, "Time", avg_time)
        #save_string(print_string, self.logdir)
        self.evaluator.reset()

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