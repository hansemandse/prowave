import os
import torch
from torch import nn, optim

from wavenet.audiodata import AudioData, AudioLoader
from wavenet.models import Model, Generator
from wavenet.utils import list_files

from argparse import ArgumentParser

def set_args():
    parser = ArgumentParser(description='Wavenet demo')
    parser.add_argument('--data', type=str, default='./audio', help='folder to training set of .wav files')
    parser.add_argument('--x_len', type=int, default=2**15, help='length of input')
    parser.add_argument('--num_classes', type=int, default=256, help='number of discrete output levels')
    parser.add_argument('--num_layers', type=int, default=13, help='number of convolutional layers per block')
    parser.add_argument('--num_blocks', type=int, default=2, help='number of repeating convolutional layer blocks')
    parser.add_argument('--num_hidden', type=int, default=32, help='number of neurons per layer')
    parser.add_argument('--kernel_size', type=int, default=2, help='width of convolutional kernel')
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--step_size', type=int, default=50, help='step size of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of learning rate scheduler')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--num_epochs', type=int, default=25, help='number of training epochs')
    parser.add_argument('--disp_interval', type=int, default=10, help='number of epochs in between display messages')
    parser.add_argument('--model_file', type=str, default='model.pt', help='filename of model')
    parser.add_argument('--visdom', type=bool, default=False, help='flag to track variables in visdom')
    parser.add_argument('--new_seq_len', type=int, default=1000, help='length of sequence to predict')
    parser.add_argument('--device', type=str, default='default', help='device to use')
    parser.add_argument('--resume_train', type=bool, default=False, help='whether to resume training existing models')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = set_args()

    # construct model
    wave_model = Model(args.x_len, num_channels=1, num_classes=args.num_classes, 
                       num_blocks=args.num_blocks, num_layers=args.num_layers,
                       num_hidden=args.num_hidden, kernel_size=args.kernel_size)

    if not (args.device == 'default'):
        wave_model.set_device(torch.device(args.device))

    # create dataset and dataloader
    filelist = list_files(args.data)
    dataset = AudioData(filelist, args.x_len, y_len=wave_model.output_width - 1,
                        num_classes=args.num_classes, store_tracks=True)

    # manage small datasets with big batch sizes
    if args.batch_size > len(dataset):
        args.batch_size = len(dataset)
    
    dataloader = AudioLoader(dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers)

    # load/train model
    if os.path.isfile(args.model_file):
        print('Loading model data from file: {}'.format(args.model_file))
        wave_model.load_state_dict(torch.load(args.model_file))

        if args.resume_train:
            print('Resuming training.')
    else:
        print('Model data not found: {}'.format(args.model_file))
        print('Training new model.')
        args.resume_train = True

    if args.resume_train:
        wave_model.criterion = nn.CrossEntropyLoss()
        wave_model.optimizer = optim.Adam(wave_model.parameters(), 
                                          lr=args.learn_rate)
        wave_model.scheduler = optim.lr_scheduler.StepLR(wave_model.optimizer, 
                                                         step_size=args.step_size, 
                                                         gamma=args.gamma)

        try:
            wave_model.train(dataloader, num_epochs=args.num_epochs, 
                             disp_interval=args.disp_interval, use_visdom=args.visdom)
        except KeyboardInterrupt:
            print('Training stopped!')

        print('Saving model data to file: {}'.format(args.model_file))
        torch.save(wave_model.state_dict(), args.model_file)

    # predict sequence with model
    wave_generator = Generator(wave_model, dataset)
    y = wave_generator.run(dataset.tracks[0]['audio'][:args.x_len], 
                           args.new_seq_len, disp_interval=100)
    dataset.save_wav('./tmp.wav', y, dataloader.dataset.sample_rate)
