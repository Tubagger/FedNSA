#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help="random seed")
    parser.add_argument('--save_dir', type=str, default='../saved_mia_models',
                        help='saving path')
    parser.add_argument('--log_folder_name', type=str, default='/training_log_correct_iid/',
                        help='saving path')
    # federated arguments
    parser.add_argument('--algorithm', type=str, default="DP-SGD", help="name of algorithm")
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--local_ep', type=int, default=1, help="rounds of local training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--local_bs', type=int, default=10000000000000, help="test batch size")
    parser.add_argument('--optim', type=str, default="SGD", help="name of algorithm")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=1, help="learning rate decay each round")
    parser.add_argument('--lr_up', type=str, default='common',
                        help='optimizer: [common, milestone, cosine]')
    parser.add_argument('--alpha', type=float, default=0, help="parameter of Dirichlet distribution")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    parser.add_argument('--dp_mechanism', type=str, default='Gaussian',
                        help='differential privacy mechanism')
    parser.add_argument('--dp_epsilon', type=float, default=20,
                        help='differential privacy epsilon')
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                        help='differential privacy delta')
    parser.add_argument('--dp_clip', type=float, default=15,
                        help='differential privacy clip')
    parser.add_argument('--dp_sample', type=float, default=1, help='sample rate for moment account')
    parser.add_argument('--sampling_type', choices=['poisson', 'uniform'],
                         default='uniform', type=str,
                         help='which kind of client sampling we use') 

    parser.add_argument('--k', type=float, default=0.0, help='t for collude clients ratio')
    parser.add_argument('--d', type=float, default=0.0, help='d for dropout clients ratio')

    parser.add_argument('--acc', type=float, default=70.0, help='acc for account')
    parser.add_argument('--account', action='store_true', help='whether account overhead or not')
    parser.add_argument('--account1', action='store_true', help='whether account one round time or not need account')
    parser.add_argument('--account2', action='store_true', help='whether account network latency time or not')

    parser.add_argument('--serial', action='store_true', help='partial serial running to save the gpu memory')
    parser.add_argument('--serial_bs', type=int, default=128, help='partial serial running batch size')

    parser.add_argument('--debug_mode', action='store_true', help='print debug information')

    parser.add_argument('--data_augment', type=int,  default =0,
                        help='data_augment')

    parser.add_argument('--schedule_milestone', type=list, default=[],
                         help="schedule lr")

    args = parser.parse_args()
    return args
