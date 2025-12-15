import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample
import datetime  # 新增：用于生成时间戳文件夹名
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import torch_optimizer as optim
from cgcnn.data import CIFData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--dropout', default=0.2, type=float,
                    help='Dropout probability (default: 0.2)')
parser.add_argument('--feature-file', type=str, required=True,
                    help='Path to the file containing 5-ring and HOMO-LUMO data')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                               'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                         help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                         help='percentage of validation data to be loaded (default '
                              '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                        help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='OPTIM',
                    choices=['SGD', 'Adam', 'AdamW', 'LAMB'],
                    help='choose an optimizer: SGD, Adam, AdamW, or LAMB (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')
# 新增参数：手动输入内容
parser.add_argument('--manual-notes', type=str, default='',
                    help='Additional manual notes to save in parameters file')


def get_predictions(data_loader, model, normalizer):
    """获取数据集的预测结果"""
    model.eval()
    targets = []
    preds = []
    cif_ids = []

    for i, (input, target, batch_cif_ids) in enumerate(data_loader):
        # 拆分结构特征和额外特征
        struct_input = input[:4]  # 前4项为结构特征
        extra_fea = input[4]  # 第5项为额外特征

        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(struct_input[0].cuda(non_blocking=True)),
                             Variable(struct_input[1].cuda(non_blocking=True)),
                             struct_input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in struct_input[3]],
                             Variable(extra_fea.cuda(non_blocking=True))  # 添加额外特征
                             )
        else:
            with torch.no_grad():
                input_var = (Variable(struct_input[0]),
                             Variable(struct_input[1]),
                             struct_input[2],
                             struct_input[3],
                             Variable(extra_fea)  # 添加额外特征
                             )

        # compute output
        output = model(*input_var)

        if args.task == 'regression':
            pred = normalizer.denorm(output.data.cpu())
            target_val = target
        else:
            pred = torch.exp(output.data.cpu())
            target_val = target.view(-1).long()

        preds += pred.view(-1).tolist()
        targets += target_val.view(-1).tolist()
        cif_ids += batch_cif_ids

    return targets, preds, cif_ids


def save_parameters_to_file(manual_notes=""):
    """保存所有参数到文件"""
    params_file = os.path.join(subdirs["train_data"], "training_parameters.txt")

    with open(params_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Crystal Graph Convolutional Neural Networks - Training Parameters\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {main_output_dir}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("MODEL PARAMETERS\n")
        f.write("=" * 50 + "\n")

        # 模型参数
        f.write(f"Task: {args.task}\n")
        f.write(f"Atom feature length: {args.atom_fea_len}\n")
        f.write(f"Hidden feature length: {args.h_fea_len}\n")
        f.write(f"Number of conv layers: {args.n_conv}\n")
        f.write(f"Number of hidden layers: {args.n_h}\n")
        f.write(f"Dropout probability: {args.dropout}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("TRAINING PARAMETERS\n")
        f.write("=" * 50 + "\n")

        # 训练参数
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Optimizer: {args.optim}\n")
        f.write(f"LR milestones: {args.lr_milestones}\n")
        f.write(f"Momentum: {args.momentum}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("DATA PARAMETERS\n")
        f.write("=" * 50 + "\n")

        # 数据参数
        f.write(f"Data path: {args.data_options[0]}\n")
        f.write(f"Feature file: {args.feature_file}\n")
        f.write(f"Train ratio: {args.train_ratio}\n")
        f.write(f"Validation ratio: {args.val_ratio}\n")
        f.write(f"Test ratio: {args.test_ratio}\n")
        f.write(f"Number of workers: {args.workers}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("HARDWARE SETTINGS\n")
        f.write("=" * 50 + "\n")

        # 硬件设置
        f.write(f"CUDA enabled: {args.cuda}\n")
        if args.cuda:
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")

        # 手动输入的备注
        if manual_notes:
            f.write("\n" + "=" * 50 + "\n")
            f.write("MANUAL NOTES\n")
            f.write("=" * 50 + "\n")
            f.write(f"{manual_notes}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF PARAMETERS\n")
        f.write("=" * 80 + "\n")

    print(f"Parameters saved to: {params_file}")
    return params_file


def get_manual_input():
    """获取用户手动输入的额外内容"""
    print("\n" + "=" * 60)
    print("MANUAL PARAMETER INPUT")
    print("=" * 60)
    print("You can now add additional notes or parameters.")
    print("Press Enter twice to finish input.")
    print("=" * 60)

    lines = []
    while True:
        try:
            line = input()
            if line == "" and len(lines) > 0 and lines[-1] == "":
                break
            lines.append(line)
        except EOFError:
            break

    # 移除最后的空行
    if lines and lines[-1] == "":
        lines = lines[:-1]

    return "\n".join(lines)


def plot_mae_curve(train_maes, val_maes):
    """
    绘制训练集和验证集的MAE曲线
    train_maes: 每个epoch的训练集MAE列表
    val_maes: 每个epoch的验证集MAE列表
    """
    # 保存MAE曲线原始数据到CSV
    mae_data = pd.DataFrame({
        'epoch': range(1, len(train_maes) + 1),
        'train_mae': train_maes,
        'val_mae': val_maes
    })
    mae_data.to_csv(os.path.join(subdirs["train_data"], 'mae_curve_data.csv'), index=False)
    print("MAE curve data saved")

    # 绘制MAE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_maes, label='Training MAE', color='red', linestyle='-', linewidth=2)
    plt.plot(val_maes, label='Validation MAE', color='blue', linestyle='--', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('MAE (eV)', fontsize=14)
    plt.title('Training and Validation MAE Curve', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(subdirs["figure"], 'mae_curve.png'), dpi=300)
    plt.close()
    print("MAE curve saved")


def main():
    global best_mae_error, subdirs, main_output_dir

    # 创建时间戳文件夹结构
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    main_output_dir = os.path.join(os.getcwd(), current_time)
    subdirs = {
        "checkpoint": os.path.join(main_output_dir, "checkpoint"),
        "csv": os.path.join(main_output_dir, "csv"),
        "figure": os.path.join(main_output_dir, "figure"),
        "train_data": os.path.join(main_output_dir, "train_data")
    }
    for dir_path in subdirs.values():
        os.makedirs(dir_path, exist_ok=True)
    print(f"所有输出文件将保存到: {main_output_dir}")

    # 获取手动输入的额外内容
    manual_notes = ""
    if args.manual_notes:
        manual_notes = args.manual_notes
    else:
        # 如果没有通过命令行参数提供，则交互式获取
        try:
            manual_notes = get_manual_input()
        except Exception as e:
            print(f"手动输入失败: {e}，将继续使用空备注")

    # 保存参数到文件
    params_file = save_parameters_to_file(manual_notes)

    if args.task == 'regression':
        best_mae_error = 1e10
    else:
        best_mae_error = 0.
    # load data
    dataset = CIFData(
        *args.data_options,
        feature_file=args.feature_file,
        max_num_nbr=12,
        radius=8
    )
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)

    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    n_extra_features = len(dataset.feature_names) if hasattr(dataset, 'feature_names') else 0
    print(f"额外特征数量: {n_extra_features}")

    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,
                                n_h=args.n_h,
                                classification=True if args.task == 'classification' else False,
                                dropout_prob=args.dropout,
                                n_extra_features=n_extra_features
                                )
    if args.cuda:
        model.cuda()

    print("Model device:", next(model.parameters()).device)

    # define loss func and optimizer
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.SmoothL1Loss()
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                      weight_decay=args.weight_decay)
    elif args.optim == 'LAMB':
        optimizer = optim.Lamb(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD, Adam, AdamW, or LAMB is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)

    # 添加记录训练和验证损失值的列表
    train_losses = []
    val_losses = []
    # 新增：记录训练和验证MAE的列表
    train_maes = []
    val_maes = []

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_loss, train_mae = train(train_loader, model, criterion, optimizer, epoch, normalizer)
        train_losses.append(train_loss)
        train_maes.append(train_mae)

        # 验证一个 epoch
        val_mae, val_targets, val_preds, val_cif_ids = validate(val_loader, model, criterion, normalizer)
        print('Validation MAE: {:.3f}'.format(val_mae))
        val_losses.append(val_mae)
        val_maes.append(val_mae)

        if val_mae != val_mae:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # 保存最佳模型
        if args.task == 'regression':
            is_best = val_mae < best_mae_error
            best_mae_error = min(val_mae, best_mae_error)
        else:
            is_best = val_mae > best_mae_error
            best_mae_error = max(val_mae, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # 绘制损失值变化曲线图
    plot_loss_curve(train_losses, val_losses)

    # 绘制MAE曲线图
    plot_mae_curve(train_maes, val_maes)

    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load(os.path.join(subdirs["checkpoint"], 'model_best.pth.tar'))
    model.load_state_dict(best_checkpoint['state_dict'])

    # 获取测试结果，包括cif_ids
    test_mae, test_targets, test_preds, test_cif_ids = validate(val_loader, model, criterion, normalizer, test=True)
    print('Test MAE: {:.3f}'.format(test_mae))

    # 获取训练集预测结果
    print('---------Getting Training Set Predictions---------------')
    train_targets, train_preds, train_cif_ids = get_predictions(train_loader, model, normalizer)

    # 获取验证集预测结果（使用最佳模型）
    print('---------Getting Validation Set Predictions---------------')
    val_targets, val_preds, val_cif_ids = get_predictions(val_loader, model, normalizer)

    # 创建包含所有数据集的DataFrame
    train_df = pd.DataFrame({
        'material_id': train_cif_ids,
        'true_value': train_targets,
        'predicted_value': train_preds,
        'dataset': 'train'
    })

    val_df = pd.DataFrame({
        'material_id': val_cif_ids,
        'true_value': val_targets,
        'predicted_value': val_preds,
        'dataset': 'validation'
    })

    test_df = pd.DataFrame({
        'material_id': test_cif_ids,
        'true_value': test_targets,
        'predicted_value': test_preds,
        'dataset': 'test'
    })

    # 合并所有数据集
    all_results_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # 保存到CSV文件
    all_results_df.to_csv(os.path.join(subdirs["csv"], 'all_predictions.csv'), index=False)
    print("All predictions saved to 'all_predictions.csv'")

    # 分别保存每个数据集
    train_df.to_csv(os.path.join(subdirs["csv"], 'train_predictions.csv'), index=False)
    val_df.to_csv(os.path.join(subdirs["csv"], 'validation_predictions.csv'), index=False)
    test_df.to_csv(os.path.join(subdirs["csv"], 'test_predictions.csv'), index=False)
    print("Individual dataset predictions also saved separately")

    # 绘制验证集散点图
    # 绘制验证集散点图（修改文件名+标签）
    plt.figure(figsize=(8, 6))
    plt.scatter(val_targets, val_preds, alpha=0.5, color='blue', label='Validation Predictions')
    plt.plot([min(val_targets), max(val_targets)],
             [min(val_targets), max(val_targets)], 'k--', lw=2, label='Ideal Prediction')
    plt.xlabel('Calculated Band Gap (eV)', fontsize=18)  # 改为带隙+单位电子伏特
    plt.ylabel('Predicted Band Gap (eV)', fontsize=18)  # 改为带隙+单位电子伏特
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(subdirs["figure"], 'validation_bandgap_scatter.png'), dpi=300)  # 文件名含带隙
    plt.close()
    print("Validation set band gap scatter plot saved")

    # 保存验证集散点图原始数据（文件名同步修改）
    val_scatter_data = pd.DataFrame({
        'true_value': val_targets,
        'predicted_value': val_preds,
        'material_id': val_cif_ids
    })
    val_scatter_data.to_csv(os.path.join(subdirs["csv"], 'validation_bandgap_scatter_data.csv'), index=False)
    print("Validation band gap scatter plot data saved")

    # 绘制测试集散点图（修改文件名+标签）
    plt.figure(figsize=(8, 6))
    plt.scatter(test_targets, test_preds, alpha=0.5)
    plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'k--', lw=2)
    plt.xlabel('Calculated Band Gap (eV)', fontsize=18)  # 改为带隙+单位电子伏特
    plt.ylabel('Predicted Band Gap (eV)', fontsize=18)  # 改为带隙+单位电子伏特
    plt.title('Test Set Band Gap: Predictions vs True Values', fontsize=16)  # 标题含带隙
    plt.grid(True)
    plt.savefig(os.path.join(subdirs["figure"], 'test_bandgap_scatter.png'), dpi=300)  # 文件名含带隙
    plt.close()
    print("Test set band gap scatter plot saved")

    # 保存测试集散点图原始数据（文件名同步修改）
    test_scatter_data = pd.DataFrame({
        'true_value': test_targets,
        'predicted_value': test_preds,
        'material_id': test_cif_ids
    })
    test_scatter_data.to_csv(os.path.join(subdirs["csv"], 'test_bandgap_scatter_data.csv'), index=False)
    print("Test band gap scatter plot data saved")


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    # 创建进度条
    pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}',
                unit='batch', ncols=100, position=0, leave=True)

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # 拆分结构特征和额外特征
        struct_input = input[:4]
        extra_fea = input[4]

        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(struct_input[0].cuda(non_blocking=True)),
                         Variable(struct_input[1].cuda(non_blocking=True)),
                         struct_input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in struct_input[3]],
                         Variable(extra_fea.cuda(non_blocking=True))  # 添加额外特征
                         )
        else:
            input_var = (Variable(struct_input[0]),
                         Variable(struct_input[1]),
                         struct_input[2],
                         struct_input[3],
                         Variable(extra_fea)  # 添加额外特征
                         )

        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # 更新进度条描述
        if args.task == 'regression':
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'MAE': f'{mae_errors.avg:.3f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        else:
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accuracies.avg:.3f}',
                'F1': f'{fscores.avg:.3f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        # 只在print_freq间隔时打印详细信息
        if i % args.print_freq == 0:
            if args.task == 'regression':
                tqdm.write(f'Epoch {epoch} - Batch {i}/{len(train_loader)} - '
                           f'Loss: {losses.avg:.4f} - MAE: {mae_errors.avg:.3f} - '
                           f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            else:
                tqdm.write(f'Epoch {epoch} - Batch {i}/{len(train_loader)} - '
                           f'Loss: {losses.avg:.4f} - Acc: {accuracies.avg:.3f} - '
                           f'F1: {fscores.avg:.3f} - LR: {optimizer.param_groups[0]["lr"]:.6f}')

        pbar.update(1)  # 更新进度条

    pbar.close()  # 关闭进度条

    if args.task == 'regression':
        return losses.avg, mae_errors.avg
    else:
        return losses.avg, 0.0


def validate(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # 初始化存储验证集/测试集的预测值和真实值
    val_targets = []
    val_preds = []
    val_cif_ids = []

    # switch to evaluate mode
    model.eval()

    # 创建验证进度条
    desc = 'Testing' if test else 'Validating'
    pbar = tqdm(total=len(val_loader), desc=desc, unit='batch', ncols=100, position=0, leave=True)

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        # 拆分结构特征和额外特征
        struct_input = input[:4]
        extra_fea = input[4]

        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(struct_input[0].cuda(non_blocking=True)),
                             Variable(struct_input[1].cuda(non_blocking=True)),
                             struct_input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in struct_input[3]],
                             Variable(extra_fea.cuda(non_blocking=True))  # 添加额外特征
                             )
        else:
            with torch.no_grad():
                input_var = (Variable(struct_input[0]),
                             Variable(struct_input[1]),
                             struct_input[2],
                             struct_input[3],
                             Variable(extra_fea)  # 添加额外特征
                             )

        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            # 记录验证集的预测值和真实值
            val_pred = normalizer.denorm(output.data.cpu())
            val_target = target
            val_preds += val_pred.view(-1).tolist()
            val_targets += val_target.view(-1).tolist()
            val_cif_ids += batch_cif_ids

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'MAE': f'{mae_errors.avg:.3f}'
            })
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accuracies.avg:.3f}'
            })

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        pbar.update(1)

    pbar.close()

    if args.task == 'regression':
        print(' * Validation MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return mae_errors.avg, val_targets, val_preds, val_cif_ids
    else:
        print(' * Validation AUC {auc.avg:.3f}'.format(auc=auc_scores))
        return auc_scores.avg, val_targets, val_preds, val_cif_ids


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best):
    checkpoint_path = os.path.join(subdirs["checkpoint"], "checkpoint.pth.tar")
    torch.save(state, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(subdirs["checkpoint"], "model_best.pth.tar")
        shutil.copyfile(checkpoint_path, best_model_path)


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def plot_loss_curve(train_losses, val_losses):
    # 保存损失曲线原始数据到 CSV
    loss_data = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_data.to_csv(os.path.join(subdirs["train_data"], 'loss_curve_data.csv'), index=False)
    print("Loss curve data saved")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='red', linestyle='-')
    plt.plot(val_losses, label='Validation Loss', color='blue', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(subdirs["figure"], 'loss_curve.png'))
    plt.close()
    print("Loss curve saved")


if __name__ == '__main__':
    # 设置默认参数
    class Args:
        def __init__(self):
            self.data_options = ['D:\\modle-new\\数据清洗\\ABO3-DATA1']  # 修改为你的数据路径
            self.dropout = 0.2
            self.task = 'regression'
            self.disable_cuda = False
            self.workers = 0
            self.epochs = 100
            self.start_epoch = 0
            self.batch_size = 64
            self.lr = 0.001
            self.lr_milestones = [25, 50, 75]
            self.momentum = 0.9
            self.weight_decay = 0
            self.print_freq = 10
            self.resume = ''
            self.train_ratio = 0.7
            self.train_size = None
            self.val_ratio = 0.15
            self.val_size = None
            self.test_ratio = 0.15
            self.test_size = None
            self.optim = 'AdamW'
            self.atom_fea_len = 128
            self.h_fea_len = 256
            self.n_conv = 3
            self.n_h = 3
            self.cuda = not self.disable_cuda and torch.cuda.is_available()
            self.feature_file = 'D:\\modle-new\\数据清洗\\ABO3-DATA1\\features.csv'  # 添加默认特征文件路径
            self.manual_notes = ''  # 新增：手动备注参数


    # 创建参数对象并运行主函数
    args = Args()
    main()