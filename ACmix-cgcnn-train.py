import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample
import datetime
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


# ==================== 自适应加权损失函数类 ====================
class AdaptiveWeightedLoss(nn.Module):
    """自适应加权损失函数，基于图复杂度计算样本权重"""

    def __init__(self, base_loss_fn=None, weight_range=(0.8, 1.5)):
        super(AdaptiveWeightedLoss, self).__init__()
        self.base_loss_fn = base_loss_fn or nn.SmoothL1Loss(reduction='none')
        self.weight_range = weight_range
        self.last_weights_mean = 0.0
        self.last_complexity_mean = 0.0
        self.last_fusion_weights = None

    def forward(self, predictions, targets, complexities=None, adaptive_weights=None):
        if hasattr(self.base_loss_fn, 'reduction'):
            original_reduction = getattr(self.base_loss_fn, 'reduction', 'mean')
            self.base_loss_fn.reduction = 'none'
            base_loss = self.base_loss_fn(predictions, targets)
            self.base_loss_fn.reduction = original_reduction
        else:
            base_loss = self.base_loss_fn(predictions, targets)

        if adaptive_weights is not None and len(adaptive_weights) == len(base_loss):
            weights = adaptive_weights
            self.last_weights_mean = weights.mean().item()
            self.last_complexity_mean = complexities.mean().item() if complexities is not None else 0.0
        elif complexities is not None and len(complexities) == len(base_loss):
            weight_min, weight_max = self.weight_range
            weights = weight_min + (weight_max - weight_min) * complexities
            self.last_weights_mean = weights.mean().item()
            self.last_complexity_mean = complexities.mean().item()
        else:
            loss = base_loss.mean() if base_loss.dim() > 0 else base_loss
            return loss

        if weights.dim() == 1 and base_loss.dim() > 1:
            weights = weights.unsqueeze(1)

        loss = (base_loss * weights).mean()
        return loss


# ==================== 自适应加权损失函数结束 ====================


parser = argparse.ArgumentParser(
    description='Crystal Graph Convolutional Neural Networks with MC Dropout and Adaptive Weighted Loss')
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
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: [100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# ==================== 加权损失相关命令行参数 ====================
parser.add_argument('--use-weighted-loss', action='store_true',
                    help='Use weighted loss based on graph complexity')
parser.add_argument('--use-adaptive-weights', action='store_true',
                    help='Use adaptive weights (learnable fusion parameters)')
parser.add_argument('--weight-min', default=0.8, type=float,
                    help='Minimum weight for weighted loss (default: 0.8)')
parser.add_argument('--weight-max', default=1.5, type=float,
                    help='Maximum weight for weighted loss (default: 1.5)')
# ==================== 加权损失相关参数结束 ====================


# ==================== MC Dropout相关参数 ====================
parser.add_argument('--mc-dropout-samples', default=50, type=int,
                    help='Number of MC Dropout forward passes for uncertainty estimation (default: 50)')
parser.add_argument('--confidence-interval', default=95, type=float,
                    help='Confidence interval percentage (default: 95)')
# ==================== MC Dropout相关参数结束 ====================


# ==================== 输出目录参数 ====================
parser.add_argument('--output-dir', default='D:\\modle-new\\modle-new\\CA-CGCNN\\时间戳文件', type=str, metavar='PATH',
                    help='Custom output directory for timestamp folders (default: D:\\modle-new\\modle-new\\CA-CGCNN\\时间戳文件)')
# ==================== 输出目录参数结束 ====================


train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                         help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                         help='percentage of validation data to be loaded (default 0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default 1000)')
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
parser.add_argument('--manual-notes', type=str, default='',
                    help='Additional manual notes to save in parameters file')


def get_predictions_with_uncertainty(data_loader, model, normalizer, mc_samples=50, confidence=95):
    """获取数据集的预测结果及不确定性估计，使用MC Dropout进行多次前向传播"""
    model.train()
    all_targets = []
    all_preds_mean = []
    all_preds_std = []
    all_cif_ids = []

    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence / 100) / 2)

    for i, (input, target, batch_cif_ids) in enumerate(tqdm(data_loader, desc='MC Dropout Inference')):
        struct_input = input[:4]
        extra_fea = input[4]

        batch_size = target.size(0)
        mc_predictions = []

        for _ in range(mc_samples):
            if args.cuda:
                with torch.no_grad():
                    input_var = (Variable(struct_input[0].cuda(non_blocking=True)),
                                 Variable(struct_input[1].cuda(non_blocking=True)),
                                 struct_input[2].cuda(non_blocking=True),
                                 [crys_idx.cuda(non_blocking=True) for crys_idx in struct_input[3]],
                                 Variable(extra_fea.cuda(non_blocking=True)))
            else:
                with torch.no_grad():
                    input_var = (Variable(struct_input[0]),
                                 Variable(struct_input[1]),
                                 struct_input[2],
                                 struct_input[3],
                                 Variable(extra_fea))

            output = model(*input_var, return_complexity=False)

            if args.task == 'regression':
                pred = normalizer.denorm(output.data.cpu())
            else:
                pred = torch.exp(output.data.cpu())

            mc_predictions.append(pred.view(-1).numpy())

        mc_predictions = np.array(mc_predictions)
        pred_mean = np.mean(mc_predictions, axis=0)
        pred_std = np.std(mc_predictions, axis=0)
        pred_ci = z_score * pred_std

        all_preds_mean.extend(pred_mean.tolist())
        all_preds_std.extend(pred_std.tolist())
        all_targets.extend(target.view(-1).tolist())
        all_cif_ids.extend(batch_cif_ids)

        if i % 10 == 0:
            print(f'  Batch {i}: Mean uncertainty = {np.mean(pred_std):.4f} eV')

    return all_targets, all_preds_mean, all_preds_std, all_cif_ids


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

        f.write(f"Task: {args.task}\n")
        f.write(f"Atom feature length: {args.atom_fea_len}\n")
        f.write(f"Hidden feature length: {args.h_fea_len}\n")
        f.write(f"Number of conv layers: {args.n_conv}\n")
        f.write(f"Number of hidden layers: {args.n_h}\n")
        f.write(f"Dropout probability: {args.dropout}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("MC DROPOUT UNCERTAINTY PARAMETERS\n")
        f.write("=" * 50 + "\n")
        f.write(f"MC Dropout samples: {args.mc_dropout_samples}\n")
        f.write(f"Confidence interval: {args.confidence_interval}%\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("WEIGHTED LOSS PARAMETERS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Use weighted loss: {args.use_weighted_loss}\n")
        if args.use_weighted_loss:
            f.write(f"Use adaptive weights: {args.use_adaptive_weights}\n")
            if args.use_adaptive_weights:
                f.write("Using learnable adaptive weight fusion\n")
            else:
                f.write(f"Weight range: [{args.weight_min}, {args.weight_max}]\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("TRAINING PARAMETERS\n")
        f.write("=" * 50 + "\n")

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

        f.write(f"Data path: {args.data_options[0]}\n")
        f.write(f"Feature file: {args.feature_file}\n")
        f.write(f"Train ratio: {args.train_ratio}\n")
        f.write(f"Validation ratio: {args.val_ratio}\n")
        f.write(f"Test ratio: {args.test_ratio}\n")
        f.write(f"Number of workers: {args.workers}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("HARDWARE SETTINGS\n")
        f.write("=" * 50 + "\n")

        f.write(f"CUDA enabled: {args.cuda}\n")
        if args.cuda:
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")

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

    if lines and lines[-1] == "":
        lines = lines[:-1]

    return "\n".join(lines)


def plot_mae_curve(train_maes, val_maes):
    """绘制训练集和验证集的MAE曲线"""
    mae_data = pd.DataFrame({
        'epoch': range(1, len(train_maes) + 1),
        'train_mae': train_maes,
        'val_mae': val_maes
    })
    mae_data.to_csv(os.path.join(subdirs["train_data"], 'mae_curve_data.csv'), index=False)
    print("MAE curve data saved")

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


def plot_predictions_with_uncertainty(targets, predictions, uncertainties, cif_ids, dataset_name):
    """绘制带有不确定性估计的预测散点图"""
    targets = np.array(targets)
    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)

    z_score = 1.96
    ci_half = z_score * uncertainties

    mean_uncertainty = np.mean(uncertainties)
    median_uncertainty = np.median(uncertainties)
    p90_uncertainty = np.percentile(uncertainties, 90)

    print(f"\n{dataset_name} Set Uncertainty Statistics:")
    print(f"  Mean uncertainty: {mean_uncertainty:.4f} eV")
    print(f"  Median uncertainty: {median_uncertainty:.4f} eV")
    print(f"  90th percentile uncertainty: {p90_uncertainty:.4f} eV")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]
    n_show = min(100, len(targets))
    indices = np.random.choice(len(targets), n_show, replace=False)

    ax1.errorbar(targets[indices], predictions[indices],
                 yerr=ci_half[indices], fmt='o', alpha=0.6,
                 capsize=3, elinewidth=1, markersize=4)

    min_val = min(min(targets), min(predictions)) - 0.1
    max_val = max(max(targets), max(predictions)) + 0.1
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal Prediction')

    ax1.set_xlabel('Calculated Band Gap (eV)', fontsize=12)
    ax1.set_ylabel('Predicted Band Gap (eV)', fontsize=12)
    ax1.set_title(f'{dataset_name} Set: Predictions with 95% CI', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # 在图下方添加标签
    ax1.text(0.5, -0.15, '(a) Scatter plot with confidence intervals',
             transform=ax1.transAxes, fontsize=12, ha='center')

    ax2 = axes[0, 1]
    scatter = ax2.scatter(targets, predictions, c=uncertainties,
                          cmap='viridis', alpha=0.7, s=20)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    ax2.set_xlabel('Calculated Band Gap (eV)', fontsize=12)
    ax2.set_ylabel('Predicted Band Gap (eV)', fontsize=12)
    ax2.set_title(f'{dataset_name} Set: Uncertainty Color Map', fontsize=14)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Uncertainty (eV)')
    # 在图下方添加标签
    ax2.text(0.5, -0.15, '(b) Uncertainty color map',
             transform=ax2.transAxes, fontsize=12, ha='center')

    ax3 = axes[1, 0]
    ax3.hist(uncertainties, bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(mean_uncertainty, color='red', linestyle='--',
                linewidth=2, label=f'Mean: {mean_uncertainty:.3f}')
    ax3.axvline(median_uncertainty, color='blue', linestyle='--',
                linewidth=2, label=f'Median: {median_uncertainty:.3f}')
    ax3.axvline(p90_uncertainty, color='green', linestyle='--',
                linewidth=2, label=f'90th: {p90_uncertainty:.3f}')
    ax3.set_xlabel('Uncertainty (eV)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title(f'{dataset_name} Set: Uncertainty Distribution', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # 在图下方添加标签
    ax3.text(0.5, -0.15, '(c) Uncertainty distribution histogram',
             transform=ax3.transAxes, fontsize=12, ha='center')

    ax4 = axes[1, 1]
    abs_errors = np.abs(targets - predictions)
    ax4.scatter(uncertainties, abs_errors, alpha=0.6, s=15)

    z = np.polyfit(uncertainties, abs_errors, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(uncertainties), max(uncertainties), 100)
    ax4.plot(x_trend, p(x_trend), 'r--', linewidth=2,
             label=f'Trend (slope: {z[0]:.3f})')

    ax4.set_xlabel('Uncertainty (eV)', fontsize=12)
    ax4.set_ylabel('Absolute Error (eV)', fontsize=12)
    ax4.set_title(f'{dataset_name} Set: Error vs Uncertainty', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    # 在图下方添加标签
    ax4.text(0.5, -0.15, '(d) Error vs uncertainty relationship',
             transform=ax4.transAxes, fontsize=12, ha='center')

    # 调整子图布局，为底部标签留出空间
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # 增加底部边距
    plt.savefig(os.path.join(subdirs["figure"], f'{dataset_name.lower()}_uncertainty_analysis.png'), dpi=300)
    plt.close()

    uncertainty_df = pd.DataFrame({
        'material_id': cif_ids,
        'true_value': targets,
        'predicted_value': predictions,
        'uncertainty': uncertainties,
        'ci_95_lower': predictions - ci_half,
        'ci_95_upper': predictions + ci_half,
        'absolute_error': abs_errors
    })
    uncertainty_df.to_csv(os.path.join(subdirs["csv"], f'{dataset_name.lower()}_uncertainty_data.csv'), index=False)

    print(f"Uncertainty analysis plots and data saved for {dataset_name} set")

def main():
    global best_mae_error, subdirs, main_output_dir

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")

    # 使用指定的输出目录
    # 确保基础目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    main_output_dir = os.path.join(args.output_dir, current_time)

    subdirs = {
        "checkpoint": os.path.join(main_output_dir, "checkpoint"),
        "csv": os.path.join(main_output_dir, "csv"),
        "figure": os.path.join(main_output_dir, "figure"),
        "train_data": os.path.join(main_output_dir, "train_data")
    }
    for dir_path in subdirs.values():
        os.makedirs(dir_path, exist_ok=True)
    print(f"所有输出文件将保存到: {main_output_dir}")

    if args.use_weighted_loss:
        print("=" * 60)
        print("使用基于图复杂度的加权损失函数")
        if args.use_adaptive_weights:
            print("使用自适应权重（可学习融合参数）")
        else:
            print(f"使用固定权重范围: [{args.weight_min}, {args.weight_max}]")
        print("=" * 60)

    manual_notes = ""
    if args.manual_notes:
        manual_notes = args.manual_notes
    else:
        try:
            manual_notes = get_manual_input()
        except Exception as e:
            print(f"手动输入失败: {e}，将继续使用空备注")

    save_parameters_to_file(manual_notes)

    if args.task == 'regression':
        best_mae_error = 1e10
    else:
        best_mae_error = 0.

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

    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        if args.use_weighted_loss:
            print("使用加权损失函数")
            base_criterion = nn.SmoothL1Loss()
            criterion = AdaptiveWeightedLoss(
                base_loss_fn=base_criterion,
                weight_range=(args.weight_min, args.weight_max)
            )
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

    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_mae = train(train_loader, model, criterion, optimizer, epoch, normalizer)
        train_losses.append(train_loss)
        train_maes.append(train_mae)

        val_mae, val_targets, val_preds, val_cif_ids = validate(val_loader, model, criterion, normalizer)
        print('Validation MAE: {:.3f}'.format(val_mae))
        val_losses.append(val_mae)
        val_maes.append(val_mae)

        if val_mae != val_mae:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

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

    plot_loss_curve(train_losses, val_losses)
    plot_mae_curve(train_maes, val_maes)

    print('\n---------Evaluate Model with MC Dropout Uncertainty Estimation---------------')
    best_checkpoint = torch.load(os.path.join(subdirs["checkpoint"], 'model_best.pth.tar'))
    model.load_state_dict(best_checkpoint['state_dict'])

    print('\n---------Getting Training Set Predictions with Uncertainty---------------')
    train_targets, train_preds, train_uncertainties, train_cif_ids = get_predictions_with_uncertainty(
        train_loader, model, normalizer,
        mc_samples=args.mc_dropout_samples,
        confidence=args.confidence_interval
    )

    print('\n---------Getting Validation Set Predictions with Uncertainty---------------')
    val_targets, val_preds, val_uncertainties, val_cif_ids = get_predictions_with_uncertainty(
        val_loader, model, normalizer,
        mc_samples=args.mc_dropout_samples,
        confidence=args.confidence_interval
    )

    print('\n---------Getting Test Set Predictions with Uncertainty---------------')
    test_targets, test_preds, test_uncertainties, test_cif_ids = get_predictions_with_uncertainty(
        test_loader, model, normalizer,
        mc_samples=args.mc_dropout_samples,
        confidence=args.confidence_interval
    )

    test_mae = np.mean(np.abs(np.array(test_targets) - np.array(test_preds)))
    print(f'\nTest MAE: {test_mae:.4f} eV')
    print(f'Test Mean Uncertainty: {np.mean(test_uncertainties):.4f} eV')

    train_df = pd.DataFrame({
        'material_id': train_cif_ids,
        'true_value': train_targets,
        'predicted_value': train_preds,
        'uncertainty': train_uncertainties,
        'ci_95_lower': np.array(train_preds) - 1.96 * np.array(train_uncertainties),
        'ci_95_upper': np.array(train_preds) + 1.96 * np.array(train_uncertainties),
        'dataset': 'train'
    })

    val_df = pd.DataFrame({
        'material_id': val_cif_ids,
        'true_value': val_targets,
        'predicted_value': val_preds,
        'uncertainty': val_uncertainties,
        'ci_95_lower': np.array(val_preds) - 1.96 * np.array(val_uncertainties),
        'ci_95_upper': np.array(val_preds) + 1.96 * np.array(val_uncertainties),
        'dataset': 'validation'
    })

    test_df = pd.DataFrame({
        'material_id': test_cif_ids,
        'true_value': test_targets,
        'predicted_value': test_preds,
        'uncertainty': test_uncertainties,
        'ci_95_lower': np.array(test_preds) - 1.96 * np.array(test_uncertainties),
        'ci_95_upper': np.array(test_preds) + 1.96 * np.array(test_uncertainties),
        'dataset': 'test'
    })

    all_results_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_results_df.to_csv(os.path.join(subdirs["csv"], 'all_predictions_with_uncertainty.csv'), index=False)

    train_df.to_csv(os.path.join(subdirs["csv"], 'train_predictions_with_uncertainty.csv'), index=False)
    val_df.to_csv(os.path.join(subdirs["csv"], 'validation_predictions_with_uncertainty.csv'), index=False)
    test_df.to_csv(os.path.join(subdirs["csv"], 'test_predictions_with_uncertainty.csv'), index=False)

    print('\n---------Generating Uncertainty Analysis Plots---------------')
    plot_predictions_with_uncertainty(val_targets, val_preds, val_uncertainties, val_cif_ids, 'Validation')
    plot_predictions_with_uncertainty(test_targets, test_preds, test_uncertainties, test_cif_ids, 'Test')

    summary_data = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'MAE (eV)': [
            np.mean(np.abs(np.array(train_targets) - np.array(train_preds))),
            np.mean(np.abs(np.array(val_targets) - np.array(val_preds))),
            np.mean(np.abs(np.array(test_targets) - np.array(test_preds)))
        ],
        'Mean Uncertainty (eV)': [
            np.mean(train_uncertainties),
            np.mean(val_uncertainties),
            np.mean(test_uncertainties)
        ],
        'Median Uncertainty (eV)': [
            np.median(train_uncertainties),
            np.median(val_uncertainties),
            np.median(test_uncertainties)
        ],
        '90th Percentile Uncertainty (eV)': [
            np.percentile(train_uncertainties, 90),
            np.percentile(val_uncertainties, 90),
            np.percentile(test_uncertainties, 90)
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(subdirs["csv"], 'uncertainty_summary.csv'), index=False)
    print("\n" + "=" * 60)
    print("UNCERTAINTY ANALYSIS SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("=" * 60)


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

    model.train()

    pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}',
                unit='batch', ncols=100, position=0, leave=True)

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        struct_input = input[:4]
        extra_fea = input[4]

        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(struct_input[0].cuda(non_blocking=True)),
                         Variable(struct_input[1].cuda(non_blocking=True)),
                         struct_input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in struct_input[3]],
                         Variable(extra_fea.cuda(non_blocking=True)))
        else:
            input_var = (Variable(struct_input[0]),
                         Variable(struct_input[1]),
                         struct_input[2],
                         struct_input[3],
                         Variable(extra_fea))

        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        if args.use_weighted_loss:
            if args.use_adaptive_weights:
                output, complexities, adaptive_weights = model(*input_var, return_complexity=True)
                loss = criterion(output, target_var, complexities=complexities, adaptive_weights=adaptive_weights)
            else:
                output, complexities = model(*input_var, return_complexity=True)
                loss = criterion(output, target_var, complexities=complexities)
        else:
            output = model(*input_var, return_complexity=False)
            loss = criterion(output, target_var)

        if args.use_weighted_loss and i == 0 and epoch % 10 == 0:
            if args.use_adaptive_weights:
                fusion_weights = model.complexity_module.fusion_weights.detach().cpu()
                weight_base = model.complexity_module.weight_base.item()
                weight_range = model.complexity_module.weight_range.item()
                print(f"[Epoch {epoch}] 自适应参数 - 融合权重: {fusion_weights.tolist()}, "
                      f"基础权重: {weight_base:.3f}, 范围: {weight_range:.3f}, "
                      f"平均权重: {criterion.last_weights_mean:.3f}")
            else:
                print(f"[Epoch {epoch}] 固定权重 - 平均复杂度: {criterion.last_complexity_mean:.3f}, "
                      f"平均权重: {criterion.last_weights_mean:.3f}")

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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

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

        if i % args.print_freq == 0:
            if args.task == 'regression':
                tqdm.write(f'Epoch {epoch} - Batch {i}/{len(train_loader)} - '
                           f'Loss: {losses.avg:.4f} - MAE: {mae_errors.avg:.3f} - '
                           f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            else:
                tqdm.write(f'Epoch {epoch} - Batch {i}/{len(train_loader)} - '
                           f'Loss: {losses.avg:.4f} - Acc: {accuracies.avg:.3f} - '
                           f'F1: {fscores.avg:.3f} - LR: {optimizer.param_groups[0]["lr"]:.6f}')

        pbar.update(1)

    pbar.close()

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

    val_targets = []
    val_preds = []
    val_cif_ids = []

    model.eval()

    desc = 'Testing' if test else 'Validating'
    pbar = tqdm(total=len(val_loader), desc=desc, unit='batch', ncols=100, position=0, leave=True)

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        struct_input = input[:4]
        extra_fea = input[4]

        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(struct_input[0].cuda(non_blocking=True)),
                             Variable(struct_input[1].cuda(non_blocking=True)),
                             struct_input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in struct_input[3]],
                             Variable(extra_fea.cuda(non_blocking=True)))
        else:
            with torch.no_grad():
                input_var = (Variable(struct_input[0]),
                             Variable(struct_input[1]),
                             struct_input[2],
                             struct_input[3],
                             Variable(extra_fea))

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

        if args.use_weighted_loss:
            if args.use_adaptive_weights:
                output, complexities, adaptive_weights = model(*input_var, return_complexity=True)
                loss = criterion(output, target_var, complexities=complexities, adaptive_weights=adaptive_weights)
            else:
                output, complexities = model(*input_var, return_complexity=True)
                loss = criterion(output, target_var, complexities=complexities)
        else:
            output = model(*input_var, return_complexity=False)
            loss = criterion(output, target_var)

        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            val_pred = normalizer.denorm(output.data.cpu())
            val_target = target
            val_preds += val_pred.view(-1).tolist()
            val_targets += val_target.view(-1).tolist()
            val_cif_ids += batch_cif_ids

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
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
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
    """Computes the mean absolute error between prediction and target"""
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
    loss_data = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_data.to_csv(os.path.join(subdirs["train_data"], 'loss_curve_data.csv'), index=False)
    print("Loss curve data saved")

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
    class Args:
        def __init__(self):
            self.data_options = ['']
            self.dropout = 0.2
            self.task = 'regression'
            self.disable_cuda = False
            self.workers = 0
            self.epochs = 100
            self.start_epoch = 0
            self.batch_size = 128
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
            self.feature_file = ''
            self.manual_notes = ''
            self.use_weighted_loss = True
            self.use_adaptive_weights = True
            self.weight_min = 0.2
            self.weight_max = 1.2
            self.mc_dropout_samples = 50
            self.confidence_interval = 95
            self.output_dir = ''


    args = Args()

    try:
        import scipy.stats
    except ImportError:
        print("Warning: scipy not installed. Installing scipy...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
        import scipy.stats

    main()
