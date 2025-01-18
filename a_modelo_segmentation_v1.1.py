# Importações Necessárias

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchnet as tnt
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import copy
from glob import glob
import os
import functools
from tqdm.auto import tqdm
import time
import csv
from plyfile import PlyData
import random
import logging
import open3d as o3d
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split

# Configurações de Depuração
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Configuração do Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verificar Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando Dispositivo:", device)

# Definir Parâmetros de Treinamento

class Args:
    pass

args = Args()
class_names = ['equipamento', 'flange']
args.n_class = len(class_names)
args.n_epoch = 10
args.subsample_size = 8192
args.batch_size = 8
args.input_feats = 'xyzrgbi'
args.lr = 1e-3
args.wd = 1e-5
args.cuda = device.type == 'cuda'

# Configuração dos Caminhos de Dados

project_dir = "./DATA/"
pointcloud_train_files = glob(os.path.join(project_dir, "train/*.ply"))
pointcloud_test_files = glob(os.path.join(project_dir, "test/*.ply"))

print(f"{len(pointcloud_train_files)} arquivos de treino, {len(pointcloud_test_files)} arquivos de teste")
print("Cuda disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Número de GPUs:", torch.cuda.device_count())
    print("Nome da GPU:", torch.cuda.get_device_name(0))

def read_ply_with_labels_plyfile(filename):
    plydata = PlyData.read(filename)
    vertex_data = plydata['vertex'].data

    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    r = vertex_data['red'] if 'red' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    g = vertex_data['green'] if 'green' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    b = vertex_data['blue'] if 'blue' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    intensity = vertex_data['intensity'] if 'intensity' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    labels = vertex_data['scalar_Classification'] if 'scalar_Classification' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.int32)

    labels = labels.astype(np.int32)

    if len(x) != len(labels):
        raise ValueError(f"Erro no arquivo {filename}: número de pontos ({len(x)}) não corresponde ao número de rótulos ({len(labels)}).")

    data = {
        'x': x,
        'y': y,
        'z': z,
        'r': r,
        'g': g,
        'b': b,
        'intensity': intensity,
        'scalar_Classification': labels
    }
    return data

# Funções de Augmentação

def random_rotation(cloud, angle=None):
    if angle is None:
        angle = np.random.uniform(-np.pi, np.pi)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    rotation_matrix = torch.from_numpy(rotation_matrix).float().to(cloud.device)
    cloud[:3, :] = rotation_matrix @ cloud[:3, :]
    return cloud

def random_scaling(cloud, scale_range=(0.8, 1.2)):
    scale = np.random.uniform(*scale_range)
    scaled_cloud = cloud.clone()
    scaled_cloud[:3, :] *= scale
    return scaled_cloud

def jitter(cloud, sigma=0.01, clip=0.05):
    jittered = cloud.clone()
    noise = torch.clamp(torch.randn_like(jittered[:3, :]) * sigma, -clip, clip)
    jittered[:3, :] += noise
    return jittered

def shift(cloud, shift_range=0.1):
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    shifted_cloud = cloud.clone()
    shifted_cloud[:3, :] += torch.tensor(shifts, device=cloud.device).unsqueeze(1)
    return shifted_cloud

def augment_cloud(cloud, gt):
    cloud = random_rotation(cloud)
    cloud = random_scaling(cloud)
    cloud = jitter(cloud)
    cloud = shift(cloud)
    return cloud, gt

def augment_flange(cloud, gt):
    flange_indices = (gt == 1)
    if flange_indices.sum() == 0:
        return cloud, gt

    rotation_angles = np.random.uniform(-np.pi/4, np.pi/4, 3)
    scale_factors = np.random.uniform(0.9, 1.1, 3)

    cloud_flange = cloud[:, flange_indices]
    for i in range(3):
        cloud_flange[i:i+1, :] *= scale_factors[i]

    Rx = torch.tensor([[1, 0, 0],
                       [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
                       [0, np.sin(rotation_angles[0]),  np.cos(rotation_angles[0])]], 
                      dtype=torch.float32, device=cloud.device)
    Ry = torch.tensor([[ np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
                       [0, 1, 0],
                       [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]], 
                      dtype=torch.float32, device=cloud.device)
    Rz = torch.tensor([[np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
                       [np.sin(rotation_angles[2]),  np.cos(rotation_angles[2]), 0],
                       [0, 0, 1]], 
                      dtype=torch.float32, device=cloud.device)
    rotation_matrix = Rz @ Ry @ Rx
    cloud_flange[:3, :] = rotation_matrix @ cloud_flange[:3, :]

    edge_noise = torch.randn_like(cloud_flange[:3, :]) * 0.002
    cloud_flange[:3, :] += edge_noise

    cloud[:, flange_indices] = cloud_flange
    return cloud, gt

def cloud_collate(batch):
    clouds, labels = list(zip(*batch))
    return clouds, labels

def estimate_curvature(xyz):
    xyz = xyz.T
    N = xyz.shape[0]
    curvature = np.zeros(N)
    tree = KDTree(xyz)
    k = 16
    for i in range(N):
        idx = tree.query(xyz[i:i+1], k=k, return_distance=False)
        neighbors = xyz[idx[0]]
        neighbors_centered = neighbors - np.mean(neighbors, axis=0)
        cov = np.dot(neighbors_centered.T, neighbors_centered)
        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.sort(eigvals)
        curvature[i] = eigvals[0] / (np.sum(eigvals) + 1e-8)
    return curvature

# Ajuste label_mapping se necessário
# Se a flange não for 1 no dataset, mude aqui.
label_mapping = {0: 0, 1: 1}  # Ajuste conforme os rótulos originais encontrados

def cloud_loader(tile_name, features_used, file_type='ply', max_points=8192, augment=False):
    data = read_ply_with_labels_plyfile(tile_name)

    # Verificando os rótulos originais
    unique_labels = np.unique(data['scalar_Classification'])
    print(f"Arquivo: {tile_name}, Rótulos originais únicos: {unique_labels}")

    xyz = np.vstack((data['x'], data['y'], data['z']))
    mean_f = np.mean(xyz, axis=1, keepdims=True)
    min_f = np.min(xyz, axis=1, keepdims=True)
    xyz[0] -= mean_f[0]
    xyz[1] -= mean_f[1]
    xyz[2] -= min_f[2]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.T)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals).T

    curvature = estimate_curvature(xyz)

    features = [xyz, normals, curvature[np.newaxis,:]]

    if 'rgb' in features_used:
        if 'r' in data and 'g' in data and 'b' in data:
            rgb = np.vstack((data['r'], data['g'], data['b']))
        else:
            print(f"RGB columns not found in {tile_name}. Using zeros.")
            rgb = np.zeros((3, len(data['x'])), dtype=np.float32)
        features.append(rgb)

    if 'i' in features_used:
        if 'intensity' in data:
            intensity = data['intensity']
            IQR = np.quantile(intensity, 0.75) - np.quantile(intensity, 0.25)
            if IQR != 0:
                n_intensity = ((intensity - np.median(intensity)) / IQR)
                n_intensity -= np.min(n_intensity)
            else:
                n_intensity = intensity - np.min(intensity)
            intensity = n_intensity[np.newaxis, :]
        else:
            print(f"'intensity' column not found in {tile_name}. Using zeros.")
            intensity = np.zeros((1, len(data['x'])), dtype=np.float32)
        features.append(intensity.astype(np.float32))

    gt = data['scalar_Classification'].astype(np.int32)
    
    gt_mapped = np.array([label_mapping.get(label, -1) for label in gt])
    # Verificar se mapeou corretamente
    unique_mapped = np.unique(gt_mapped)
    print(f"Arquivo: {tile_name}, Rótulos mapeados únicos: {unique_mapped}")

    if -1 in gt_mapped:
        invalid_labels = set(gt[gt_mapped == -1])
        raise ValueError(f"Found labels {invalid_labels} not in mapping in {tile_name}")

    for i in range(len(features)):
        features[i] = (features[i] - np.mean(features[i], axis=1, keepdims=True)) / (
                np.std(features[i], axis=1, keepdims=True) + 1e-8)

    cloud_data = np.vstack(features)
    cloud_data = torch.from_numpy(cloud_data).type(torch.float32)
    gt_mapped = torch.from_numpy(gt_mapped).long()

    # Amostragem estratificada
    flange_indices = (gt_mapped == 1).nonzero(as_tuple=False).squeeze()
    equipment_indices = (gt_mapped == 0).nonzero(as_tuple=False).squeeze()

    num_flange = flange_indices.shape[0] if flange_indices.ndim > 0 else 0
    num_equipment = equipment_indices.shape[0] if equipment_indices.ndim > 0 else 0

    # Ajuste a porcentagem conforme necessidade
    desired_flange_ratio = 0.3  
    desired_flange_points = max(int(max_points * desired_flange_ratio), 1)
    desired_equipment_points = max_points - desired_flange_points

    if num_flange == 0:
        # Sem flanges
        sampled_equipment_indices = np.random.choice(equipment_indices.cpu(), max_points, replace=True)
        sampled_indices = sampled_equipment_indices
    else:
        sampled_flange_indices = np.random.choice(flange_indices.cpu(), min(num_flange, desired_flange_points), replace=True)
        sampled_equipment_indices = np.random.choice(equipment_indices.cpu(), desired_equipment_points, replace=True)
        sampled_indices = np.concatenate([sampled_flange_indices, sampled_equipment_indices])
        np.random.shuffle(sampled_indices)

    # Verifique quantos pontos realmente conseguiu de cada classe
    final_gt = gt_mapped[sampled_indices]
    final_flange_count = (final_gt == 1).sum().item()
    final_equip_count = (final_gt == 0).sum().item()
    print(f"Arquivo: {tile_name}, pontos finais flange: {final_flange_count}, pontos finais equipamento: {final_equip_count}")

    cloud_data = cloud_data[:, sampled_indices]
    gt_mapped = final_gt

    if augment:
        cloud_data, gt_mapped = augment_cloud(cloud_data, gt_mapped)
        cloud_data, gt_mapped = augment_flange(cloud_data, gt_mapped)

    return cloud_data, gt_mapped

cloud_features = "xyzrgbi"
file_type = 'ply'

max_points = args.subsample_size

num_input_features = 0
num_input_features += 3  # xyz
num_input_features += 3  # normals
num_input_features += 1  # curvature
if 'rgb' in cloud_features:
    num_input_features += 3
if 'i' in cloud_features:
    num_input_features += 1
args.n_input_feats = num_input_features

print(f"Número de características de entrada: {args.n_input_feats}")

# Função para obter a presença de flanges
def get_flange_presence(file_list):
    flange_presence = []
    for file in file_list:
        data = read_ply_with_labels_plyfile(file)
        labels = data['scalar_Classification'].astype(np.int32)
        flange_presence.append(1 if np.sum(labels == 1) > 0 else 0)  # Ajuste caso flange não seja 1
    return flange_presence

train_flange_presence = get_flange_presence(pointcloud_train_files)
test_flange_presence = get_flange_presence(pointcloud_test_files)

train_files, valid_files, train_labels, valid_labels = train_test_split(
    pointcloud_train_files,
    train_flange_presence,
    test_size=0.1,
    stratify=train_flange_presence,
    random_state=42
)

test_files = pointcloud_test_files

print(f"{len(train_files)} arquivos de treino, {len(valid_files)} arquivos de validação, {len(test_files)} arquivos de teste")

import torchnet as tnt

train_set = tnt.dataset.ListDataset(
    train_files,
    functools.partial(
        cloud_loader,
        features_used=cloud_features,
        file_type=file_type,
        max_points=max_points,
        augment=True
    )
)

valid_set = tnt.dataset.ListDataset(
    valid_files,
    functools.partial(
        cloud_loader,
        features_used=cloud_features,
        file_type=file_type,
        max_points=max_points,
        augment=False
    )
)

test_set = tnt.dataset.ListDataset(
    test_files,
    functools.partial(
        cloud_loader,
        features_used=cloud_features,
        file_type=file_type,
        max_points=max_points,
        augment=False
    )
)

from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule

class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes=2, input_channels=8):
        super().__init__()
        self.sa1 = PointnetSAModuleMSG(
            npoint=2048,
            radii=[0.1, 0.2],
            nsamples=[32, 64],
            mlps=[
                [input_channels, 64, 64, 128], 
                [input_channels, 64, 96, 128]
            ],
            use_xyz=True
        )

        self.sa2 = PointnetSAModuleMSG(
            npoint=512,
            radii=[0.2, 0.4],
            nsamples=[64, 128],
            mlps=[
                [256, 128, 128, 256],
                [256, 128, 196, 256]
            ],
            use_xyz=True
        )

        self.sa3 = PointnetSAModule(
            npoint=128,
            radius=0.8,
            nsample=256,
            mlp=[512, 256, 512, 1024],
            use_xyz=True
        )

        self.attention = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1, 1),
            nn.Sigmoid()
        )

        self.fp3 = PointnetFPModule(mlp=[1536, 512, 512])
        self.fp2 = PointnetFPModule(mlp=[768, 256, 256])
        self.fp1 = PointnetFPModule(mlp=[267, 128, 128])

        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz, features):
        xyz = xyz.float()
        features = features.float()

        l0_xyz = xyz
        l0_points = features

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        attention_weights = self.attention(l3_points)
        l3_points = l3_points * attention_weights

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz.transpose(1,2), features], dim=1), l1_points)

        x = self.classifier(l0_points)
        x = x.transpose(2, 1).contiguous()

        return x

class PointCloudClassifier:
    def __init__(self, args, device=torch.device("cpu")):
        self.subsample_size = args.subsample_size
        self.n_input_feats = args.n_input_feats
        self.n_class = args.n_class
        self.is_cuda = args.cuda
        self.device = device

    def run(self, model, clouds, train=False):
        if train:
            model.train()
        else:
            model.eval()

        clouds = clouds.to(self.device, non_blocking=True)
        pointcloud = clouds.permute(0, 2, 1).float()

        xyz = pointcloud[:, :, :3].contiguous()
        features = pointcloud[:, :, 3:].transpose(1, 2).contiguous()

        pred = model(xyz, features)
        return pred

class ConfusionMatrix:
    def __init__(self, n_class, class_names):
        self.CM = np.zeros((n_class, n_class))
        self.n_class = n_class
        self.class_names = class_names

    def clear(self):
        self.CM = np.zeros((self.n_class, self.n_class))

    def add_batch(self, gt, pred):
        if len(gt) != len(pred):
            raise ValueError(f"Inconsistent number of samples: {len(gt)} in gt, {len(pred)} in pred")
        self.CM += confusion_matrix(gt, pred, labels=list(range(self.n_class)))

    def overall_accuracy(self):
        return 100 * self.CM.trace() / self.CM.sum()

    def class_IoU(self, show=1):
        ious = np.diag(self.CM) / (np.sum(self.CM, 1) + np.sum(self.CM, 0) - np.diag(self.CM))
        if show:
            print(' / '.join('{} : {:3.2f}%'.format(name, 100 * iou) for name, iou in zip(self.class_names, ious)))
        return 100 * np.nanmean(ious)

    def print_confusion_matrix(self):
        print("Matriz de Confusão:")
        print(self.CM)
        if self.n_class <= 10:
            y_true = []
            y_pred = []
            for i in range(self.n_class):
                y_true.extend([i] * int(self.CM[i].sum()))
                y_pred.extend([i] * int(self.CM[i][i]))
                for j in range(self.n_class):
                    if j != i:
                        y_pred.extend([j] * int(self.CM[i][j]))
            print(classification_report(y_true, y_pred, target_names=self.class_names))
        else:
            print("Muitas classes para exibir relatório detalhado.")

def dice_loss(pred, target, smooth=1.):
    num_classes = pred.shape[2]
    pred = torch.softmax(pred, dim=2)
    with torch.no_grad():
        target_one_hot = F.one_hot(target, num_classes)
    target_one_hot = target_one_hot.permute(0, 2, 1).float()
    intersection = (pred.transpose(1, 2) * target_one_hot).sum(dim=2)
    union = pred.transpose(1, 2).sum(dim=2) + target_one_hot.sum(dim=2)
    dice_score = (2. * intersection + smooth) / (union + smooth)
    dice_loss_value = 1 - dice_score
    return dice_loss_value.mean()

def focal_loss(inputs, targets, alpha, gamma=2.0, reduction='mean'):
    alpha = alpha.to(inputs.device)
    inputs_flat = inputs.view(-1, inputs.shape[-1])
    targets_flat = targets.view(-1)
    CE_loss = F.cross_entropy(inputs_flat, targets_flat, reduction='none')
    pt = torch.exp(-CE_loss)
    F_loss = alpha[targets_flat] * (1 - pt) ** gamma * CE_loss
    return F_loss.mean() if reduction == 'mean' else F_loss.sum()

def custom_flange_loss(pred, target, class_weights):
    focal = focal_loss(pred, target, class_weights)
    dice = dice_loss(pred, target)
    total_loss = focal + dice
    return total_loss

def train(model, PCC, optimizer, args, device, class_weights):
    model.train()
    loader = torch.utils.data.DataLoader(
        train_set,
        collate_fn=cloud_collate,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )
    loader = tqdm(loader, ncols=80, leave=False, desc="Training")
    cm = ConfusionMatrix(args.n_class, class_names=class_names)
    loss_meter = tnt.meter.AverageValueMeter()

    for clouds, gts in loader:
        optimizer.zero_grad()
        clouds = [c.to(device, non_blocking=True) for c in clouds]
        gts = [g.to(device=device, dtype=torch.long, non_blocking=True) for g in gts]

        clouds = torch.stack(clouds, dim=0)
        gts = torch.stack(gts, dim=0)

        pred = PCC.run(model, clouds, train=True)
        total_loss = custom_flange_loss(pred, gts, class_weights)

        total_loss.backward()
        optimizer.step()

        loss_meter.add(total_loss.item())
        cm.add_batch(gts.cpu().flatten(), pred.argmax(2).cpu().flatten())

    cm.print_confusion_matrix()
    return cm, loss_meter.value()[0]

def eval_model(model, PCC, test, args, device, class_weights):
    model.eval()
    if test:
        dataset = test_set
        desc = "Test"
    else:
        dataset = valid_set
        desc = "Validation"

    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=cloud_collate,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0
    )
    loader = tqdm(loader, ncols=80, leave=False, desc=desc)

    loss_meter = tnt.meter.AverageValueMeter()
    cm = ConfusionMatrix(args.n_class, class_names=class_names)

    with torch.no_grad():
        for clouds, gts in loader:
            clouds = [c.to(device, non_blocking=True) for c in clouds]
            gts = [g.to(device=device, dtype=torch.long, non_blocking=True) for g in gts]

            clouds = torch.stack(clouds, dim=0)
            gts = torch.stack(gts, dim=0)

            pred = PCC.run(model, clouds, train=False)
            total_loss = custom_flange_loss(pred, gts, class_weights)

            loss_meter.add(total_loss.item())
            cm.add_batch(gts.cpu().flatten(), pred.argmax(2).cpu().flatten())

    cm.print_confusion_matrix()
    return cm, loss_meter.value()[0]

def train_full(args, device):
    input_channels = args.n_input_feats - 3

    model = PointNet2SemSeg(num_classes=args.n_class, input_channels=input_channels).to(device)
    print('Número total de parâmetros:', sum(p.numel() for p in model.parameters()))

    best_model = None
    best_mIoU = 0
    no_improve_epochs = 0
    early_stop_patience = 10

    PCC = PointCloudClassifier(args, device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Ajustando o peso da classe flange para 10.0 (em vez de 50.0)
    class_weights = torch.tensor([1.0, 50.0], dtype=torch.float32, device=device)
    class_weights.requires_grad = False
    print(f"Pesos das Classes: {class_weights}")

    total_flange_points = 0
    for _, gt in train_set:
        total_flange_points += (gt == 1).sum().item()
    print(f"Total de pontos de flange no conjunto de treinamento: {total_flange_points}")
    if total_flange_points == 0:
        raise ValueError("Nenhum ponto de flange encontrado no conjunto de treinamento.")

    metrics_pn = {}
    metrics_pn['definition'] = [
        ['train_oa', 'train_mIoU', 'train_loss'],
        ['valid_oa', 'valid_mIoU', 'valid_loss'],
        ['test_oa', 'test_mIoU', 'test_loss']
    ]

    for i_epoch in tqdm(range(args.n_epoch), desc='Training'):
        cm_train, loss_train = train(model, PCC, optimizer, args, device, class_weights)
        mIoU = cm_train.class_IoU()
        tqdm.write(f'Época {i_epoch:3d} -> Acurácia de Treino: {cm_train.overall_accuracy():.2f}%, mIoU de Treino: {mIoU:.2f}%, Loss de Treino: {loss_train:.4f}')
        metrics_pn[i_epoch] = [[cm_train.overall_accuracy(), mIoU, loss_train]]

        cm_valid, loss_valid = eval_model(model, PCC, False, args=args, device=device, class_weights=class_weights)
        mIoU_valid = cm_valid.class_IoU()
        metrics_pn[i_epoch].append([cm_valid.overall_accuracy(), mIoU_valid, loss_valid])

        scheduler.step()

        best_valid = False
        if mIoU_valid > best_mIoU:
            best_valid = True
            best_mIoU = mIoU_valid
            best_model = copy.deepcopy(model)
            no_improve_epochs = 0
            tqdm.write(f'Melhor desempenho na época {i_epoch:3d} -> Acurácia de Validação: {cm_valid.overall_accuracy():.2f}%, mIoU de Validação: {mIoU_valid:.2f}%, Loss de Validação: {loss_valid:.4f}')
        else:
            no_improve_epochs += 1
            tqdm.write(f'Época {i_epoch:3d} -> Acurácia de Validação: {cm_valid.overall_accuracy():.2f}%, mIoU de Validação: {mIoU_valid:.2f}%, Loss de Validação: {loss_valid:.4f}')

        if no_improve_epochs >= early_stop_patience:
            tqdm.write(f'Early stopping na época {i_epoch} devido a {early_stop_patience} épocas sem melhora.')
            break

        if i_epoch == args.n_epoch - 1 or best_valid:
            cm_test, loss_test = eval_model(best_model, PCC, True, args=args, device=device, class_weights=class_weights)
            mIoU_test = cm_test.class_IoU()
            tqdm.write(f'Época {i_epoch:3d} -> Acurácia de Teste: {cm_test.overall_accuracy():.2f}%, mIoU de Teste: {mIoU_test:.2f}%, Loss de Teste: {loss_test:.4f}')
            metrics_pn[i_epoch].append([cm_test.overall_accuracy(), mIoU_test, loss_test])

    return best_model, metrics_pn, optimizer, scheduler

def tile_prediction(tile_name, model=None, PCC=None, Visualization=True, features_used='xyzrgbi', file_type='ply'):
    cloud, gt = cloud_loader(tile_name, features_used, file_type=file_type)

    if cloud.shape[1] > args.subsample_size:
        cloud = cloud[:, :args.subsample_size]
        gt = gt[:args.subsample_size]

    clouds = cloud.unsqueeze(0).to(device)
    labels = PCC.run(model, clouds, train=False)
    labels = labels.argmax(2).squeeze(0).cpu().numpy()

    xyz = cloud[:3, :].numpy().T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if Visualization:
        colors = np.zeros((labels.shape[0], 3))
        colormap = {
            0: [0, 0, 1],
            1: [1, 0, 0],
        }
        for label in np.unique(labels):
            colors[labels == label] = colormap.get(label, [0.5, 0.5, 0.5])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

    return pcd, labels

def export_results(trained_model, metrics_pn, project_dir, optimizer, scheduler):
    model_path = f'./pointnet2_final_model_{os.path.basename(project_dir)}.pth'
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, model_path)
    print(f"Modelo salvo em {model_path}")

    metrics_path = f"./metrics_pointnet2_final_{os.path.basename(project_dir)}.csv"
    with open(metrics_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_OA', 'Train_mIoU', 'Train_Loss',
                         'Valid_OA', 'Valid_mIoU', 'Valid_Loss',
                         'Test_OA', 'Test_mIoU', 'Test_Loss'])
        for epoch, metrics in metrics_pn.items():
            if epoch == 'definition':
                continue
            if len(metrics) == 3:
                train_metrics, valid_metrics, test_metrics = metrics
            elif len(metrics) == 2:
                train_metrics, valid_metrics = metrics
                test_metrics = [0, 0, 0]
            else:
                train_metrics = metrics[0]
                valid_metrics = [0, 0, 0]
                test_metrics = [0, 0, 0]
            writer.writerow([epoch] + train_metrics + valid_metrics + test_metrics)
    print(f"Métricas salvas em {metrics_path}")

if __name__ == "__main__":
    t0 = time.time()
    trained_model, metrics_pn, optimizer, scheduler = train_full(args, device)
    t1 = time.time()

    print('-' * 50)
    print(f"Tempo total de treinamento: {t1 - t0:.2f} segundos")
    print('=' * 50)

    PCC = PointCloudClassifier(args, device=device)

    if len(test_files) > 0:
        for idx in range(min(2, len(test_files))):
            selection = test_files[idx]
            print(f"Visualizando arquivo de teste: {selection}")
            pcd, labels = tile_prediction(selection, model=trained_model, PCC=PCC, file_type=file_type)
    else:
        print("Nenhum arquivo de teste encontrado. Pulando visualização da predição.")

    export_results(trained_model, metrics_pn, project_dir, optimizer, scheduler)
