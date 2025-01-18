# 1. Setup Environment

# Math library
import numpy as np

# Drawing library
import matplotlib.pyplot as plt
import open3d as o3d
import torch.nn.functional as F

# Deep Learning Libraries
import torch
import torch.nn.functional as nnf
import torch.nn as nn
import torchnet as tnt
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

# Utility libraries
import copy
from glob import glob
import os
import functools
import mock
from tqdm.auto import tqdm
import time
import csv  # For saving metrics
from plyfile import PlyData
from collections import Counter

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# 2. Define Training Parameters Early

# Define the training parameters
args = mock.Mock()

# Define class names
class_names = ['equipamento', 'flange']
args.n_class = len(class_names)  # Set to 2 for two classes
args.n_epoch = 50  # Increased epochs
args.subsample_size = 8192  # Increased point sampling

# Other parameters
args.n_epoch_test = 1
args.batch_size = 4
args.input_feats = 'xyzrgbi'
args.n_input_feats = len(args.input_feats)
args.MLP_1 = [128, 256, 512]
args.MLP_2 = [512, 1024, 2048]
args.MLP_3 = [1024, 512, 256]
args.show_test = 0
args.lr = 3e-4
args.wd = 1e-5
args.cuda = device.type == 'cuda'

# 3. Data Path Setup
project_dir = "./DATA/"
pointcloud_train_files = glob(os.path.join(project_dir, "train/*.ply"))
pointcloud_test_files = glob(os.path.join(project_dir, "test/*.ply"))

print(f"{len(pointcloud_train_files)} tiles in train set, {len(pointcloud_test_files)} tiles in test set")

# 4. Define PLY File Reader Using plyfile

def read_ply_with_labels_plyfile(filename):
    """
    Reads a PLY file and returns a dictionary with the data, including labels.
    Assumes that the PLY file contains a 'scalar_Classification' property.
    """
    plydata = PlyData.read(filename)
    vertex_data = plydata['vertex'].data

    # Extract properties with fallback defaults
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    r = vertex_data['red'] if 'red' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    g = vertex_data['green'] if 'green' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    b = vertex_data['blue'] if 'blue' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    intensity = vertex_data['intensity'] if 'intensity' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    labels = vertex_data['scalar_Classification'] if 'scalar_Classification' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.int32)

    # Convert labels to a smaller data type (int32) to save memory
    labels = labels.astype(np.int32)

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

# 5. Define Data Augmentation Functions (Optional but Recommended)

def random_rotation(cloud):
    """
    Randomly rotates the point cloud around the Z-axis.
    """
    angle = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,             0,              1]
    ])
    # Convert rotation_matrix to a Torch tensor and move to the same device as cloud
    rotation_matrix_tensor = torch.from_numpy(rotation_matrix).float().to(cloud.device)
    rotated_cloud = rotation_matrix_tensor @ cloud[:3, :]
    return torch.cat((rotated_cloud, cloud[3:, :]), dim=0)

def random_scaling(cloud, scale_range=(0.9, 1.1)):
    """
    Randomly scales the point cloud.
    """
    scale = np.random.uniform(*scale_range)
    scaled_cloud = cloud.clone()
    scaled_cloud[:3, :] *= scale
    return scaled_cloud

def jitter(cloud, sigma=0.01, clip=0.05):
    """
    Adds Gaussian noise to the point cloud to simulate sensor noise.
    """
    jittered = cloud.clone()
    noise = torch.clamp(torch.randn_like(jittered[:3, :]) * sigma, -clip, clip)
    jittered[:3, :] += noise
    return jittered

def preprocess_point_cloud(cloud):
    # Normalize coordinates
    centroid = np.mean(cloud[:3, :], axis=1, keepdims=True)
    cloud[:3, :] = cloud[:3, :] - centroid

    # Scale to unit sphere
    m = np.max(np.sqrt(np.sum(cloud[:3, :] ** 2, axis=0)))
    cloud[:3, :] = cloud[:3, :] / m

    return cloud

# 6. Modify cloud_loader to Include Proper Returns and Augmentations

def cloud_loader(tile_name, features_used, file_type='ply', max_points=1000000, augment=False):
    """
    Loads, preprocesses, and processes a PLY or TXT file.
    """
    if file_type == 'ply':
        # Use plyfile to read PLY files with labels
        data = read_ply_with_labels_plyfile(tile_name)

        # Extract and preprocess features
        features = []

        if 'xyz' in features_used:
            xyz = np.vstack((data['x'], data['y'], data['z']))
            features.append(xyz)

        if 'rgb' in features_used:
            if 'r' in data and 'g' in data and 'b' in data:
                # Normalize RGB values to [0,1] range if they're in [0,255]
                rgb = np.vstack((data['r'], data['g'], data['b']))
                if rgb.max() > 1:
                    rgb = rgb / 255.0
                features.append(rgb)
            else:
                print(f"RGB columns not found in {tile_name}. Using zeros.")
                colors = np.zeros((3, len(data['x'])), dtype=np.float32)
                features.append(colors)

        if 'i' in features_used:
            if 'intensity' in data:
                intensity = data['intensity']
                # Normalize intensity using robust scaling
                IQR = np.quantile(intensity, 0.75) - np.quantile(intensity, 0.25)
                if IQR != 0:
                    n_intensity = ((intensity - np.median(intensity)) / IQR)
                    n_intensity -= np.min(n_intensity)
                else:
                    n_intensity = intensity - np.min(intensity)
                intensity = n_intensity[np.newaxis, :]
                features.append(intensity.astype(np.float32))
            else:
                print(f"'intensity' column not found in {tile_name}. Using zeros.")
                intensity = np.zeros((1, len(data['x'])), dtype=np.float32)
                features.append(intensity)

        # Ground truth labels
        gt = data['scalar_Classification']
        gt = np.clip(gt, 0, args.n_class - 1)  # Ensure labels are within range [0, n_class-1]
        gt = torch.from_numpy(gt).long()

        # Convert features to tensor and stack them
        cloud_data = np.vstack(features)

        # Apply preprocessing to the point cloud
        cloud_data = preprocess_point_cloud(cloud_data)

    elif file_type == 'txt':
        try:
            cloud_data = np.loadtxt(tile_name)
        except Exception as e:
            raise ValueError(f"Error loading TXT file {tile_name}: {e}")

        features = []

        if 'xyz' in features_used:
            xyz = cloud_data[:, 0:3].T
            features.append(xyz)

        if 'rgb' in features_used:
            rgb = cloud_data[:, 3:6].T
            if rgb.max() > 1:  # Normalize if in [0,255] range
                rgb = rgb / 255.0
            features.append(rgb)

        if 'i' in features_used:
            intensity = cloud_data[:, 6].T
            # Normalize intensity
            IQR = np.quantile(intensity, 0.75) - np.quantile(intensity, 0.25)
            if IQR != 0:
                n_intensity = ((intensity - np.median(intensity)) / IQR)
                n_intensity -= np.min(n_intensity)
            else:
                n_intensity = intensity - np.min(intensity)
            features.append(n_intensity[np.newaxis, :])

        # Ground truth labels
        gt = cloud_data[:, -1].astype(np.int32)
        gt = np.clip(gt, 0, args.n_class - 1)
        gt = torch.from_numpy(gt).long()

        # Stack features and preprocess
        cloud_data = np.vstack(features).astype(np.float32)
        cloud_data = preprocess_point_cloud(cloud_data)

    else:
        raise ValueError("Unsupported file type. Use 'ply' or 'txt'.")

    # Downsample if necessary
    num_points = cloud_data.shape[1]
    if num_points > max_points:
        print(f"Downsampling from {num_points} to {max_points} points in {tile_name}.")
        indices = np.random.choice(num_points, max_points, replace=False)
        cloud_data = cloud_data[:, indices]
        gt = gt[indices]

    # Convert to torch.Tensor
    cloud_data = torch.from_numpy(cloud_data).float()

    # Apply Data Augmentation if specified
    if augment:
        cloud_data = random_rotation(cloud_data)
        cloud_data = random_scaling(cloud_data)
        cloud_data = jitter(cloud_data)

    return cloud_data, gt

# 7. Split Data into Train, Validation, and Test Sets

valid_ratio = 0.2
num_valid = int(len(pointcloud_train_files) * valid_ratio)
valid_index = np.random.choice(len(pointcloud_train_files), num_valid, replace=False)
valid_list = [pointcloud_train_files[i] for i in valid_index]
train_list = [pointcloud_train_files[i] for i in np.setdiff1d(list(range(len(pointcloud_train_files))), valid_index)]
test_list = pointcloud_test_files

print(f"{len(train_list)} tiles in train set, {len(valid_list)} tiles in validation set, {len(test_list)} tiles in test set")

# 8. Define Feature Usage and File Type

cloud_features = "xyzrgbi"
file_type = 'ply'  # Change to 'txt' if using TXT files

# 9. Define Maximum Points to Load per File

max_points = 1000000  # Adjust based on your system's memory

# 10. Create Datasets with Specified max_points and Augmentation

# Apply augmentation only to training data
train_set = tnt.dataset.ListDataset(
    train_list,
    functools.partial(
        cloud_loader,
        features_used=cloud_features,
        file_type=file_type,
        max_points=max_points,
        augment=True  # Enable augmentation
    )
)

# No augmentation for validation and test sets
valid_set = tnt.dataset.ListDataset(
    valid_list,
    functools.partial(
        cloud_loader,
        features_used=cloud_features,
        file_type=file_type,
        max_points=max_points,
        augment=False
    )
)
test_set = tnt.dataset.ListDataset(
    test_list,
    functools.partial(
        cloud_loader,
        features_used=cloud_features,
        file_type=file_type,
        max_points=max_points,
        augment=False
    )
)

# 11. Verification: Inspect a Sample from the Training Set

sample_cloud, sample_gt = train_set[0]
print(f"Sample Cloud Shape: {sample_cloud.shape}")  # Expected: (n_features, N)
print(f"Sample GT Shape: {sample_gt.shape}")        # Expected: (N,)
print(f"Sample GT Unique Labels: {torch.unique(sample_gt)}")  # Should show labels 0 and 1

# 12. Define PointNet Architecture

def cloud_collate(batch):
    """
    Collects a list of point clouds into a batch.
    Returns a list of point clouds and a single tensor of labels.
    """
    clouds, labels = list(zip(*batch))
    labels = torch.cat(labels, 0)
    return clouds, labels

class PointNet(nn.Module):
    def __init__(self, MLP_1, MLP_2, MLP_3, n_classes=2, input_feat=7, subsample_size=2048, device=torch.device("cpu")):
        super(PointNet, self).__init__()

        self.device = device
        self.subsample_size = subsample_size

        # T-Net for input transform
        self.transform_input = nn.Sequential(
            nn.Conv1d(input_feat, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Spatial transformer
        self.spatial_transform = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Enhanced feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv1d(input_feat, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Multi-scale feature aggregation
        self.multi_scale = nn.ModuleList([
            nn.Conv1d(256, 128, 1),
            nn.Conv1d(256, 128, 1),
            nn.Conv1d(256, 128, 1)
        ])

        # Local geometric feature learning
        self.geometric_features = nn.Sequential(
            nn.Conv1d(384, 512, 1),  # 384 = 128 * 3 from multi-scale
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Global context encoding
        self.global_features = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # Feature fusion module
        self.fusion = nn.Sequential(
            nn.Conv1d(1280, 512, 1),  # 1280 = 256 + 1024 (local + global)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, n_classes, 1)
        )

        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.to(self.device)

    def forward(self, x):
        batch_size = x.size(0)

        # Input transformation
        trans_feat = self.transform_input(x)
        trans = self.spatial_transform(trans_feat)

        # Extract point features
        point_feat = self.feature_extraction(x)

        # Multi-scale feature aggregation
        ms_feats = []
        for conv in self.multi_scale:
            ms_feats.append(conv(point_feat))
        ms_feat = torch.cat(ms_feats, 1)

        # Local geometric features
        local_feat = self.geometric_features(ms_feat)

        # Global context
        global_feat = self.global_features(local_feat)
        global_feat = self.maxpool(global_feat)
        global_feat = global_feat.repeat(1, 1, x.size(2))

        # Feature fusion
        fusion_feat = torch.cat([local_feat, global_feat], 1)
        fusion_feat = self.fusion(fusion_feat)

        # Segmentation
        out = self.segmentation_head(fusion_feat)

        return out

# 13. Define PointCloudClassifier

class PointCloudClassifier:
    """
    Main point cloud classifier class.
    """
    def __init__(self, args, device=torch.device("cpu")):
        self.subsample_size = args.subsample_size
        self.n_inputs_feats = 3
        if 'i' in args.input_feats:
            self.n_inputs_feats += 1
        if 'rgb' in args.input_feats:
            self.n_inputs_feats += 3
        self.n_class = args.n_class
        self.is_cuda = args.cuda
        self.device = device

    def run(self, model, clouds):
        """
        INPUT:
        - model: the neural network
        - clouds: list of tensors, each of size [n_features, n_points_i]

        OUTPUT:
        - pred: tensor of size [sum_i n_points_i, n_class]
        """
        n_batch = len(clouds)
        prediction_batch = torch.zeros((self.n_class, 0)).to(self.device)
        sampled_clouds = torch.zeros((n_batch, self.n_inputs_feats, self.subsample_size)).to(self.device)

        for i_batch in range(n_batch):
            cloud = clouds[i_batch].to(self.device)
            n_points = cloud.shape[1]
            if n_points < self.subsample_size:
                selected_points = np.random.choice(n_points, self.subsample_size, replace=True)
            else:
                selected_points = np.random.choice(n_points, self.subsample_size, replace=False)
            sampled_cloud = cloud[:, selected_points]
            sampled_clouds[i_batch, :, :] = sampled_cloud

        sampled_prediction = model(sampled_clouds)  # [n_batch, n_class, subsample_size]

        for i_batch in range(n_batch):
            cloud = clouds[i_batch][:3, :].cpu().numpy().T  # [n_points, 3]
            sampled_cloud = sampled_clouds[i_batch, :3, :].cpu().numpy().T  # [subsample_size, 3]

            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(sampled_cloud)
            _, closest_point = knn.kneighbors(cloud)  # closest_point: [n_points, 1]
            closest_point = closest_point.squeeze()  # [n_points]

            # Convert closest_point to Torch tensor
            closest_point_tensor = torch.from_numpy(closest_point).long().to(self.device)  # [n_points]

            # Expand dimensions to match for gathering
            closest_point_tensor = closest_point_tensor.unsqueeze(0).repeat(self.n_class, 1)  # [n_class, n_points]

            # Gather predictions
            prediction_full_cloud = sampled_prediction[i_batch].gather(1, closest_point_tensor)  # [n_class, n_points]
            prediction_batch = torch.cat((prediction_batch, prediction_full_cloud), 1)  # [n_class, total_points]

        return prediction_batch.permute(1, 0)  # [total_points, n_class]

# 14. Define ConfusionMatrix

class ConfusionMatrix:
    """
    Computes confusion matrix and related metrics.
    """
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
        return 100 * np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()

# 15. Define Training and Evaluation Functions

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def train(model, PCC, optimizer, scheduler, args, device, class_weights):
    model.train()
    
    # Create data loader with balanced sampling and augmentation
    dataset = copy.deepcopy(train_set)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=cloud_collate,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss_meter = tnt.meter.AverageValueMeter()
    cm = ConfusionMatrix(args.n_class, class_names=class_names)
    
    for cloud, gt in tqdm(loader, desc='Training', leave=False):
        cloud = [c.to(device) for c in cloud]
        gt = gt.to(device)
        
        # Apply augmentations
        augmented_clouds = []
        for c in cloud:
            # Random rotation
            if np.random.random() > 0.5:
                c = random_rotation(c)
            # Random scaling
            if np.random.random() > 0.5:
                c = random_scaling(c, scale_range=(0.8, 1.2))
            # Add noise
            if np.random.random() > 0.5:
                c = jitter(c, sigma=0.01, clip=0.02)
            augmented_clouds.append(c)
        
        optimizer.zero_grad()
        pred = PCC.run(model, augmented_clouds)
        
        # Combined loss
        ce_loss = F.cross_entropy(pred, gt, weight=class_weights)
        f_loss = focal_loss(pred, gt)
        loss = 0.5 * ce_loss + 0.5 * f_loss
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # Step the scheduler after optimizer.step()
        
        loss_meter.add(loss.item())
        
        cm.add_batch(gt.cpu().numpy(), pred.argmax(1).cpu().detach().numpy())
    
    return cm, loss_meter.value()[0]

def eval_model(model, PCC, test, args, device, class_weights):
    """
    Evaluate the model on the validation or test set.
    """
    model.eval()
    if test:
        loader = torch.utils.data.DataLoader(
            test_set,
            collate_fn=cloud_collate,
            batch_size=args.batch_size,
            shuffle=False
        )
        loader = tqdm(loader, ncols=500, leave=False, desc="Test")
    else:
        loader = torch.utils.data.DataLoader(
            valid_set,
            collate_fn=cloud_collate,
            batch_size=60,
            shuffle=False,
            drop_last=False
        )
        loader = tqdm(loader, ncols=500, leave=False, desc="Val")

    loss_meter = tnt.meter.AverageValueMeter()
    cm = ConfusionMatrix(args.n_class, class_names=class_names)

    with torch.no_grad():
        for index_batch, (cloud, gt) in enumerate(loader):
            cloud = [c.to(device) for c in cloud]
            gt = gt.to(device)

            pred = PCC.run(model, cloud)  # [n_points, n_class]

            # Compute loss on all points
            loss = nn.functional.cross_entropy(pred, gt, weight=class_weights)
            loss_meter.add(loss.item())

            # Update confusion matrix
            cm.add_batch(gt.cpu().numpy(), pred.argmax(1).cpu().detach().numpy())

    return cm, loss_meter.value()[0]

# 16. Define Full Training Loop

def train_full(args, device):
    """
    Full training loop with added learning rate scheduling
    """
    model = PointNet(
        MLP_1=args.MLP_1,
        MLP_2=args.MLP_2,
        MLP_3=args.MLP_3,
        n_classes=args.n_class,
        input_feat=args.n_input_feats,
        subsample_size=args.subsample_size,
        device=device
    ).to(device)

    print('Total number of parameters:', sum(p.numel() for p in model.parameters()))

    best_model = None
    best_mIoU = 0

    PCC = PointCloudClassifier(args, device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # Initialize the cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Initial restart interval
        T_mult=2,  # Multiplier for restart interval
        eta_min=1e-6  # Minimum learning rate
    )

    # Calculate class weights
    label_counts = Counter()
    for _, gt in train_set:
        label_counts.update(gt.tolist())

    class_weights = torch.tensor([1.0, 30.0], dtype=torch.float32, device=device)
    print(f"Pesos das Classes: {class_weights}")

    TESTCOLOR = '\033[104m'
    TRAINCOLOR = '\033[100m'
    VALIDCOLOR = '\033[45m'
    NORMALCOLOR = '\033[0m'

    metrics_pn = {}
    metrics_pn['definition'] = [
        ['train_oa', 'train_mIoU', 'train_loss'], 
        ['valid_oa', 'valid_mIoU', 'valid_loss'], 
        ['test_oa', 'test_mIoU', 'test_loss']
    ]

    for i_epoch in tqdm(range(args.n_epoch), desc='Training'):
        # Train for one epoch with scheduler
        cm_train, loss_train = train(model, PCC, optimizer, scheduler, args, device, class_weights)
        mIoU = cm_train.class_IoU(show=0)
        
        current_lr = optimizer.param_groups[0]['lr']
        tqdm.write(TRAINCOLOR + 
                   f'Epoch {i_epoch:3d} (lr={current_lr:.2e}) -> Train overall accuracy: {cm_train.overall_accuracy():.2f}%, ' + 
                   f'Train mIoU: {mIoU:.2f}%, Train Loss: {loss_train:.4f}' + NORMALCOLOR)

        metrics_pn[i_epoch] = [[cm_train.overall_accuracy(), mIoU, loss_train]]

        # Validation
        cm_valid, loss_valid = eval_model(model, PCC, False, args=args, device=device, class_weights=class_weights)
        mIoU_valid = cm_valid.class_IoU(show=0)

        metrics_pn[i_epoch].append([cm_valid.overall_accuracy(), mIoU_valid, loss_valid])

        best_valid = False
        if mIoU_valid > best_mIoU:
            best_valid = True
            best_mIoU = mIoU_valid
            best_model = copy.deepcopy(model)
            tqdm.write(VALIDCOLOR + 
                       f'Best performance at epoch {i_epoch:3d} -> ' + 
                       f'Valid overall accuracy: {cm_valid.overall_accuracy():.2f}%, ' + 
                       f'Valid mIoU: {mIoU_valid:.2f}%, Valid Loss: {loss_valid:.4f}' + NORMALCOLOR)
        else:
            tqdm.write(VALIDCOLOR + 
                       f'Epoch {i_epoch:3d} -> Valid overall accuracy: {cm_valid.overall_accuracy():.2f}%, ' + 
                       f'Valid mIoU: {mIoU_valid:.2f}%, Valid Loss: {loss_valid:.4f}' + NORMALCOLOR)

        # Test evaluation
        if i_epoch == args.n_epoch - 1 or best_valid:
            cm_test, loss_test = eval_model(best_model, PCC, True, args=args, device=device, class_weights=class_weights)
            mIoU_test = cm_test.class_IoU(show=0)
            tqdm.write(TESTCOLOR + 
                       f'Epoch {i_epoch:3d} -> Test overall accuracy: {cm_test.overall_accuracy():.2f}%, ' + 
                       f'Test mIoU: {mIoU_test:.2f}%, Test Loss: {loss_test:.4f}' + NORMALCOLOR)
            
            metrics_pn[i_epoch].append([cm_test.overall_accuracy(), mIoU_test, loss_test])

    return best_model, metrics_pn

# 17. Prediction Visualization

def tile_prediction(tile_name, model=None, PCC=None, Visualization=True, features_used='xyzrgbi', file_type='ply'):
    """
    Predict and visualize a single tile's point cloud.
    """
    # Load the tile
    cloud, gt = cloud_loader(tile_name, features_used, file_type=file_type)
    
    # Make the predictions
    labels = PCC.run(model, [cloud])
    labels = labels.argmax(1).cpu().numpy()
    
    # Prepare the data for export
    xyz = cloud[:3, :].numpy().T
    
    # Prepare the data for Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Visualization with Open3D
    if Visualization:
        # Assign colors based on predicted labels
        colors = plt.get_cmap("tab10")(labels)
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        pcd.estimate_normals(fast_normal_computation=True)
        o3d.visualization.draw_geometries([pcd])
    
    return pcd, labels

def inference_with_tta(model, PCC, cloud, n_augmentations=5):
    predictions = []
    
    # Original prediction
    pred = PCC.run(model, [cloud])
    predictions.append(pred)
    
    # Test-time augmentations
    for _ in range(n_augmentations):
        # Apply random rotation and scaling
        augmented_cloud = random_rotation(cloud.clone())
        augmented_cloud = random_scaling(augmented_cloud)
        
        pred = PCC.run(model, [augmented_cloud])
        predictions.append(pred)
    
    # Average predictions
    final_pred = torch.stack(predictions).mean(0)
    return final_pred

# 18. Export the Model and Metrics

def export_results(trained_model, metrics_pn, project_dir):
    """
    Saves the trained model and metrics to disk.
    """
    # Save the model
    model_path = f'./pointnet_model2_{os.path.basename(project_dir)}.torch'
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save the metrics to a CSV file
    metrics_path = f"./metrics2_{os.path.basename(project_dir)}.csv"
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
    print(f"Metrics saved to {metrics_path}")

# 19. Main Execution

if __name__ == "__main__":
    
    # Start the training process
    t0 = time.time()
    trained_model, metrics_pn = train_full(args, device)
    t1 = time.time()

    print(trained_model)
    print('-' * 50)
    print(f"Total training time: {t1 - t0:.2f} seconds")
    print('=' * 50)

    # Example Prediction on a Test Tile
    if len(test_list) > 0:
        PCC = PointCloudClassifier(args, device=device)  # Initialize PCC after training
        selection = test_list[0]  # Ensure this index is within range
        pcd, labels = tile_prediction(selection, model=trained_model, PCC=PCC, file_type=file_type)
    else:
        print("No test files found. Skipping prediction visualization.")

    # Export the Model and Metrics
    export_results(trained_model, metrics_pn, project_dir)
