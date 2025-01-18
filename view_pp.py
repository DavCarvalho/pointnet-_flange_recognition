import os
import numpy as np
import torch
import open3d as o3d
from plyfile import PlyData
import torch.nn as nn

# Importar o modelo PointNet2Classifier e as funções necessárias
from teste5 import PointNet2Classifier  # Importe o PointNet2Classifier do seu código
from teste5 import load_ply, normalize_point_cloud

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device: ", device)

def load_model(model_path, num_classes=2):
    model = PointNet2Classifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def prepare_new_data(file_path, num_points=1024):
    # Carregar e preparar o dado para inferência
    coords, _ = load_ply(file_path)
    coords = coords.astype(np.float32)

    # Normalização
    coords = normalize_point_cloud(coords)

    # Ajuste do número de pontos
    if coords.shape[0] > num_points:
        choice = np.random.choice(coords.shape[0], num_points, replace=False)
        coords = coords[choice, :]
    elif coords.shape[0] < num_points:
        choice = np.random.choice(coords.shape[0], num_points - coords.shape[0], replace=True)
        coords = np.concatenate([coords, coords[choice, :]], axis=0)

    # Converter para tensor e adicionar dimensão batch
    coords = torch.from_numpy(coords).float()  # (N, 3)
    coords = coords.unsqueeze(0)  # (1, N, 3)
    return coords

def run_inference(model, xyz):
    xyz = xyz.to(device)  # (1, N, 3)
    with torch.no_grad():
        out = model(xyz)  # Output shape: (1, num_classes)
        _, pred_label = torch.max(out, dim=1)
        pred_label = pred_label.cpu().numpy()[0]
    return pred_label

def visualize_prediction(xyz, prediction, class_names):
    coords = xyz.squeeze(0).cpu().numpy()  # (N, 3)

    # Visualização com Open3D
    class_colors = {
        0: [0, 0, 1],   # Azul para 'Non-Flange'
        1: [1, 0, 0],   # Vermelho para 'Flange'
    }
    color = class_colors.get(prediction, [0.5, 0.5, 0.5])
    colors = np.tile(color, (coords.shape[0], 1))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"Predicted: {class_names[prediction]}")
    o3d.visualization.draw_geometries([pcd])

def main():
    model_path = 'best_pointnet2_model.pth'  # Certifique-se de usar o caminho correto para o modelo salvo
    class_names = ['Non-Flange', 'Flange']

    # Carregar o modelo treinado
    model = load_model(model_path, num_classes=len(class_names))

    # Diretório com os novos dados para inferência
    new_data_dir = './DATA/inference'
    for file_name in os.listdir(new_data_dir):
        if file_name.endswith('.ply'):
            file_path = os.path.join(new_data_dir, file_name)
            xyz = prepare_new_data(file_path)
            prediction = run_inference(model, xyz)
            print(f"File: {file_name}")
            visualize_prediction(xyz, prediction, class_names)

if __name__ == "__main__":
    main()
