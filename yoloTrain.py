from ultralytics import YOLO
import torch
from pathlib import Path

def train_yolo(
        data_yaml_path, 
        model_size="yolo11n", 
        device="cuda" if torch.cuda.is_available() else "cpu",
        project_name="AircraftClassification",
        experiment_name="yolo_train",
        resume=True,
        pretrained=True,
        ):
    
    keepClass = [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,17,18,19,21,22,32,33,36,41]

    if pretrained:
        model = YOLO(r'runs\detect\AircraftClassification\yolo_train\weights\best.pt')
    else:
        model = YOLO(model_size + ".yaml")
    args = {
        'data': data_yaml_path,
        'epochs': 40,
        'imgsz': 640,
        'batch': 28,
        'device': device,
        'workers': 2,

        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'classes': keepClass,
        'amp': True,
        'fraction': 1.0,
        'cache': False,
        'patience': 10,
        'save': True,
        'save_period': 10,
        'exist_ok': True,
        'pretrained': pretrained,

        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,

        'project': project_name,
        'name': experiment_name,
        'resume': resume,
        'verbose': True,
        'seed': 42,
    }

    try:
        results = model.train(**args)
        return model, results
    except Exception as e:
        print(f"Error during training: {e}")
        return None, None
    
def validate_model(model, data_yaml_path):
    print("\n" + "="*60)
    print("VALIDATING MODEL")
    print("="*60)
    
    metrics = model.val(data=data_yaml_path)
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics

def export_model(model, format):
    print("\n Exporting model...")
    export_path = model.export(format=format)
    print(f"\nModel exported to: {export_path}")
    return export_path

if __name__ == "__main__":
    data_yaml = r"E:\Završni rad\yolo_dataset\data.yaml"
    if not Path(data_yaml).exists():
        print(f"\nData YAML file not found at: {data_yaml}")
        exit(1)

    model, results = train_yolo(
        data_yaml_path=data_yaml
    )
    if model:
        metrics = validate_model(model, data_yaml)
        
        export_model(model, format='engine')
    print("\nTraining complete")