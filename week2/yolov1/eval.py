import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from yolov1 import Yolov1
from dataset import VOCDataset
from transform import Compose
from utils import(
    get_bboxes,
    mean_average_precision,
    non_max_suppression,
    plot_image,
    load_checkpoint
)


# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True
IMG_DIR = "images"
LABEL_DIR = "labels"
CHECKPOINT_FILE = "checkpoint_epoch_999.pth.tar"  # 학습된 모델 체크포인트 파일 경로

# Transform 정의 (train.py에서 사용한 transform과 동일하게 설정)
transform = Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

# 평가 함수
def evaluate(model, loader, iou_threshold, threshold):
    model.eval()  # 평가 모드로 전환
    pred_boxes, target_boxes = get_bboxes(
        loader, model, iou_threshold=iou_threshold, threshold=threshold, device=DEVICE
    )

    # mAP 계산
    map_value = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=iou_threshold, box_format="midpoint"
    )

    print(f"mAP: {map_value}")
    model.train()  # 다시 훈련 모드로 전환

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0)

    # 체크포인트 파일 로드
    load_checkpoint(torch.load(CHECKPOINT_FILE), model, optimizer)

    test_dataset = VOCDataset(
        "test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    # 모델 평가
    evaluate(model, test_loader, iou_threshold=0.5, threshold=0.4)

if __name__ == "__main__":
    main()
