# =========================================================
# EfficientNet-B0 Facial Emotion Recognition (RAF-DB)
# =========================================================
from facenet_pytorch import MTCNN

import os, cv2, numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report



# -------------------------------
# Config
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

NAME_TO_TARGET = {
    "happy": 1,
    "sad": 0,
    "angry": 2,
    "surprise": 2,
    "fear": 0,
    "neutral": 3
}
TARGET_NAMES = {0: "Sad", 1: "Happy", 2: "Energetic", 3: "Calm"}


import torch.nn.functional as F

def predict_from_image(model, img, transform=None, threshold=0.6):
    """
    Predict emotion label from image.
    Returns (label, confidence) only if above threshold, else "Uncertain".
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    t = transform or get_transforms(train=False)
    xb = t(img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model(xb)
        probs = F.softmax(out, dim=1)
        conf, idx = torch.max(probs, dim=1)
        conf_val = conf.item()
        idx_val = idx.item()

    if conf_val >= threshold:
        label = TARGET_NAMES[idx_val]
        return f"{label} ({conf_val*100:.1f}%)"
    

# -------------------------------
# Face preprocessing util
# -------------------------------
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

from facenet_pytorch import MTCNN
mtcnn_detector = MTCNN(keep_all=False, device=DEVICE)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def rigid_preprocess(img_pil):
    """Crop face using Haar Cascade and normalize lighting."""
    img = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        img = img[y:y+h, x:x+w]
    else:
        # fallback: center crop
        h, w, _ = img.shape
        s = int(min(h, w) * 0.8)
        y1, x1 = (h - s) // 2, (w - s) // 2
        img = img[y1:y1 + s, x1:x1 + s]

    # Histogram equalization (light normalization)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    return Image.fromarray(img)

# -------------------------------
# Dataset
# -------------------------------
class RAFDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.items, self.transform = [], transform
        for label_name in os.listdir(root_dir):
            lpath = os.path.join(root_dir, label_name)
            if not os.path.isdir(lpath): continue
            tgt = NAME_TO_TARGET.get(label_name.lower())
            if tgt is None: continue
            for fn in os.listdir(lpath):
                if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.items.append((os.path.join(lpath, fn), tgt))
        if len(self.items) == 0:
            raise RuntimeError(f"No images in {root_dir}")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = rigid_preprocess(img)
        if self.transform: img = self.transform(img)
        return img, label

# -------------------------------
# Transforms (rigid + augment)
# -------------------------------
def get_transforms(train=True):
    if train:
        return T.Compose([
            T.Resize((256,256)),
            T.RandomResizedCrop(224, scale=(0.85,1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2,0.2,0.2,0.1),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

# -------------------------------
# Model: EfficientNet-B0
# -------------------------------
def get_model(num_classes=NUM_CLASSES, pretrained=True):
    m = models.efficientnet_b0(pretrained=pretrained)
    # Unfreeze last two blocks
    for name, param in m.features.named_parameters():
        if "6" in name or "7" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, num_classes)
    )
    return m.to(DEVICE)


def run_webcam(model):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    print("🎥 Webcam running... press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            label = predict_from_image(model, roi)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# Train / Evaluate
# -------------------------------
def run_epoch(model, loader, opt, loss_fn, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in tqdm(loader, desc="Train" if train else "Eval", leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if train: opt.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(xb)
            loss = loss_fn(out, yb)
            if train:
                loss.backward()
                opt.step()
        total_loss += loss.item()*xb.size(0)
        correct += (out.argmax(1)==yb).sum().item()
        total += xb.size(0)
    return total_loss/total, correct/total

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc = 0
    for e in range(1, epochs+1):
        tr_l,tr_a = run_epoch(model,train_loader,opt,loss_fn,True)
        vl_l,vl_a = run_epoch(model,val_loader,opt,loss_fn,False)
        scheduler.step(vl_l)
        print(f"[{e:02d}/{epochs}] train_loss={tr_l:.4f} acc={tr_a:.4f} | val_loss={vl_l:.4f} acc={vl_a:.4f}")
        if vl_a > best_acc:
            best_acc = vl_a
            torch.save(model.state_dict(), "best_condensed_effnetb0_rafdb.pth")
    print(f"✅ Best val accuracy: {best_acc:.4f}")
    return model

# -------------------------------
# Final test evaluation
# -------------------------------
def evaluate_on_test(model, test_root):
    print("\n🔍 Testing on final TEST set...")
    test_ds = RAFDataset(test_root, transform=get_transforms(False))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    loss_fn = nn.CrossEntropyLoss()
    y_true,y_pred = [],[]
    test_loss,correct,total=0,0,0
    model.eval()
    with torch.no_grad():
        for xb,yb in tqdm(test_loader, desc="Testing"):
            xb,yb=xb.to(DEVICE), yb.to(DEVICE)
            out=model(xb)
            loss=loss_fn(out,yb)
            test_loss+=loss.item()*xb.size(0)
            preds=out.argmax(1)
            correct+=(preds==yb).sum().item()
            total+=xb.size(0)
            y_true.extend(yb.cpu().numpy()); y_pred.extend(preds.cpu().numpy())
    acc=correct/total; print(f"✅ Test loss={test_loss/total:.4f}, acc={acc:.4f}")
    print(classification_report(y_true,y_pred,target_names=[TARGET_NAMES[i] for i in range(NUM_CLASSES)],zero_division=0))
    cm=confusion_matrix(y_true,y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
                xticklabels=[TARGET_NAMES[i] for i in range(NUM_CLASSES)],
                yticklabels=[TARGET_NAMES[i] for i in range(NUM_CLASSES)])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Final Confusion Matrix (Test)")
    plt.tight_layout(); plt.savefig("final_condensed_confusion_effnetb0.png"); plt.show()


IMAGE_PATHS = [
    r"C:\Softwares\VSCODE\PROGRAMS\Python\ML\Dataset\360_F_48883681_3YSVqKeyIvDNGZ9t0A8ynIFaeo64sHDm.jpg"
    
]

def run_on_images(model, image_paths):
    transform = get_transforms(train=False)
    for p in image_paths:
        if not os.path.isfile(p):
            print(f"Image not found: {p}")
            continue
        img = Image.open(p).convert("RGB")
        label = predict_from_image(model, img, transform=transform)
        print(f"{os.path.basename(p)} -> {label}")
# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_root = os.path.join(base_dir, "Dataset","New","train")
    val_root   = os.path.join(base_dir, "Dataset","New","val")
    test_root  = os.path.join(base_dir, "Dataset","New","test")

    train_ds = RAFDataset(train_root, get_transforms(True))
    val_ds   = RAFDataset(val_root,   get_transforms(False))
    print(f"Train={len(train_ds)} | Val={len(val_ds)}")

    train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
    val_loader   = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)

    model = get_model(pretrained=True)
    #model = train_model(model, train_loader, val_loader)
    model.load_state_dict(torch.load("fine_tunedme2_sad_effnetb0.pth", map_location=DEVICE))
    #evaluate_on_test(model, test_root)
    run_webcam(model) 
    #run_on_images(model, IMAGE_PATHS)