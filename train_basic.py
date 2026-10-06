import argparse
import pathlib as pl
import time

import yaml
import torch
import numpy as np

from src.data import dataloader
from basic.resnet import ResNet50
from basic.convnext_v1 import convnext_tiny
from basic.convnext_v2 import convnextv2_tiny

def validation(args, data_conf, model, image_size, device, dtype, class_mapping=None):
    val_loader = dataloader.build_dataloader(args.data_name, 
                                            data_conf,
                                            data_split="val",
                                            image_size=image_size,
                                            batch_size=args.batch_size//2,
                                            num_workers=args.num_workers,
                                            class_mapping=class_mapping,
                                            )
    softmax = torch.nn.Softmax(dim=-1) 
    model.eval()

    total = 0
    correct = 0
    for (imgs, labels) in val_loader:
        imgs = imgs.to(device, dtype)
        preds = model(imgs)
        if type(preds) is tuple:
            preds = preds[0]
        preds = preds.detach().cpu()
        preds = softmax(preds)
        pred_label = torch.argmax(preds, dim=-1)
        total += len(labels)
        correct += torch.sum(pred_label==labels)

    if total == 0:
        return 0
    else:
        return correct / total

def main(args):
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "xpu" and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    data_type = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # load data config from the current folder
    with open("data_conf.yaml") as reader:
        data_conf = yaml.safe_load(reader)[args.data_name]

    image_size = args.image_size if args.image_size is not None else data_conf["image_size"][0]

    train_loader = dataloader.build_dataloader(args.data_name, 
                                            data_conf,
                                            data_split="train",
                                            image_size=image_size,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            dtype=data_type,
                                            )
    
    #model = build_model.build(args.model_name, data_conf["num_classes"], data_conf["image_size"])
    if args.model_name == "resnet":
        model = ResNet50(num_classes=data_conf["num_classes"])
    elif args.model_name == "convnext_tiny":
        model = convnext_tiny(num_classes=data_conf["num_classes"])
    elif args.model_name == "convnextv2_tiny":
        model = convnextv2_tiny(num_classes=data_conf["num_classes"])
    elif args.model_name == "lowres_convnext":
        from basic.lowres_convnext import convnext_tiny as lowres_convnext_tiny
        model = lowres_convnext_tiny(num_classes=data_conf["num_classes"])
    elif args.model_name == "lowres_convnextv2":
        from basic.lowres_convnextv2 import convnextv2_tiny as lowres_convnextv2_tiny
        model = lowres_convnextv2_tiny(num_classes=data_conf["num_classes"])

    model = model.to(device, data_type)
    print (f"Training a {args.model_name} model on {args.data_name} dataset using {device} with {data_type} data type.")

    if args.pre_val: 
        val_acc = validation(args, data_conf, model, image_size, device, data_type, train_loader.dataset.get_class_mapping())

    loss_func = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Observe that all parameters are being optimized
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # Decay LR by a factor of 0.1 every 7 epochs
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_step, gamma=args.gamma)
    
    lr = args.learning_rate * (args.batch_size / 512)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    if args.model_name in ['MobileNetV2', 'MobileNetV2_1_0', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_1_5']:
        args.learning_rate = 0.01
    
    exp_output_dir = args.output_dir / args.data_name / args.model_type / args.model_name
    if not exp_output_dir.is_dir():
        exp_output_dir.mkdir(parents=True)

    # Initialize SummaryWriter
    best_val_acc = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        epoch_start = time.time()

        for i, batch in enumerate(train_loader):
            if isinstance(batch, dict):
                imgs = batch["pixel_values"]
                labels = batch["label"]
            else:                
                imgs, labels = batch
            iter_start = time.time()
            optimizer.zero_grad()
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            if type(preds) is tuple:
                preds = preds[0]

            loss = loss_func(preds, labels)

            epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            iter_elapsed = time.time() - iter_start
            if args.print_freq > 0 and i % args.print_freq == 0:
                print (f"Epoch: {epoch}, Iteration: {i}, Progress: {i/len(train_loader):.2%}, Loss: {loss.item()}, Iter Elapsed: {iter_elapsed:.3f}")

        scheduler.step()

        epoch_elapsed = time.time () - epoch_start

        val_acc = validation(args, data_conf, model, image_size, device, data_type, train_loader.dataset.get_class_mapping())
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state = {"state_dict": model.state_dict()}
            torch.save(state, exp_output_dir/"best_epoch.pt")
        
        epoch_log = f"Epoch: {epoch+1}, Loss: {np.mean(epoch_loss)}, Elapsed: {epoch_elapsed:.3f} Val Accuracy: {val_acc:.3f}"
        
        print (epoch_log)
        with open(exp_output_dir / "log.txt", "a") as file:
            file.write(epoch_log + "\n")

        if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
            state = {"state_dict": model.state_dict()}
            torch.save(state, exp_output_dir/f"epoch_{epoch+1}.pt")

    torch.save(state, exp_output_dir/f"epoch_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="cifar100", choices=["tiny-imagenet", "cifar10", "cifar100", "imagewoof", "imagenet"], help="The path of the data root.")
    parser.add_argument("--model_type", type=str, default="teacher", help="The type of the model, either teacher or student.")
    parser.add_argument("--config_path", type=pl.Path, default="data_conf.yaml", help="The path of the data config file.")
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size.")
    parser.add_argument("--image_size", type=int, default=224, help="The image size.")
    parser.add_argument("--num_workers", type=int, default=8, help="The number of workers for the dataloader.")
    parser.add_argument("--model_name", type=str, default="resnet", help="The ViT model used for training.")
    parser.add_argument("--model_weight", type=str, default=None, help="The path of the model weight for initialization.")
    parser.add_argument("--output_dir", type=pl.Path, default="result", help="The path of the data root.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "xpu", "cpu"], help="The device used for training.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"], help="The data type of the input tensor.")

    parser.add_argument("--epochs", type=int, default=200, help="The number of epochs for training.")
    parser.add_argument("--learning_rate", type=float, default=4e-3, help="0.01 for MobileNet/ShuffleNet series architectures and 0.05 for other architectures.")
    parser.add_argument("--weight_decay", type=float, default=5e-2, help="Weight decay.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for the loss function.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the optimizer.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for the learning rate scheduler.")
    parser.add_argument("--decay_step", type=int, default=[150, 180, 210], help="Decay the LR.")
    parser.add_argument("--print_freq", type=int, default=0, help="The frequency for printing the log.")
    parser.add_argument("--save_freq", type=int, default=100, help="The frequency for saving the model.")
    parser.add_argument("--pre_val", action="store_true", help="Whether to perform validation before training.")

    args = parser.parse_args()
    main(args)
