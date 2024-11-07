import torch
import torch.nn.functional as F
import clip

from utils import *

from loralib.utils import (
    mark_only_lora_as_trainable,
    apply_lora,
    get_lora_parameters,
    save_lora,
    load_lora,
)
from loralib import layers as lora_layers
import random

import sys
import logging

class Logger:
    def __init__(self, log_file_path):
        # Configure logging
        self.logger = logging.getLogger('CustomLogger')
        self.logger.setLevel(logging.DEBUG)

        # Create a file handler to write logs to a file
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        # Create a stream handler to print to console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        # Set a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

logger = Logger('train_crossarea_vit-l.log')


def clip_loss(text_features, image_features, logit_scale):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()
    labels = torch.arange(logits_per_image.shape[0]).cuda()
    loss = (F.cross_entropy(logits_per_image, labels) + 
            F.cross_entropy(logits_per_text, labels)) / 2
    return loss

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()

        # Compare predictions with targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # Compute accuracy
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res  # Returns a list with accuracies for each k
    
def recall(output, target, topk=(1,)):
    """Computes the recall over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # Shape: [batch_size, maxk]
        pred = pred.t()  # Shape: [maxk, batch_size]

        # Initialize list to store recall values
        res = []
        for k in topk:
            # Check if the target is among the top k predictions
            correct = pred[:k].eq(target.view(1, -1).expand_as(pred[:k]))
            # For each sample, if any of the top-k predictions is correct, it's counted as a hit
            correct_k = correct.any(dim=0).float().sum(0)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res  # Returns a list with recall values for each k

def evaluate_lora(args, clip_model, loader, logit_scale, visualize=False, num_visualizations=5, iter=0):
    clip_model.eval()
    
    if visualize:
        vis_images = []
        vis_captions = []
        vis_top5_captions = []
        select_batch = random.randint(0, len(loader) - 1)
        current_batch = 0
        with torch.no_grad():
            for images, captions in loader:
                if current_batch != select_batch:
                    current_batch += 1
                    continue
                # Randomly select images to visualize from the batch
                batch_size = images.size(0)
                num_images_to_select = min(batch_size, num_visualizations - len(vis_images))
                indices = random.sample(range(batch_size), k=num_images_to_select)
                
                selected_images = images[indices].cuda()
                selected_captions = [captions[i] for i in indices]
                
                # Compute image features for selected images
                with torch.cuda.amp.autocast():
                    image_features = clip_model.encode_image(selected_images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Compute text features for all captions in the batch
                texts = clip.tokenize(captions, truncate=True).cuda()
                with torch.cuda.amp.autocast():
                    text_features = clip_model.encode_text(texts)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity between selected images and all captions in the batch
                logits_per_image = logit_scale * image_features @ text_features.t()
                
                # Get top-k captions for each selected image
                topk = 5
                topk_indices = logits_per_image.topk(topk, dim=1)[1].cpu()
                
                # Collect data for visualization
                for i in range(selected_images.size(0)):
                    vis_images.append(selected_images[i].cpu())
                    vis_captions.append(selected_captions[i])
                    topk_idx = topk_indices[i]
                    # Retrieve top-k captions from all captions in the batch
                    topk_captions = [captions[j] for j in topk_idx]
                    vis_top5_captions.append(topk_captions)
                    
                    # Break if we've collected enough samples
                    if len(vis_images) >= num_visualizations:
                        break
                
                if len(vis_images) >= num_visualizations:
                    break
        
        # Call the visualization function
        visualize_results(vis_images, vis_captions, vis_top5_captions, iter)
    else:
        # Metrics mode: compute loss, accuracy, and recall
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        total_samples = 0

        total_recall_i2t_at1 = 0.0
        total_recall_i2t_at5 = 0.0
        total_recall_t2i_at1 = 0.0
        total_recall_t2i_at5 = 0.0

        with torch.no_grad():
            for images, captions in loader:
                images = images.cuda()
                # Tokenize captions
                texts = clip.tokenize(captions, truncate=True).cuda()

                # Encode images and captions
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
                    text_features = clip_model.encode_text(texts)
                
                # Compute loss
                loss = clip_loss(text_features, image_features, logit_scale)
                total_loss += loss.item() * images.size(0)

                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute cosine similarity
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logit_scale * text_features @ image_features.t()

                # Ground truth labels (assumes matching pairs are at the same indices)
                ground_truth = torch.arange(len(images)).cuda()

                total_samples += images.size(0)
                
                # Compute accuracy
                acc1 = accuracy(logits_per_image, ground_truth, topk=(1,))[0]
                total_top1 += acc1 * images.size(0)

                acc5 = accuracy(logits_per_image, ground_truth, topk=(5,))[0]
                total_top5 += acc5 * images.size(0)

                # Compute recall
                recalls_i2t = recall(logits_per_image, ground_truth, topk=(1, 5))
                recalls_t2i = recall(logits_per_text, ground_truth, topk=(1, 5))

                total_recall_i2t_at1 += recalls_i2t[0] * images.size(0)
                total_recall_i2t_at5 += recalls_i2t[1] * images.size(0)
                total_recall_t2i_at1 += recalls_t2i[0] * images.size(0)
                total_recall_t2i_at5 += recalls_t2i[1] * images.size(0)

        # Compute average metrics
        avg_acc1 = total_top1 / total_samples
        avg_acc5 = total_top5 / total_samples
        avg_loss = total_loss / total_samples
        recall_i2t_at1 = total_recall_i2t_at1 / total_samples
        recall_i2t_at5 = total_recall_i2t_at5 / total_samples
        recall_t2i_at1 = total_recall_t2i_at1 / total_samples
        recall_t2i_at5 = total_recall_t2i_at5 / total_samples

        # Print the metrics
        # print(f"Validation Loss: {avg_loss:.4f}")
        # print(f"Top-1 Accuracy: {avg_acc1.item() * 100:.2f}%")
        # print(f"Top-5 Accuracy: {avg_acc5.item() * 100:.2f}%")
        # print(f"Image-to-Text Recall@1: {recall_i2t_at1 * 100:.2f}%")
        # print(f"Image-to-Text Recall@5: {recall_i2t_at5 * 100:.2f}%")
        # print(f"Text-to-Image Recall@1: {recall_t2i_at1 * 100:.2f}%")
        # print(f"Text-to-Image Recall@5: {recall_t2i_at5 * 100:.2f}%")

        logger.info(f"Validation Loss: {avg_loss:.4f}")
        logger.info(f"Top-1 Accuracy: {avg_acc1.item() * 100:.2f}%")
        logger.info(f"Top-5 Accuracy: {avg_acc5.item() * 100:.2f}%")
        logger.info(f"Image-to-Text Recall@1: {recall_i2t_at1 * 100:.2f}%")
        logger.info(f"Image-to-Text Recall@5: {recall_i2t_at5 * 100:.2f}%")
        logger.info(f"Text-to-Image Recall@1: {recall_t2i_at1 * 100:.2f}%")
        logger.info(f"Text-to-Image Recall@5: {recall_t2i_at5 * 100:.2f}%")


def visualize_results(images_list, captions_list, top5_captions_list, iter):
    """Visualizes images with top 5 retrieved captions."""
    import matplotlib.pyplot as plt
    import textwrap
    import torch
    from torchvision import transforms

    num_visualizations = len(images_list)
    for idx in range(num_visualizations):
        image = images_list[idx]
        caption = captions_list[idx]
        top5_captions = top5_captions_list[idx]

        # Convert image tensor to PIL image
        inv_normalize = transforms.Normalize(
            mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
            std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711]
        )
        image = inv_normalize(image)
        image = torch.clamp(image, 0, 1)
        image = transforms.ToPILImage()(image)

        # Create a wider layout with a larger width for the image
        fig, (ax_image, ax_text) = plt.subplots(1, 2, figsize=(13, 8), gridspec_kw={'width_ratios': [2, 3]})
        ax_image.imshow(image)
        ax_image.axis('off')

        # Prepare captions for display
        caption_wrapped = textwrap.fill(f"Ground Truth:\n{caption}", width=60)
        top5_captions_wrapped = "\n\n".join(
            [f"{j+1}. {textwrap.fill(top5_captions[j], width=60)}"
             for j in range(len(top5_captions))]
        )

        # Display captions on the right with top alignment
        ax_text.text(0, 1, f"{caption_wrapped}\n\nTop 5 Retrieved Captions:\n\n{top5_captions_wrapped}",
                     va='top', fontsize=10, wrap=True)
        ax_text.axis('off')

        plt.tight_layout()
        plt.savefig(f"/home/erzurumlu.1/yunus/git/CLIP-LoRA/vis/vit-l/visualization_{iter}_{idx}.png", bbox_inches='tight')

def run_lora(args, clip_model, logit_scale, train_loader, val_loader):

    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()

    mark_only_lora_as_trainable(clip_model)

    total_iters = 80000

    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    

    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    
    while count_iters < total_iters:
        clip_model.train()
        total_loss = 0.

        for images, captions in train_loader:
            images = images.cuda()
            texts = clip.tokenize(captions, truncate=True).cuda()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                image_features = clip_model.encode_image(images)
                text_features = clip_model.encode_text(texts)

                loss = clip_loss(text_features, image_features, logit_scale)
            if count_iters  == 1:
                evaluate_lora(args, clip_model, val_loader,logit_scale)
                evaluate_lora(args, clip_model, val_loader,logit_scale, visualize=True, iter=count_iters)

                clip_model.train()
            # loss.backward()
            # optimizer.step()
            # scheduler.step()
                        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            total_loss += loss.item()
            count_iters += 1
            
            if count_iters % 1 == 0:
                # print(f"Iteration: {count_iters}, Loss: {loss.item()}")
                logger.info(f"Iteration: {count_iters}, Loss: {loss.item()}")
            if count_iters == total_iters:
                finish = True
                break
            
            if count_iters % 500 == 0:
                evaluate_lora(args, clip_model, val_loader,logit_scale, iter=count_iters)
                clip_model.train()
                save_lora(args,list_lora_layers, count_iters)
                logger.info("**** LoRA model saved. ****\n")

            if count_iters % 1000 == 0:
                evaluate_lora(args, clip_model, val_loader,logit_scale, visualize=True, iter=count_iters)
                # print("**** LoRA model saved. ****\n")
                clip_model.train()

        # print(f"Iteration: {count_iters}, Epoch Loss: {total_loss / tot_samples if tot_samples > 0 else 0}")
        logger.info(f"Iteration: {count_iters}, Epoch Loss: {total_loss / len(train_loader) if len(train_loader) > 0 else 0}")
        if finish:
            break


