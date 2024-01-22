import os
import time
import json
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed.nn
import torch.distributed as dist

from uniclip.clip.model import convert_state_dict


def is_master(args):
    return args.rank == 0



# 计算损失
def get_loss(model, images, texts, ocrs, ocr_existss, fake_ocrs, ocr_contents, loss_factory, args, accum_image_features=None, accum_text_features=None, accum_idx=-1):
    if args.accum_freq == 1:
        #image_features, text_features, logit_scale = model(images, texts, args.mask_ratio)
        # ----------------------- tiny bert related code ------------------------
        if args.ocr_content:
            clip_out = model(images, texts, fake_ocrs, args.mask_ratio)
        else:
            clip_out = model(images, texts, args.mask_ratio)
        image_features = clip_out.get('image_features')
        text_features = clip_out.get('text_features')

        # tiny_image_features = clip_out.get('tiny_image_features')
        # tiny_text_features = clip_out.get('tiny_text_features')
        logit_scale = clip_out.get('logit_scale')

        if args.ocr_exist:
            ocr_exist_est = clip_out.get('ocr_exist_est')
        if args.ocr_content:
            ocr_content_est = clip_out.get('ocr_content_est')

        # -----------------------------------------------------------------------
        # print("cc_log_final_image_features: "+str(image_features.shape)) # bs*128
        # print("cc_log_final_text_features: "+str(text_features.shape)) # bs*128
    else:
        assert accum_image_features and accum_text_features and accum_idx != -1
        chunk_image_features, chunk_text_features, logit_scale = model(images, texts, args.mask_ratio)
        image_features = torch.cat(
            accum_image_features[:accum_idx] + [chunk_image_features] + accum_image_features[accum_idx + 1:])
        text_features = torch.cat(
            accum_text_features[:accum_idx] + [chunk_text_features] + accum_text_features[accum_idx + 1:])
    
    logit_scale = logit_scale.mean()

    if args.aggregate: # default: True
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        if args.gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]

            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)

            all_image_features = torch.cat(
                [image_features]
                + gathered_image_features[:rank]
                + gathered_image_features[rank + 1 :]
            )
            all_text_features = torch.cat(
                [text_features]
                + gathered_text_features[:rank]
                + gathered_text_features[rank + 1 :]
            )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()


    ground_truth = torch.arange(len(logits_per_image)).long()
    ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)

    #total_loss = (
    #    loss_img(logits_per_image, ground_truth)
    #    + loss_txt(logits_per_text, ground_truth)
    #) / 2

    loss_img_txt = (
        loss_factory['img2txt'](logits_per_image, ground_truth)
        + loss_factory['txt2img'](logits_per_text, ground_truth)
    ) / 2

    if args.ocr_exist and not args.ocr_content:
        loss_ocr_est = loss_factory['ocr_exist'](ocr_exist_est, ocr_existss)
        total_loss = 0.8 * loss_img_txt + 0.2 * loss_ocr_est
        loss_dict = {'total_loss': total_loss,
                    'img_txt': loss_img_txt,
                    'ocr_exist': loss_ocr_est
                    }
    elif args.ocr_exist and args.ocr_content:

        loss_ocr_est = loss_factory['ocr_exist'](ocr_exist_est, ocr_existss)
        loss_ocr_content = loss_factory['ocr_content'](ocr_content_est, ocr_contents)

        # 固定权重
        total_loss = 0.8 * loss_img_txt + 0.1 * loss_ocr_est + 0.1 * loss_ocr_content

        loss_dict = {'total_loss': total_loss,
                    'img_txt': loss_img_txt,
                    'ocr_exist': loss_ocr_est,
                    'ocr_content': loss_ocr_content
                    }
    elif not args.ocr_exist and args.ocr_content:
        loss_ocr_content = loss_factory['ocr_content'](ocr_content_est, ocr_contents)

        # 固定权重
        total_loss = 0.8 * loss_img_txt + 0.2 * loss_ocr_content

        loss_dict = {'total_loss': total_loss,
                    'img_txt': loss_img_txt,
                    'ocr_content': loss_ocr_content
                    }        
    else:
        total_loss = loss_img_txt
        loss_dict = {'total_loss': total_loss,
                    'img_txt': loss_img_txt,
                    }

    acc = None
    #if args.report_training_batch_acc:
    #    i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
    #    t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
    #    acc = {"i2t": i2t_acc, "t2i": t2i_acc}

    #return total_loss, acc
    
    if args.report_training_batch_acc:
        i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
        t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)

        if args.ocr_exist and not args.ocr_content:
            ocr_exist_acc = (ocr_exist_est.argmax(-1) == ocr_existss).sum() / len(ocr_exist_est)
            acc = {"i2t": i2t_acc, "t2i": t2i_acc, 'ocr_exist': ocr_exist_acc}
        elif args.ocr_exist and args.ocr_content:
            ocr_exist_acc = (ocr_exist_est.argmax(-1) == ocr_existss).sum() / len(ocr_exist_est)
            ocr_content_acc = (ocr_content_est.argmax(-1) == ocr_contents).sum() / len(ocr_content_est)
            acc = {"i2t": i2t_acc, "t2i": t2i_acc, 'ocr_exist': ocr_exist_acc, 'ocr_content': ocr_content_acc}
        elif not args.ocr_exist and args.ocr_content:
            ocr_content_acc = (ocr_content_est.argmax(-1) == ocr_contents).sum() / len(ocr_content_est)
            acc = {"i2t": i2t_acc, "t2i": t2i_acc,  'ocr_content': ocr_content_acc}
        else:
            acc = {"i2t": i2t_acc, "t2i": t2i_acc}
    
    return loss_dict, acc


def freeze_vision_text(args, model):
    if args.freeze_vision:
        for name, parameter in model.named_parameters():
            #print(name)
            if '.visual.' in name:
                parameter.requires_grad = False
        logging.info("Freeze image encoder!")
    
    if args.freeze_text:
        for name, parameter in model.named_parameters():        
            if '.bert.' in name:
                parameter.requires_grad = False
        logging.info("Freeze text encoder!")
    

class KLDiversity(nn.Module):
    def __init__(self, temperature=10):
        super().__init__()
        self.T = temperature
        self.kld = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits, teacher_logits):
        kld_loss = self.kld(nn.functional.log_softmax(logits / self.T, dim=1),
                            nn.functional.softmax(teacher_logits / self.T, dim=1))
        return kld_loss

class SoftmaxCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, y):
        logits = nn.functional.softmax(logits, dim=-1)
        loss = -(y * torch.log(logits)).sum(dim=-1)
        loss = loss.mean()
        return loss

def train(model, data, epoch, optimizer, scaler, scheduler, args, global_trained_steps):
    # os.environ["WDS_EPOCH"] = str(epoch)

    model.train()
    if args.freeze_vision or args.freeze_text:
        freeze_vision_text(args, model)

    dataloader, sampler = data['train'].dataloader,  data['train'].sampler

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    
    # loss_tiny_img = nn.CrossEntropyLoss()
    # loss_tiny_txt = nn.CrossEntropyLoss()
    # loss_kld = KLDiversity(temperature=10)

    loss_img = loss_img.cuda(args.local_device_rank)
    loss_txt = loss_txt.cuda(args.local_device_rank)

    if args.ocr_content:
        loss_ocr_content = nn.CrossEntropyLoss()
        loss_ocr_content = loss_ocr_content.cuda(args.local_device_rank)

    if args.ocr_exist:
        loss_ocr_exist = nn.CrossEntropyLoss()
        loss_ocr_exist = loss_ocr_exist.cuda(args.local_device_rank)
    # loss_tiny_img = loss_tiny_img.cuda(args.local_device_rank)
    # loss_tiny_txt = loss_tiny_txt.cuda(args.local_device_rank)
    # loss_kld = loss_kld.cuda(args.local_device_rank)

    # loss_factory = {'img2txt': loss_img, 'txt2img': loss_txt,
    #                 'tiny_img2txt': loss_tiny_img, 'tiny_txt2img': loss_tiny_txt,
    #                 'kld': loss_kld
    #                 }
    if args.ocr_exist and not args.ocr_content:
        loss_factory = {'img2txt': loss_img, 'txt2img': loss_txt, 'ocr_exist':loss_ocr_exist}
    elif args.ocr_exist and args.ocr_content:
        loss_factory = {'img2txt': loss_img, 'txt2img': loss_txt, 'ocr_exist':loss_ocr_exist, 'ocr_content':loss_ocr_content}
    elif not args.ocr_exist and args.ocr_content:
        loss_factory = {'img2txt': loss_img, 'txt2img': loss_txt, 'ocr_content':loss_ocr_content}
    else:
        loss_factory = {'img2txt': loss_img, 'txt2img': loss_txt}

    if sampler is not None:
        sampler.set_epoch(epoch)

    num_steps_per_epoch = dataloader.num_batches // args.accum_freq
    data_iter = iter(dataloader)

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

    end = time.time()
    epoch_trained_steps = 0
    for i in range(0, dataloader.num_batches):
        batch = next(data_iter)

        i_accum = i // args.accum_freq
        step = num_steps_per_epoch * epoch + i_accum
        # reach the args.max_steps, exit training:
        if step >= args.max_steps:
            logging.info("Stopping training due to step {} has reached max_steps {}".format(step, args.max_steps // args.accum_freq))
            return epoch_trained_steps
        scheduler(step)

        optimizer.zero_grad()

        images, querys, ocrs, ocr_existss, fake_ocrs, ocr_contents, eos_indices = batch

        images = images.cuda(args.local_device_rank, non_blocking=True)
        querys = querys.cuda(args.local_device_rank, non_blocking=True)
        
        ocrs = ocrs.cuda(args.local_device_rank, non_blocking=True)
        ocr_existss = ocr_existss.cuda(args.local_device_rank, non_blocking=True)
        eos_indices = eos_indices.cuda(args.local_device_rank, non_blocking=True)
        
        fake_ocrs = fake_ocrs.cuda(args.local_device_rank, non_blocking=True)
        ocr_contents = ocr_contents.cuda(args.local_device_rank, non_blocking=True)

        data_time = time.time() - end

        m = model.module

        if args.accum_freq == 1:
            # with automatic mixed precision.
            if args.precision == "amp":
                with autocast():
                    #total_loss, acc = get_loss(model, images, texts, loss_img, loss_txt, args)
                    loss_dict, acc = get_loss(model, images, querys, ocrs, ocr_existss, fake_ocrs, ocr_contents, loss_factory, args)
                    total_loss = loss_dict['total_loss']
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                scaler.update()
            else:
                exit()
                total_loss, acc = get_loss(model, images, querys, ocrs, ocr_existss, fake_ocrs, ocr_contents, loss_img, loss_txt, args)
                total_loss.backward()
                optimizer.step()
        else:
            # not used
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast(enabled=(args.precision == "amp")):
                    chunk_image_features, chunk_text_features, _ = model(images, texts)
                accum_image_features.append(chunk_image_features)
                accum_text_features.append(chunk_text_features)

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast(enabled=(args.precision == "amp")):
                    # `total_loss` and `acc` are coarsely sampled, taking only the last result in the loop.
                    # Although each result should be the same in theory, it will be slightly different in practice
                    total_loss, acc = get_loss(model, images, texts, loss_factory, args, accum_image_features, accum_text_features, j)
                if args.precision == "amp":
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

            if args.precision == "amp":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        epoch_trained_steps += 1

        if is_master(args) and ((step + 1) % args.log_interval) == 0:
            batch_size = len(images) * args.accum_freq
            num_samples = (i_accum + 1) * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * (i_accum + 1) / num_steps_per_epoch

            if args.ocr_exist and not args.ocr_content:
                logging.info(
                    f"Global Steps: {step + 1}/{args.max_steps} | " +
                    f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
                    f"Loss: {total_loss.item():.6f} | " +
                    f"Loss_ocr_est: {loss_dict['ocr_exist'].item():.6f} | " +
                    f"Loss_Img_Txt: {loss_dict['img_txt'].item():.6f} | " +
                    (f"Image2Text Acc: {acc['i2t'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                    (f"Text2Image Acc: {acc['t2i'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                    (f"OCR_exist_est Acc: {acc['ocr_exist'].item() * 100:.2f} | ") +
                    f"Data Time: {data_time:.3f}s | " +
                    f"Batch Time: {batch_time:.3f}s | " +
                    f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
                    f"logit_scale: {m.logit_scale.data:.3f} | " +
                    f"Global Batch Size: {batch_size * args.world_size}"
                )
            elif args.ocr_exist and args.ocr_content:
                logging.info(
                    f"Global Steps: {step + 1}/{args.max_steps} | " +
                    f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
                    f"Loss: {total_loss.item():.6f} | " +
                    f"Loss_ocr_exist: {loss_dict['ocr_exist'].item():.6f} | " +
                    f"Loss_ocr_content: {loss_dict['ocr_content'].item():.6f} | " +
                    f"Loss_Img_Txt: {loss_dict['img_txt'].item():.6f} | " +
                    (f"Image2Text Acc: {acc['i2t'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                    (f"Text2Image Acc: {acc['t2i'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                    (f"OCR_exist_est Acc: {acc['ocr_exist'].item() * 100:.2f} | ") +
                    (f"OCR_content_est Acc: {acc['ocr_content'].item() * 100:.2f} | ") +
                    f"Data Time: {data_time:.3f}s | " +
                    f"Batch Time: {batch_time:.3f}s | " +
                    f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
                    f"logit_scale: {m.logit_scale.data:.3f} | " +
                    f"Global Batch Size: {batch_size * args.world_size}"
                )
            elif not args.ocr_exist and args.ocr_content:
                logging.info(
                    f"Global Steps: {step + 1}/{args.max_steps} | " +
                    f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
                    f"Loss: {total_loss.item():.6f} | " +
                    f"Loss_ocr_content: {loss_dict['ocr_content'].item():.6f} | " +
                    f"Loss_Img_Txt: {loss_dict['img_txt'].item():.6f} | " +
                    (f"Image2Text Acc: {acc['i2t'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                    (f"Text2Image Acc: {acc['t2i'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                    (f"OCR_content_est Acc: {acc['ocr_content'].item() * 100:.2f} | ") +
                    f"Data Time: {data_time:.3f}s | " +
                    f"Batch Time: {batch_time:.3f}s | " +
                    f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
                    f"logit_scale: {m.logit_scale.data:.3f} | " +
                    f"Global Batch Size: {batch_size * args.world_size}"
                )                      
            else:
                logging.info(
                    f"Global Steps: {step + 1}/{args.max_steps} | " +
                    f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
                    f"Loss: {total_loss.item():.6f} | " +
                    f"Loss_Img_Txt: {loss_dict['img_txt'].item():.6f} | " +
                    (f"Image2Text Acc: {acc['i2t'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                    (f"Text2Image Acc: {acc['t2i'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                    f"Data Time: {data_time:.3f}s | " +
                    f"Batch Time: {batch_time:.3f}s | " +
                    f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
                    f"logit_scale: {m.logit_scale.data:.3f} | " +
                    f"Global Batch Size: {batch_size * args.world_size}"
                )


        if args.val_data is not None and args.valid_step_interval is not None and ((step + 1) % args.valid_step_interval) == 0:
            assert "val" in data, "Error: Valid dataset has not been built."
            if not args.use_flash_attention:
                evaluate(model, data, epoch, args, step + 1)
            else:
                # fp16 is needed in flash attention
                with autocast():
                    evaluate(model, data, epoch, args, step + 1)
            # set model back to train mode
            model.train()

            # if args.freeze_vision or args.freeze_text:
            #     freeze_vision_text(args, model)

        if args.should_save and args.save_step_frequency > 0 and ((step + 1) % args.save_step_frequency) == 0:
            save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_{step + 1}.pt")
            t1 = time.time()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1, step + 1, time.time() - t1))

            # Save the latest params
            t1 = time.time()
            save_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1, step + 1, time.time() - t1))
        
    return epoch_trained_steps


def evaluate(model, data, epoch, args, steps):

    logging.info("Begin to eval on validation set (epoch {} @ {} steps)...".format(epoch + 1, steps))

    model.eval()

    dataloader = data['val'].dataloader
    data_iter = iter(dataloader)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.cuda(args.local_device_rank)
    loss_txt = loss_txt.cuda(args.local_device_rank)

    cumulative_loss = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    cumulative_i2t_acc = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    cumulative_t2i_acc = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)

    if args.ocr_exist:
        ocr_exist_correct = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    if args.ocr_content:
        ocr_content_correct = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)

    num_elements = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for i in range(dataloader.num_batches):
            batch = next(data_iter)
            images, texts, ocrs, ocr_existss, fake_ocrs, ocr_contents, eos_indices = batch

            images = images.cuda(args.local_device_rank, non_blocking=True)
            texts = texts.cuda(args.local_device_rank, non_blocking=True)
            ocrs = ocrs.cuda(args.local_device_rank, non_blocking=True)
            ocr_existss = ocr_existss.cuda(args.local_device_rank, non_blocking=True)
            eos_indices = eos_indices.cuda(args.local_device_rank, non_blocking=True)
            fake_ocrs = fake_ocrs.cuda(args.local_device_rank, non_blocking=True)
            ocr_contents = ocr_contents.cuda(args.local_device_rank, non_blocking=True)

            if args.ocr_exist and not args.ocr_content:
                image_features, text_features, logit_scale, ocr_exist_est = model(images, texts)
                _, predicted = torch.max(ocr_exist_est, dim=1)
                ocr_exist_correct += ((predicted == ocr_existss).sum()).float()
            elif args.ocr_exist and args.ocr_content:
                image_features, text_features, logit_scale, ocr_exist_est, ocr_content_est = model(images, texts, fake_ocr)
                
                _, exist_predicted = torch.max(ocr_exist_est, dim=1)
                ocr_exist_correct += ((exist_predicted == ocr_existss).sum()).float()

                _, content_predicted = torch.max(ocr_content_est, dim=1)
                ocr_content_correct += ((content_predicted == ocr_contents).sum()).float()
            elif not args.ocr_exist and args.ocr_content:
                image_features, text_features, logit_scale, ocr_content_est = model(images, texts, fake_ocr)
                
                _, content_predicted = torch.max(ocr_content_est, dim=1)
                ocr_content_correct += ((content_predicted == ocr_contents).sum()).float()
            else:
                image_features, text_features, logit_scale = model(images, texts)
            

            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logit_scale = logit_scale.mean()

            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(images)).long()
            ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)
            total_loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_txt(logits_per_text, ground_truth)
            ) / 2

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

            cumulative_i2t_acc += ((logits_per_image.argmax(-1) == ground_truth).sum()).float()
            cumulative_t2i_acc += (logits_per_text.argmax(-1) == ground_truth).sum().float()

            if (i + 1) % 100 == 0:
                logging.info("Evaluated {}/{} batches...".format(i + 1, dataloader.num_batches))

        dist.all_reduce(cumulative_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(cumulative_i2t_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(cumulative_t2i_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_elements, op=dist.ReduceOp.SUM)

        if args.ocr_exist:
            dist.all_reduce(ocr_exist_correct, op=dist.ReduceOp.SUM)
            ocr_acc = ocr_exist_correct / num_elements
        if args.ocr_content:
            dist.all_reduce(ocr_content_correct, op=dist.ReduceOp.SUM)
            content_acc = ocr_content_correct / num_elements

        loss = cumulative_loss / num_elements
        i2t_acc = cumulative_i2t_acc / num_elements
        t2i_acc = cumulative_t2i_acc / num_elements
        

        assert num_elements.item() == dataloader.num_samples # sanity check


        if args.ocr_exist and not args.ocr_content:
            logging.info(
                f"Validation Result (epoch {epoch + 1} @ {steps} steps) | "
                f"Valid Loss: {loss.item():.6f} | "
                f"Image2Text Acc: {i2t_acc.item() * 100:.2f} | " 
                f"Text2Image Acc: {t2i_acc.item() * 100:.2f} | " 
                f"logit_scale: {model.module.logit_scale.data:.3f} | "
                f"ocr_acc: {ocr_acc.item() * 100:.2} | "
                f"Valid Batch Size: {batch_size}"
            )
        elif args.ocr_exist and args.ocr_content:
            logging.info(
                f"Validation Result (epoch {epoch + 1} @ {steps} steps) | "
                f"Valid Loss: {loss.item():.6f} | "
                f"Image2Text Acc: {i2t_acc.item() * 100:.2f} | " 
                f"Text2Image Acc: {t2i_acc.item() * 100:.2f} | " 
                f"logit_scale: {model.module.logit_scale.data:.3f} | "
                f"ocr_exist_acc: {ocr_acc.item() * 100:.2} | "
                f"ocr_content_acc: {content_acc.item() * 100:.2} | "
                f"Valid Batch Size: {batch_size}"
            )
        elif not args.ocr_exist and args.ocr_content:
            logging.info(
                f"Validation Result (epoch {epoch + 1} @ {steps} steps) | "
                f"Valid Loss: {loss.item():.6f} | "
                f"Image2Text Acc: {i2t_acc.item() * 100:.2f} | " 
                f"Text2Image Acc: {t2i_acc.item() * 100:.2f} | " 
                f"logit_scale: {model.module.logit_scale.data:.3f} | "
                f"ocr_content_acc: {content_acc.item() * 100:.2} | "
                f"Valid Batch Size: {batch_size}"
            )
        else:
            logging.info(
                f"Validation Result (epoch {epoch + 1} @ {steps} steps) | "
                f"Valid Loss: {loss.item():.6f} | "
                f"Image2Text Acc: {i2t_acc.item() * 100:.2f} | " 
                f"Text2Image Acc: {t2i_acc.item() * 100:.2f} | " 
                f"logit_scale: {model.module.logit_scale.data:.3f} | "
                f"Valid Batch Size: {batch_size}"
            )