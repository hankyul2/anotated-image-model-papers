import sys
import time
import datetime
import volo

from pic.utils import setup, get_args_parser, save_checkpoint, resume_from_checkpoint
from pic.utils import print_metadata, Result
from pic.data import get_dataset, get_dataloader
from pic.model import get_model, get_ema_ddp_model
from pic.criterion import get_scaler_criterion
from pic.optimizer import get_optimizer_and_scheduler
from pic.use_case import validate, train_one_epoch


def run(args):
    # 0. init ddp & logger
    setup(args)

    # 1. load dataset
    train_dataset, valid_dataset = get_dataset(args)
    train_dataloader, valid_dataloader = get_dataloader(train_dataset, valid_dataset, args)

    # 2. make model
    model = get_model(args)
    model, ema_model, ddp_model = get_ema_ddp_model(model, args)

    # 3. load optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    # 4. load criterion
    criterion, valid_criterion, scaler = get_scaler_criterion(args)

    # 5. print metadata
    print_metadata(model, train_dataset, valid_dataset, args)

    # 6. control logic for checkpoint & validate
    if args.resume:
        start_epoch = resume_from_checkpoint(args.checkpoint_path, optimizer, scaler, scheduler)
    else:
        start_epoch = 0

    start_epoch = args.start_epoch if args.start_epoch else start_epoch
    end_epoch = args.end_epoch if args.end_epoch else args.epoch

    if scheduler is not None and start_epoch:
        # Todo: sequential lr does not support step with epoch as positional variable
        scheduler.step(start_epoch)

    if args.validate_only:
        validate(valid_dataloader, model, valid_criterion, args, 'org')
        if args.ema:
            validate(valid_dataloader, ema_model, valid_criterion, args, 'ema')
        return

    # 7. train
    best_epoch = 0
    best_acc = 0
    top1_list = []
    top5_list = []
    start_time = time.time()

    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(train_dataloader, ddp_model if args.distributed else model, optimizer, criterion, args, ema_model, scheduler, scaler, epoch)
        val_loss, top1, top5 = validate(valid_dataloader, ddp_model if args.distributed else model, valid_criterion, args, 'org')
        if args.ema:
            eval_ema_metric = validate(valid_dataloader, ema_model.module, valid_criterion, args, 'ema')

        if args.use_wandb:
            args.log({'train_loss':train_loss, 'val_loss':val_loss, 'top1':top1, 'top5':top5}, metric=True)

        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch
        top1_list.append(top1)
        top5_list.append(top5)

        if args.save_checkpoint and args.is_rank_zero:
            save_checkpoint(args.log_dir, model, ema_model, optimizer,
                            scaler, scheduler, epoch, is_best=best_epoch == epoch)

    # 8. summary train result in csv
    if args.is_rank_zero:
        best_acc = round(float(best_acc), 4)
        top1 = round(float(sum(top1_list[-3:]) / 3), 4)
        top5 = round(float(sum(top5_list[-3:]) / 3), 4)
        duration = str(datetime.timedelta(seconds=time.time() - start_time)).split('.')[0]
        Result(args.output_dir).save_result(args, top1_list, top5_list,
                                            dict(duration=duration, best_acc=best_acc, avg_top1_acc=top1, avg_top5_acc=top5))

    # 9. save model weight
    if args.save_last_epoch and args.is_rank_zero:
        save_checkpoint(args.log_dir, model, ema_model, optimizer, scaler, scheduler, end_epoch-1, is_best=False)


if __name__ == '__main__':
    config = "../data --dataset_type CIFAR100 --model-name volo_d1_224 --train-size 224 224 "\
             "--train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 "\
             "--center-crop-ptr 1.0 --interpolation bicubic --mean 0.4914 0.4825 0.4467 "\
             "--std 0.2471 0.2435 0.2616 --cutmix 1.0 --mixup 0.0 --remode 0.0 "\
             "--drop-path-rate 0.0 --smoothing 0.0 --epoch 300 --optimizer sgd --nesterov "\
             "--lr 0.25 --min-lr 1e-4 --weight-decay 1e-4 --warmup-epoch 5 --scheduler cosine "\
             "-b 128 -j 4 --pin-memory --amp --channels-last --cuda 8"
    args_parser = get_args_parser()
    args = args_parser.parse_args(config.split(" ") + sys.argv[1:])
    run(args)
