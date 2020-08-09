import os
import math
import argparse
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

from model import MatchModel
from utils import set_seed, load_vocab, load_dataset, build_vocab, model_init


logger = logging.getLogger(__name__)


def train(args, train_dataset, model, tokenizer, word2id):

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = range(
        epochs_trained, int(args.num_train_epochs)
    )

    # set_seed(args)  # Added here for reproductibility

    loss_show = []

    best_acc = 0

    for epoch, _ in enumerate(train_iterator):
        epoch_loss = 0.0


        epoch_iterator = train_dataloader
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids_a": batch[0], "attention_mask_a": batch[1],
                      "input_ids_b": batch[2], "attention_mask_b": batch[3], "labels": batch[4]}

            outputs = model(**inputs)
            # outputs = model(batch[0], batch[1], batch[2], batch[3], batch[4])
            loss = outputs[0]  # model outputs are always tuple

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            epoch_loss += loss.item()

            loss_show.append(loss.item())
            if (step + 1) % args.logging_steps == 0:
                logger.info('epochs: %d, train step: %d, total step: %d, loss: %s', epoch, step, len(epoch_iterator), loss.item())

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                # logger.info("Decaying learning rate to %g" % scheduler.get_lr()[0])

                base_ratio = args.min_learning_rate / args.learning_rate
                if global_step < args.warmup_steps:
                    ratio = base_ratio + (1. - base_ratio) / max(1., args.warmup_steps) * global_step
                else:
                    ratio = max(base_ratio, args.lr_decay_rate ** math.floor((global_step - args.warmup_steps) /
                                                                             args.lr_decay_steps))
                optimizer.param_groups[0]['lr'] = args.learning_rate * ratio

                model.zero_grad()
                global_step += 1

            if (step + 1) % args.eval_steps == 0:
                logger.info(" train average loss = %s", epoch_loss / step)

                dev_dataset = load_dataset(args, word2id, 'test')
                acc, preds = evaluate(args, dev_dataset, model, tokenizer, word2id)
                
                if acc > best_acc:
                    model_path = os.path.join(args.output_dir, 'damm_'+args.task+'_model_best.pth')
                    torch.save(model, model_path)
                    logger.info(' save model to output %s', model_path)
                best_acc = max(acc, best_acc)
                logger.info('best acc: %s', best_acc)
                

        logger.info(" train average loss = %s", epoch_loss / step)

        dev_dataset = load_dataset(args, word2id, 'test')
        f1, preds = evaluate(args, dev_dataset, model, tokenizer, word2id)
        best_acc = max(f1, best_acc)
        logger.info('best acc: %s', best_acc)

        # f = codecs.open('snli/test.txt', 'r')
        # f_out = codecs.open('snli_bad_case/bad_case_' + str(epoch) + '.txt', 'w')
        # lines = f.readlines()
        # for i, line in enumerate(lines):
        #     if int(line.strip()[-1]) != preds[i]:
        #         f_out.write(line.strip() + '\t' + str(preds[i]) + '\n')
        # logger.info('write bad case!!!')

    logger.info('*' * 20)
    logger.info('best acc: %s', best_acc)


    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, model, tokenizer, word2id):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)


    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids_a": batch[0], "attention_mask_a": batch[1],
                      "input_ids_b": batch[2], "attention_mask_b": batch[3], "labels": batch[4]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    # result = f1_score(out_label_ids, preds)
    result = accuracy_score(out_label_ids, preds)
    logger.info("eval average loss = %s, accuracy_score = %s", eval_loss, result)

    # output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    #     logger.info("***** Eval results {} *****".format(prefix))
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))

    return result, list(preds)


def main():
    parser = argparse.ArgumentParser()

    # Data And Task
    parser.add_argument("--data_dir", default='data', type=str)
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument("--task", default='snli', type=str)
    parser.add_argument("--labels", default=['0', '1', '2'], type=list)

    # Training parameters
    parser.add_argument("--do_train", default=True, type=bool)
    parser.add_argument("--do_eval", default=True, type=bool)
    parser.add_argument("--do_test", default=True, type=bool)
    parser.add_argument("--do_lower_case", default=True, type=bool)
    parser.add_argument("--per_gpu_train_batch_size", default=512, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=512, type=int)
    parser.add_argument("--per_gpu_test_batch_size", default=512, type=int)
    parser.add_argument("--max_seq_length_a", default=32, type=int)
    parser.add_argument("--max_seq_length_b", default=32, type=int)
    parser.add_argument("--num_train_epochs", default=50, type=int)

    # Model parameters
    parser.add_argument("--vocab_size", default=100000, type=int)
    parser.add_argument("--fix_embedding", default=True, type=bool)
    parser.add_argument("--embedding_size", default=300, type=int)
    parser.add_argument("--hidden_size", default=300, type=int)
    parser.add_argument("--intermediate_size", default=2048, type=int)
    parser.add_argument("--num_hidden_layers", default=3, type=int)
    parser.add_argument("--num_encoder_layers", default=2, type=int)
    parser.add_argument("--num_attention_heads", default=6, type=int)
    parser.add_argument("--num_last_selfatt_layers", default=2, type=int)
    parser.add_argument("--embedding_dropout_prob", default=0.2, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.2, type=float)
    parser.add_argument("--attention_dropout_prob", default=0.2, type=float)
    parser.add_argument("--norm_eps", default=1e-12, type=float)
    parser.add_argument("--cnn_num_filters", default=200, type=int)
    parser.add_argument("--cnn_filter_sizes", default=[1,2,3], type=list)
    parser.add_argument("--use_smooth", default=False, type=bool)


    # Optimizer parameters
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--min_learning_rate", default=1e-7, type=float)
    parser.add_argument("--lr_decay_rate", default=0.95, type=float)
    parser.add_argument("--lr_decay_steps", default=40000, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    # Other parameters
    parser.add_argument("--no_cuda", default=False, type=bool)
    parser.add_argument("--seed", default=2020, type=int)
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Eval every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )

    args = parser.parse_args()
    # args = parser.parse_args(args=[])

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO, )
    logger.warning("device: %s, n_gpu: %s,", device, args.n_gpu)

    args.logger = logger

    # Set seed
    set_seed(args)

    # Set Vocab
    vocab, word2id = load_vocab(args)

    args.vocab_size = max(args.vocab_size, len(vocab)+1)

    # Build Model
    model = MatchModel(args)
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    model.to(args.device)
    model_init(model, args)
    logger.info("Training parameters %s", args)

    if args.do_train:
        train_dataset = load_dataset(args, word2id, 'train')
        global_step, tr_loss = train(args, train_dataset, model, vocab, word2id)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == '__main__':
    main()






























































#
# """
#     # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
#     if args.do_train:
#         # Create output directory if needed
#         if not os.path.exists(args.output_dir):
#             os.makedirs(args.output_dir)
#
#
#
#         logger.info("Saving model checkpoint to %s", args.output_dir)
#         # Save a trained model, configuration and tokenizer using `save_pretrained()`.
#         # They can then be reloaded using `from_pretrained()`
#         model_to_save = (
#             model.module if hasattr(model, "module") else model
#         )  # Take care of distributed/parallel training
#         model_to_save.save_pretrained(args.output_dir)
#         tokenizer.save_pretrained(args.output_dir)
#
#         # Good practice: save your training arguments together with the trained model
#         torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
#
#         # Load a trained model and vocabulary that you have fine-tuned
#         model = model_class.from_pretrained(args.output_dir)
#         tokenizer = tokenizer_class.from_pretrained(args.output_dir)
#         model.to(args.device)
#
#
#     # Evaluation
#     results = {}
#
#     if args.do_eval:
#         tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
#         checkpoints = [args.output_dir]
#         if args.eval_all_checkpoints:
#             checkpoints = list(
#                 os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
#             )
#             logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
#         logger.info("Evaluate the following checkpoints: %s", checkpoints)
#         for checkpoint in checkpoints:
#             global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
#             prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
#
#             model = model_class.from_pretrained(checkpoint)
#             model.to(args.device)
#             result = evaluate(args, model, tokenizer, prefix=prefix)
#             result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
#             results.update(result)
#
#
#     return results
#
#     """
#
#
# # f = codecs.open('snli/test.txt', 'r')
#                 # f_out = codecs.open('snli_bad_case/bad_case_'+str(epoch) +'.txt', 'w')
#                 # lines = f.readlines()
#                 # for i, line in enumerate(lines):
#                 #     if int(line.strip()[-1]) != preds[i]:
#                 #         f_out.write(line.strip() + '\t' + str(preds[i]) + '\n')
#                 # logger.info('write bad case!!!')
#
#
# """
#     # Prepare optimizer and schedule (linear warmup and decay)
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#             "weight_decay": args.weight_decay,
#         },
#         {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
#     ]
#
#
#     optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
#     )
#
#     # Check if saved optimizer or scheduler states exist
#     if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
#         os.path.join(args.model_name_or_path, "scheduler.pt")
#     ):
#         # Load in optimizer and scheduler states
#         optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
#         scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
#     """
#
#
# # show loss pic
#     x = []
#     y = loss_show
#     for i in range(len(y)):
#         x.append(i)
#
#     import matplotlib
#     import matplotlib.pyplot as plt
#     from matplotlib.pyplot import MultipleLocator
#     # 从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
#
#     % matplotlib
#     inline
#     plt.plot(x, y)
#     plt.title('Loss Change')
#     plt.xlabel("Step")
#     plt.ylabel("Loss")
#     plt.grid(True)
#
#     #     x_major_locator=MultipleLocator(0.05)
#     #     #把x轴的刻度间隔设置为0.05，并存在变量里
#     #     y_major_locator=MultipleLocator(0.1)
#     #     #把y轴的刻度间隔设置为0.1，并存在变量里
#     #     ax=plt.gca()
#     #     #ax为两条坐标轴的实例
#     #     ax.xaxis.set_major_locator(x_major_locator)
#     #     #把x轴的主刻度设置为0.05的倍数
#     #     ax.yaxis.set_major_locator(y_major_locator)
#     #     #把y轴的主刻度设置为0.1的倍数
#     #     plt.xlim(0, 1)
#     #     #把x轴的刻度范围设置为0到1
#     #     plt.ylim(0, 5)
#     # 把y轴的刻度范围设置为0到5
#
#     # plt.savefig('CrossEntropyLoss.png')
#     plt.show()
#
# # Check if continuing training from a checkpoint
#     """
#     if os.path.exists(args.model_name_or_path):
#         # set global_step to gobal_step of last saved checkpoint from model path
#         global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
#         epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
#         steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
#
#         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
#         logger.info("  Continuing training from epoch %d", epochs_trained)
#         logger.info("  Continuing training from global step %d", global_step)
#         logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
#     """
