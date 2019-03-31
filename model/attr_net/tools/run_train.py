from options import Options
from datasets import get_dataloader, get_dataset, get_tester
from model import get_model
from trainer import get_trainer
import programs as program_utils
from gqa_executor import gqaExecutor

args = Options().parse()
train_ds, val_ds, _ = get_dataset(args)
train_loader = get_dataloader(args, train_ds)
val_loader = get_dataloader(args, val_ds)
executor = gqaExecutor(args, vocabulary=args.vocabulary_file, protocol=args.protocol_file)
tester = get_tester(args, executor, program_utils)
print('==> get model')
model = get_model(args)
print('==> get the trainer')
trainer = get_trainer(args, model, train_loader, val_loader, tester) #TODO: for debug only

trainer.train()
