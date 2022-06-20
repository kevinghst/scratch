BSIZE = 4
GACC = 6
NGPU = 8
EPOCHS = 30

STEPS = len(train_dataset) // (BSIZE * GACC * NGPU)

print(f"Steps per epoch: {STEPS}")

training_args = TrainingArguments(
  output_dir="italian/",
  group_by_length=False,
  dataloader_num_workers=30,
  per_device_train_batch_size=BSIZE,
  per_device_eval_batch_size=1,
  gradient_accumulation_steps=GACC,
  evaluation_strategy="steps",
  num_train_epochs=EPOCHS,
  fp16=False,
  save_steps=STEPS//2,
  eval_steps=STEPS//2,
  logging_steps=100,
  learning_rate=5e-5,
  warmup_steps=12000,
  save_total_limit=20,
  remove_unused_columns=False,
)

trainer = RNNTTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    optimizers=(optimizer, scheduler),
)

class RNNTTrainer(Trainer):

    def evaluate(
            self,
            eval_dataset = None,
            ignore_keys = None,
            metric_key_prefix: str = "eval",
        ):
            # memory metrics - must set up as early as possible
            self._memory_tracker.start()

            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            model_evaluator = DecoderUtils(self.model, self.args.world_size)
            # Perform decoding and loss calculations here
            _, wer = model_evaluator.evaluate(eval_dataloader, self.tokenizer)
            metrics = {'wer': wer}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            self._memory_tracker.stop_and_update_metrics(metrics)

            return metrics

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        print('ing remove unused columns')
        return dataset

class DecoderUtils:
    def __init__(self, model, world_size, decode_args=None):
        self.world_size = world_size
        if not decode_args:
            decode_args = {
            'algo': 'greedy2',
            'compare_with_greedy': False,
            'decode_compare': False,
            'delimit': False,
            'split_length': 240000,
            'overlap': True
            }
        for key in decode_args.keys():
            setattr(self, key, decode_args[key])
        self.model = model
    
    def evaluate(self, test_loader, tokenizer=None, calc_wer=True):
        self.model.eval()
        valid_loss = []

        total_ed_dist = 0
        total_ref_len = 0

        with torch.no_grad():
            for i, _data in enumerate(tqdm(test_loader)):
                spectrograms = _data['input_values']
                input_lengths = _data['input_length']
                labels = _data['labels']
                label_lengths = _data['label_lengths']
                
                # calculate wer and cer
                if calc_wer:
                    for j, xs in enumerate(spectrograms):
                        sample_input_length = int(input_lengths[j])
                        sample_label_length = int(label_lengths[j])
                        sample_spec = torch.unsqueeze(xs[:sample_input_length], 0)
                        sample_label = torch.unsqueeze(labels[j][:sample_label_length], 0)

                        ref = sample_label[0].tolist()
                        ref = [int(x) for x in ref]
                        rt = ' '.join(tokenizer.convert_ids_to_tokens(ref)).replace(' ##', '').replace(" ' ",
                                                                                                        "'").replace(
                            ' [pad]', '')

                        yt = self.master_decode(sample_spec.cuda(), [], tokenizer)
                        _, ed_dist, ref_len = wer(rt, yt)
                        
                        total_ed_dist += ed_dist
                        total_ref_len += ref_len
                input_lengths = torch.tensor(input_lengths, dtype=torch.int32)
                label_lengths = torch.tensor(label_lengths, dtype=torch.int32)
                labels = torch.tensor(labels, dtype=torch.int64)
                
                spectrograms, input_lengths = spectrograms.cuda(), input_lengths.cuda()
                labels, label_lengths = labels.cuda(), label_lengths.cuda()
                
                loss_val = 0
                valid_loss.append(loss_val)
        avg_valid_loss = 0
        avg_valid_loss = sum(valid_loss) / len(valid_loss)
        avg_wer = total_ed_dist / total_ref_len

        return avg_valid_loss, avg_wer

