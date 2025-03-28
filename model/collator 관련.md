## left padding, right padding
- torch에서 rnn.sequence 어쩌구 해서 padding 하는게 있는데 그건 default가 right padding이다.
- 따라서 left padding을 구현하려면 직접 구현해야 한다.

  ```python
  def collate_fn(batch_samples):
    input_ids_list = [torch.tensor(sample["input_ids"], dtype=torch.long) for sample in batch_samples]
    raw_batch = []
    max_len = max(len(ids) for ids in input_ids_list)
    
    padded_input_ids = []
    for i, ids in enumerate(input_ids_list):
        pad_len = max_len - len(ids)
        # left padding
        padded = torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long), ids])
        padded_input_ids.append(padded)
        raw_batch.append(batch_samples[i])
    
    input_ids = torch.stack(padded_input_ids)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    return {"input_ids": input_ids, "attention_mask": attention_mask}, raw_batch
  
  dataloader = DataLoader(tokenized_dataset_splits['test'], batch_size=64, collate_fn=collate_fn)```
