import torch
from peft import get_peft_model, LoraConfig, TaskType


from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPTNeoForCausalLM,
    GPT2Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
import pandas as pd
from sklearn.model_selection import train_test_split

import os
from tqdm import tqdm

def get_lora_config(r=8, alpha=32, target_modules=["q_proj", "v_proj"]):
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

class PoliticalTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        cleaned_texts = [str(text).strip() for text in texts if pd.notna(text)]
        
        formatted_texts = [f"Generate a political response: {text}" for text in cleaned_texts]
        
        self.encodings = tokenizer(
            formatted_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

class GPTNeoPoliticalQA:
    def __init__(self, model_name='EleutherAI/gpt-neo-125m', device='cuda:1' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        base_model = GPTNeoForCausalLM.from_pretrained(model_name)
        
        lora_config = get_lora_config(
            target_modules=["query_key_value"]  
        )
        
        self.model = get_peft_model(base_model, lora_config)
        self.model.to(device)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def prepare_data(self, df, text_column='text', test_size=0.2):

        if isinstance(df, pd.DataFrame):
            texts = df[text_column].astype(str).tolist()
        else:
            texts = [str(x) for x in df.tolist()]
        
        texts = [text.strip() for text in texts if text.strip()]
        
        print(f"Total number of texts after cleaning: {len(texts)}")
        
        train_texts, val_texts = train_test_split(texts, test_size=test_size, random_state=42)
        print(f"Training texts: {len(train_texts)}, Validation texts: {len(val_texts)}")
        
        train_dataset = PoliticalTextDataset(train_texts, self.tokenizer)
        val_dataset = PoliticalTextDataset(val_texts, self.tokenizer)
        
        return train_dataset, val_dataset

    def train(self, train_dataset, val_dataset, 
              batch_size=16, 
              epochs=5, 
              learning_rate=2e-5,
              warmup_steps=200,
              output_dir='political_gpt_neo'):
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            train_loss = 0
            for batch in tqdm(train_loader, desc="Training"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['input_ids']
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            avg_train_loss = train_loss / len(train_loader)
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['input_ids']
                    )
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"Average validation loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                print(f"Model saved to {output_dir}")
            
            self.model.train()

    def answer_question(self, question, max_length=3000, temperature=0.85):
        input_text = f"Question: {question}\nAnswer:"
        input_ids = self.tokenizer(input_text, return_tensors='pt').to(self.device)['input_ids']
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                min_length=100,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                length_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            return answer.split("Answer:")[1].strip()
        except:
            return answer.strip()

def main():
    df = pd.read_csv('train.csv')

    print("Dataset shape:", df.shape)
    print("\nSample of the dataset:")
    print(df.head())

    political_qa = GPTNeoPoliticalQA()
    
    train_dataset, val_dataset = political_qa.prepare_data(df)

    political_qa.train(
        train_dataset,
        val_dataset,
        batch_size=16,
        epochs=5,
        output_dir='political_gpt_neo_model'
    )

    questions_df = pd.read_csv('questions.csv')

    results = []
    for idx, row in tqdm(questions_df.iterrows(), desc="Generating answers", total=len(questions_df)):
        answer = political_qa.answer_question(row['question'])
        results.append({
            'question': row['question'],
            'answer': answer,
            'label': row['Labels']
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('gpt_neo_results.csv', index=False)
    print(f"\nResults saved to gpt_neo_results.csv")


if __name__ == "__main__":
    main()
