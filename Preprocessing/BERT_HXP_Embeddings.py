from transformers import AutoTokenizer
import torch
from transformers import AutoModel
import pickle, os
from tqdm import tqdm
import torch.nn as nn

FOLDER_NAME = '/backup/girish_datasets/HateMM/'

class Text_Model(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = None
        if not model_name:
            raise ValueError("Invalid model name provided.")
        try:
            self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        except Exception as e:
            print(f"Error loading model: {e}")
        
    def forward(self, x, mask):
        try:
            embeddings = self.model(x, mask)
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            embeddings = None
        return embeddings

def tokenize(sentences, tokenizer, padding=True, max_len=512):
    input_ids, attention_masks, token_type_ids = [], [], []
    for sent in sentences:
        try:
            encoded_dict = tokenizer.encode_plus(sent,
                                             add_special_tokens=True,
                                             max_length=max_len, 
                                             padding='max_length', 
                                             return_attention_mask=True,
                                             return_tensors='pt', 
                                             truncation=True)
        except Exception as e:
            print(f"Error encoding sentence '{sent}': {e}")
            continue
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return {'input_ids': input_ids, 'attention_masks': attention_masks}

def extract_features_from_pickled_file(file_path, model_name, output_file):
    with open(file_path, 'rb') as fp:
        transCript = pickle.load(fp)
        if not isinstance(transCript, dict):
            raise ValueError("Expected 'transCript' to be a dictionary")

    if not isinstance(transCript, dict):
        raise ValueError("Expected 'transCript' to be a dictionary")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Text_Model(model_name)

    allEmbedding = {}
    for i in tqdm(transCript):
        try:
            apr = tokenize([transCript[i]], tokenizer)
            with torch.no_grad():
                allEmbedding[i] = (model(apr['input_ids'], apr['attention_masks'])[2][0]).detach().numpy()
            del(apr)
            print(f"Successfully processed text ID: {i}")
        except KeyError as e:
            print(f"KeyError for i: {i} - {e}")
        except Exception as e:
            print(f"Unexpected error for i: {i} - {e}")
            continue

    with open(output_file, 'wb') as fp:
        pickle.dump(allEmbedding, fp)

def extract_features_from_huggingface(dataset, model_name, output_file):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Text_Model(model_name)

    allEmbedding = {}

    for example in dataset:
        if 'id' not in example or 'text' not in example:
            print(f"Invalid example format: {example}")
            continue
        try:
            text_id = example['id']
            text = example['text']
            inputs = tokenize([text], tokenizer)
            with torch.no_grad():
                embeddings = model(inputs['input_ids'], inputs['attention_masks'])[2][0].detach().numpy()          
                allEmbedding[text_id] = embeddings
        except Exception as e:
            print(f"Error processing text with ID: {text_id}. Skipping this sample.")
            continue

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'rb') as fp:
            existing_data = pickle.load(fp)
    else:
        existing_data = {}

    existing_data.update(allEmbedding)

    with open(output_file, 'wb') as fp:
        pickle.dump(existing_data, fp)

# Example usage
# extract_features_from_pickled_file(FOLDER_NAME+'all_whisper_tiny_transcripts.pkl', "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two", FOLDER_NAME+'all_HateXPlainembedding_whisper.pkl')

# from datasets import load_dataset
# dataset = load_dataset('limjiayi/hateful_memes_expanded')
# extract_features_from_huggingface(dataset['train'], "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two", FOLDER_NAME + 'all_hatefulmemes_train_hatexplain_embedding.p')
# extract_features_from_huggingface(dataset['validation'], "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two", FOLDER_NAME + 'all_hatefulmemes_validation_hatexplain_embedding.p')
# extract_features_from_huggingface(dataset['test'], "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two", FOLDER_NAME + 'all_hatefulmemes_test_hatexplain_embedding.p')
# extract_features_from_huggingface(dataset['test'], "bert-base-uncased", FOLDER_NAME + 'all_hatefulmemes_test_rawBERTembedding.p')
