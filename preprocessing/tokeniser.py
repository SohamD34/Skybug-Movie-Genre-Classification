def Tokeniser(text, model_name='bert-base-uncased'):

    from transformers import BertTokenizer, BertModel
    import torch

    tokenizer = BertTokenizer.from_pretrained(model_name, padding=True, truncation=True)
    model = BertModel.from_pretrained(model_name)

    tokens = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**tokens)
    
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embeddings
