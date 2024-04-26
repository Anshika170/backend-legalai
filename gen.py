
from transformers import GPT2LMHeadModel, GPT2Tokenizer  

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large",pad_token_id=tokenizer.eos_token_id)



def generate_contract(prompt):

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=150,  # Adjust max length as needed
        num_return_sequences=1,  # Generate only one sequence
        temperature=0.7,  # Adjust temperature for randomness
        top_k=50,  # Adjust top-k sampling parameter
        top_p=0.9,  # Adjust nucleus sampling parameter
        num_beams=5,  # Adjust beam width
        no_repeat_ngram_size=3,  # Avoid repeating n-grams
        early_stopping=True,  # Stop generation when EOS token is reached
        pad_token_id=tokenizer.eos_token_id)

    generated_contract = tokenizer.decode(output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
    return ".".join(generated_contract.split(".")[:-1])+"."


#max_length=100,num_beams=5, no_repeat_ngram_size=2,early_stopping= True