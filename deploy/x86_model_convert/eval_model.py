from transformers import AutoConfig

path = "/share_data/public/openvla_models/hf_llama"
config = AutoConfig.from_pretrained(path, local_files_only=True)

print("model_type:", config.model_type)
print("architectures:", config.architectures)
print("hidden_size:", config.hidden_size)
print("num_hidden_layers:", config.num_hidden_layers)
print("num_attention_heads:", config.num_attention_heads)
print("num_key_value_heads:", config.num_key_value_heads)
print("vocab_size:", config.vocab_size)
print("max_position_embeddings:", config.max_position_embeddings)