text = "Welcome to use paddle paddle and paddlenlp!"
torch_model_name = "uclanlp/visualbert-vqa"
paddle_model_name = "visualbert-vqa"

# torch output
import torch
from transformers import BertTokenizer as PTBertTokenizer, VisualBertForQuestionAnswering as PTVisualBertForQuestionAnswering

import paddle
from paddlenlp.transformers import BertTokenizer as PDBertTokenizer, VisualBertForQuestionAnswering as PDVisualBertForQuestionAnswering
import numpy as np

torch_model = PTVisualBertForQuestionAnswering.from_pretrained(torch_model_name)
torch_tokenizer = PTBertTokenizer.from_pretrained("bert-base-uncased")
torch_model.eval()

torch_inputs = torch_tokenizer(text, return_tensors="pt")
torch_visual_embeds = torch.ones([100,2048]).unsqueeze(0)
torch_visual_token_type_ids = torch.ones(torch_visual_embeds.shape[:-1], dtype=torch.int64)
torch_visual_attention_mask = torch.ones(torch_visual_embeds.shape[:-1], dtype=torch.int64)
torch_inputs.update({
    "visual_embeds": torch_visual_embeds,
    "visual_token_type_ids": torch_visual_token_type_ids,
    "visual_attention_mask": torch_visual_attention_mask
})

torch_labels = torch.nn.functional.one_hot(torch.tensor(50), num_classes=3129).unsqueeze(0).to(torch.float32) # Batch size 1, Num labels 3092

with torch.no_grad():
    torch_outputs = torch_model(**torch_inputs, labels = torch_labels)
    
torch_loss = torch_outputs[0].cpu().detach().numpy()
torch_logits = torch_outputs[1]
torch_array = torch_logits.cpu().detach().numpy()

print("torch_prediction_loss:{}".format(torch_loss))
print("torch_prediction_logits shape:{}".format(torch_array.shape))
print("torch_prediction_logits:{}".format(torch_array))

# ========================================================================================================
paddle_model = PDVisualBertForQuestionAnswering.from_pretrained(paddle_model_name, num_classes=3129)
paddle_tokenizer = PDBertTokenizer.from_pretrained("bert-base-uncased")
paddle_model.eval()

paddle_inputs = paddle_tokenizer(text)
paddle_inputs['input_ids'] = paddle.to_tensor([paddle_inputs['input_ids']])
paddle_inputs['token_type_ids'] = paddle.to_tensor([paddle_inputs['token_type_ids']])
paddle_visual_embeds = paddle.ones([100,2048]).unsqueeze(0)
paddle_visual_token_type_ids = paddle.ones(paddle_visual_embeds.shape[:-1])
paddle_visual_attention_mask = paddle.ones(paddle_visual_embeds.shape[:-1])

return_dict = False
paddle_inputs.update({
    "visual_embeds": paddle_visual_embeds,
    "visual_token_type_ids": paddle_visual_token_type_ids,
    "visual_attention_mask": paddle_visual_attention_mask,
    "return_dict": return_dict
})

paddle_labels = paddle.nn.functional.one_hot(paddle.to_tensor(50), num_classes=3129).astype(paddle.float32) # Batch size 1, Num labels 3092

with paddle.no_grad():
    paddle_outputs = paddle_model(**paddle_inputs, labels = paddle_labels)

if not return_dict:
    paddle_loss = paddle_outputs[0].cpu().detach().numpy()
    paddle_logits = paddle_outputs[1]
else:
    paddle_loss = paddle_outputs['loss']
    paddle_logits = paddle_outputs['logits']
paddle_array = paddle_logits.cpu().detach().numpy()

print("paddle_prediction_loss:{}".format(paddle_loss))
print("paddle_prediction_logits shape:{}".format(paddle_array.shape))
print("paddle_prediction_logits:{}".format(paddle_array))

# ==============================================================================
assert torch_array.shape == paddle_array.shape, "the output logits should have the same shape, but got : {} and {} instead".format(torch_array.shape, paddle_array.shape)
diff = torch_array - paddle_array
print(np.amax(abs(diff)))