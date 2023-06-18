from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tensorflow as tf
import yaml
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import AutoProcessor

def do_synthesis(input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name):
  input_ids = processor.text_to_sequence(input_text)

  # text2mel part
  if text2mel_name == "TACOTRON":
    _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        tf.convert_to_tensor([len(input_ids)], tf.int32),
        tf.convert_to_tensor([0], dtype=tf.int32)
    )
  elif text2mel_name == "FASTSPEECH":
    mel_before, mel_outputs, duration_outputs = text2mel_model.inference(
        input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    )
  elif text2mel_name == "FASTSPEECH2":
    mel_before, mel_outputs, duration_outputs, _, _ = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    )
  else:
    raise ValueError("Only TACOTRON, FASTSPEECH, FASTSPEECH2 are supported on text2mel_name")

  # vocoder part
  if vocoder_name == "MELGAN" or vocoder_name == "MELGAN-STFT":
    audio = vocoder_model(mel_outputs)[0, :, 0]
  elif vocoder_name == "MB-MELGAN":
    audio = vocoder_model(mel_outputs)[0, :, 0]
  else:
    raise ValueError("Only MELGAN, MELGAN-STFT and MB_MELGAN are supported on vocoder_name")

  if text2mel_name == "TACOTRON":
    return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
  else:
    return mel_outputs.numpy(), audio.numpy()


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def dialogGPT2TTS(input_text):
  model_name = "microsoft/DialoGPT-medium"

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name)

  # chatting 5 times with nucleus sampling & tweaking temperature
  # for step in range(5):
  # take user input
  text = input_text
  # encode the input and add end of string token
  input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
  # concatenate new user input with chat history (if there is)
  # bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids
  bot_input_ids = input_ids
  # generate a bot response
  chat_history_ids = model.generate(
      bot_input_ids,
      max_length=1000,
      do_sample=True,
      top_p=0.95,
      top_k=0,
      temperature=0.75,
      pad_token_id=tokenizer.eos_token_id
  )
  #print the output
  output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
  print(f"DialoGPT: {output}")

  mels, audios = do_synthesis(output, fastspeech2, mb_melgan, "FASTSPEECH2", "MB-MELGAN")

  return output, audios

if __name__ == "__main__":
    # test
    input_text = "Can you recommend some medieval art pieces?"
    output, audios = dialogGPT2TTS(input_text)
    print(output)
    ipd.Audio(audios, rate=22050)
