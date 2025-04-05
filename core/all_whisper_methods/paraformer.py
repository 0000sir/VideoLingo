from funasr import AutoModel
import torch
from http import HTTPStatus
from dashscope.audio.asr import Transcription
import dashscope
import json
from urllib import request

dashscope.api_key = 'sk-4b398290c459486789b6e8db1f97948e'

def transcribe_audio_local(audio_file: str, words: dict=None):
  dev = "cuda" if torch.cuda.is_available() else "cpu"
  model = AutoModel(model="paraformer-zh",  vad_model="fsmn-vad", punc_model="ct-punc", 
                    device=dev
                    # spk_model="cam++"
                    )
  res = model.generate(input= audio_file, 
              batch_size_s=300, 
              hotword='魔搭')
  return res

# test file:
# https://peggy-belly.oss-cn-beijing.aliyuncs.com/for_whisper.mp3?Expires=1743339518&OSSAccessKeyId=TMP.3KpduJiVURgExdzX1yYJX9igmvB52DAJzxVq3WvNNnWFsN6Vwy7XprAzUccVs5nGDAZF4YZURkBA6oZ7JhkTiKspBvNkn9&Signature=Sm%2FdL8EgIbJGBvnGQakJFQrdFeM%3D
def transcribe_audio_dashscope(audio_file: str):
  url  = "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/prod/paraformer-v2/20250330/20%3A57/0af297db-76c8-4818-9233-76fbe94e4e88-1.json?Expires=1743425821&OSSAccessKeyId=LTAI5tQZd8AEcZX6KZV4G8qL&Signature=4kBoIsEKA4BrKCVTX9%2FdX%2BxcNwk%3D"
  results = json.loads(request.urlopen(url).read().decode('utf8'))
  for sentence in results['transcripts'][0]['sentences']:
    fixed_text = fix_with_llm(sentence['text'])
    sentence['text'] = fixed_text
  return results

  task_response = Transcription.async_call(
    model='paraformer-v2',
    file_urls=['https://peggy-belly.oss-cn-beijing.aliyuncs.com/for_whisper.mp3?Expires=1743340816&OSSAccessKeyId=TMP.3KpduJiVURgExdzX1yYJX9igmvB52DAJzxVq3WvNNnWFsN6Vwy7XprAzUccVs5nGDAZF4YZURkBA6oZ7JhkTiKspBvNkn9&Signature=cNDqTxaNn58qiN8zWo2jGg9bgp0%3D'],
    disfluency_removal_enabled = True,
    diarization_enabled = True,
    language_hints=['zh']  # “language_hints”只支持paraformer-v2模型
  )

  transcribe_response = Transcription.wait(task=task_response.output.task_id)
  if transcribe_response.status_code == HTTPStatus.OK:
    print(transcribe_response.output['results'])
    if transcribe_response.output['results'][0]['subtask_status'] == "SUCCEEDED":
      result_url = transcribe_response.output['results'][0]['transcription_url']
      results = json.loads(request.urlopen(url).read().decode('utf8'))
      for sentence in results['transcripts'][0]['sentences']:
        fixed_text = fix_with_llm(sentence['text'])
        sentence['text'] = fixed_text
      return results


# 根据语义校对
def fix_with_llm(line: str):
  system = """请根据以下背景知识改正用户发来句子中的错别字，仅输出改正后的句子。以下是相关背景：

公元1790年，即乾隆55年，正值高宗八十大寿，全国上下都在庆祝。各地纷纷进贡稀世珍品，而扬州、徽州的盐商们则决定献上一份特别的贺礼——三庆班。三庆班以演唱徽调昆曲为主，广受欢迎，最终留在京城并吸收了秦腔、楚调等剧种的优点，形成了京剧。京剧形成后迅速走红，戏班数量增多，对人才的需求也日益增长，科班应运而生。
1904年，叶春善创立了喜连升京剧科班，最初仅有六名弟子。随后，科班逐渐壮大，培养了许多优秀演员。科班的教学方式包括口传心授和严格的训练，学生需签订生死合同，并在出科后为科班演戏三年。科班还通过演出维持运营，但因时局动荡一度陷入困境。后来，沈家接手科班，改名为复连城，使其起死回生。
随着女演员的出现，女科班也开始兴起，如崇雅社。复连城科班在叶春善的带领下，培养了大批杰出的京剧人才，如马连良、谭富英等，被誉为“京剧第一科班”。
  """
  messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": line}
  ]
  response = dashscope.Generation.call(
    model = "qwen-max",
    messages = messages,
    result_format = "message",
    stream = False
  )
  return response.output.choices[0].message.content

if __name__ == "__main__":
  print(fix_with_llm("各种稀式贡品从全国各地陆续运往京城。"))