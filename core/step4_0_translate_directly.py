import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import json
import concurrent.futures
from core.translate_once import translate_lines
from core.step4_1_summarize import search_things_to_note_in_prompt
from core.step8_1_gen_audio_task import check_len_then_trim
from core.step6_generate_final_timeline import align_timestamp
from core.config_utils import load_key
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from difflib import SequenceMatcher
from datetime import timedelta

console = Console()

SENTENCE_SPLIT_FILE = "output/log/sentence_splitbymeaning.txt"
RAW_TRANSLATION_RESULTS_FILE =  "output/log/translation_results.json"
TRANSLATION_RESULTS_FILE = "output/log/translation_results.xlsx"
TERMINOLOGY_FILE = "output/log/terminology.json"
CLEANED_CHUNKS_FILE = "output/log/cleaned_chunks.xlsx"
PROOFREADED_ASR = "output/audio/asr_proofreaded.json"

# Function to split text into chunks
def split_chunks_by_chars(chunk_size=400, max_i=8): 
    """Split text into chunks based on character count, return a list of multi-line text chunks"""
    with open(SENTENCE_SPLIT_FILE, "r", encoding="utf-8") as file:
        sentences = file.read().strip().split('\n')

    chunks = []
    chunk = ''
    sentence_count = 0
    for sentence in sentences:
        if len(chunk) + len(sentence + '\n') > chunk_size or sentence_count == max_i:
            chunks.append(chunk.strip())
            chunk = sentence + '\n'
            sentence_count = 1
        else:
            chunk += sentence + '\n'
            sentence_count += 1
    chunks.append(chunk.strip())
    return chunks

def split_asr_result(max_sentence=10):
  original = json.loads(open(PROOFREADED_ASR, 'r', encoding='utf-8').read())
  segments = original['segments']
  chunks = []
  for i in range(0, len(segments), max_sentence):
      batch = segments[i:i+max_sentence]
      texts = [item['text'] for item in batch]
      texts = "\n".join(texts)
      chunks.append(texts)
  return chunks


# Get context from surrounding chunks
def get_previous_content(chunks, chunk_index):
    return None if chunk_index == 0 else chunks[chunk_index - 1].split('\n')[-3:] # Get last 3 lines
def get_after_content(chunks, chunk_index):
    return None if chunk_index == len(chunks) - 1 else chunks[chunk_index + 1].split('\n')[:2] # Get first 2 lines

# 🔍 Translate a single chunk
def translate_chunk(chunk, chunks, theme_prompt, i):
    things_to_note_prompt = search_things_to_note_in_prompt(chunk)
    previous_content_prompt = get_previous_content(chunks, i)
    after_content_prompt = get_after_content(chunks, i)
    # translation , original_text
    translation, english_result = translate_lines(chunk, previous_content_prompt, after_content_prompt, things_to_note_prompt, theme_prompt, i)
    return i, english_result, translation

# Add similarity calculation function
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# 🚀 Main function to translate all chunks
def translate_all():
    # Check if the file exists
    # if os.path.exists(TRANSLATION_RESULTS_FILE):
    #     console.print(Panel("🚨 File `translation_results.xlsx` already exists, skipping TRANSLATE ALL.", title="Warning", border_style="yellow"))
    #     return
    
    console.print("[bold green]Start Translating All...[/bold green]")
    if os.path.exists(RAW_TRANSLATION_RESULTS_FILE):
      results = json.loads(open(RAW_TRANSLATION_RESULTS_FILE, 'r', encoding='utf-8').read())
    else:
      # chunks = split_chunks_by_chars(chunk_size=500, max_i=10)
      chunks = split_asr_result()

      # NO theme FOUND, maybe topic?
      with open(TERMINOLOGY_FILE, 'r', encoding='utf-8') as file:
          theme_prompt = json.load(file).get('topic')
      
      # 🔄 Use concurrent execution for translation
      with Progress(
          SpinnerColumn(),
          TextColumn("[progress.description]{task.description}"),
          transient=True,
      ) as progress:
          task = progress.add_task("[cyan]Translating chunks...", total=len(chunks))
          with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
              futures = []
              for i, chunk in enumerate(chunks):
                  future = executor.submit(translate_chunk, chunk, chunks, theme_prompt, i)
                  futures.append(future)

              results = []
              for future in concurrent.futures.as_completed(futures):
                  results.append(future.result())
                  progress.update(task, advance=1)

      results.sort(key=lambda x: x[0])  # Sort results based on original order
      with open(RAW_TRANSLATION_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # # 💾 Save results to lists and Excel file
    # src_text, trans_text = [], []
    # for i, chunk in enumerate(chunks):
    #     chunk_lines = chunk.split('\n')
    #     src_text.extend(chunk_lines)
        
    #     # Calculate similarity between current chunk and translation results
    #     chunk_text = ''.join(chunk_lines).lower()
    #     matching_results = [(r, similar(''.join(r[1].split('\n')).lower(), chunk_text)) 
    #                       for r in results]
    #     best_match = max(matching_results, key=lambda x: x[1])
        
    #     # Check similarity and handle exceptions
    #     if best_match[1] < 0.9:
    #         console.print(f"[yellow]Warning: No matching translation found for chunk {i}[/yellow]")
    #         raise ValueError(f"Translation matching failed (chunk {i})")
    #     elif best_match[1] < 1.0:
    #         console.print(f"[yellow]Warning: Similar match found (chunk {i}, similarity: {best_match[1]:.3f})[/yellow]")
            
    #     trans_text.extend(best_match[0][2].split('\n'))
    
    # # Trim long translation text
    # df_text = pd.read_excel(CLEANED_CHUNKS_FILE)
    # df_text['text'] = df_text['text'].str.strip('"').str.strip()
    # df_translate = pd.DataFrame({'Source': src_text, 'Translation': trans_text})
    # subtitle_output_configs = [('trans_subs_for_audio.srt', ['Translation'])]
    # df_time = align_timestamp(df_text, df_translate, subtitle_output_configs, output_dir=None, for_display=False)
    # console.print(df_time)
    # # apply check_len_then_trim to df_time['Translation'], only when duration > MIN_TRIM_DURATION.
    # df_time['Translation'] = df_time.apply(lambda x: check_len_then_trim(x['Translation'], x['duration']) if x['duration'] > load_key("min_trim_duration") else x['Translation'], axis=1)
    # console.print(df_time)

    # 拆分翻译结果
    source_lines = []
    target_lines = []
    timestamps = []
    durations = []
    for group in results:
      source_lines.extend(group[1].split("\n"))
      target_lines.extend(group[2].split("\n"))
    print(f"source lines: {len(source_lines)}\ntarget_lines: {len(target_lines)}")
    # 从校对后的ASR结果中找到对应的时间轴
    asr_results = json.loads(open(PROOFREADED_ASR, 'r', encoding='utf-8').read())
    asr_results = asr_results['segments']
    for i in range(len(source_lines)):
      timestamps.append(f"{convert_seconds_to_time_string(asr_results[i]['start'])} --> {convert_seconds_to_time_string(asr_results[i]['end'])}")
      durations.append(asr_results[i]['end']-asr_results[i]['start'])

    df_time = pd.DataFrame({'Source': source_lines, 'Translation': target_lines, 'timestamp': timestamps, 'duration': durations})
    
    df_time.to_excel(TRANSLATION_RESULTS_FILE, index=False)
    console.print("[bold green]✅ Translation completed and results saved.[/bold green]")

def convert_seconds_to_time_string(seconds:float):
    # 使用 timedelta 来处理时间转换
    td = timedelta(milliseconds=seconds*1000)
    
    # 提取小时、分钟、秒和毫秒
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000  # 转换为毫秒

    # 格式化输出
    time_str = f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
    return time_str

if __name__ == '__main__':
    translate_all()