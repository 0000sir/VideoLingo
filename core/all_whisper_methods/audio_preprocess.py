import os, sys, subprocess
import pandas as pd
from typing import Dict, List, Tuple
from rich import print
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.config_utils import update_key
from core.ask_gpt import ask_gpt
import json
from rich import print as rprint
from step4_1_summarize import get_summary

AUDIO_DIR = "output/audio"
RAW_AUDIO_FILE = "output/audio/raw.mp3"
CLEANED_CHUNKS_EXCEL_PATH = "output/log/cleaned_chunks.xlsx"
TERMINOLOGY_JSON_PATH = 'output/log/terminology.json'
SUMMARY_CACHE = 'output/log/summary.txt'

def compress_audio(input_file: str, output_file: str):
    """将输入音频文件压缩为低质量音频文件，用于转录"""
    if not os.path.exists(output_file):
        print(f"🗜️ Converting to low quality audio with FFmpeg ......")
        # 16000 Hz, 1 channel, (Whisper default) , 96kbps to keep more details as well as smaller file size
        subprocess.run([
            'ffmpeg', '-y', '-i', input_file, '-vn', '-b:a', '96k',
            '-ar', '16000', '-ac', '1', '-metadata', 'encoding=UTF-8',
            '-f', 'mp3', output_file
        ], check=True, stderr=subprocess.PIPE)
        print(f"🗜️ Converted <{input_file}> to <{output_file}> with FFmpeg")
    return output_file

def convert_video_to_audio(video_file: str):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    if not os.path.exists(RAW_AUDIO_FILE):
        print(f"🎬➡️🎵 Converting to high quality audio with FFmpeg ......")
        subprocess.run([
            'ffmpeg', '-y', '-i', video_file, '-vn',
            '-c:a', 'libmp3lame', '-b:a', '128k',
            '-ar', '32000',
            '-ac', '1', 
            '-metadata', 'encoding=UTF-8', RAW_AUDIO_FILE
        ], check=True, stderr=subprocess.PIPE)
        print(f"🎬➡️🎵 Converted <{video_file}> to <{RAW_AUDIO_FILE}> with FFmpeg\n")

def _detect_silence(audio_file: str, start: float, end: float) -> List[float]:
    """Detect silence points in the given audio segment"""
    cmd = ['ffmpeg', '-y', '-i', audio_file, 
           '-ss', str(start), '-to', str(end),
           '-af', 'silencedetect=n=-30dB:d=0.5', 
           '-f', 'null', '-']
    
    output = subprocess.run(cmd, capture_output=True, text=True, 
                          encoding='utf-8').stderr
    
    return [float(line.split('silence_end: ')[1].split(' ')[0])
            for line in output.split('\n')
            if 'silence_end' in line]

def get_audio_duration(audio_file: str) -> float:
    """Get the duration of an audio file using ffmpeg."""
    cmd = ['ffmpeg', '-i', audio_file]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    output = stderr.decode('utf-8', errors='ignore')
    
    try:
        duration_str = [line for line in output.split('\n') if 'Duration' in line][0]
        duration_parts = duration_str.split('Duration: ')[1].split(',')[0].split(':')
        duration = float(duration_parts[0])*3600 + float(duration_parts[1])*60 + float(duration_parts[2])
    except Exception as e:
        print(f"[red]❌ Error: Failed to get audio duration: {e}[/red]")
        duration = 0
    return duration

def split_audio(audio_file: str, target_len: int = 30*60, win: int = 60) -> List[Tuple[float, float]]:
    # 30 min 16000 Hz 96kbps ~ 22MB < 25MB required by whisper
    print("[bold blue]🔪 Starting audio segmentation...[/]")
    
    duration = get_audio_duration(audio_file)
    
    segments = []
    pos = 0
    while pos < duration:
        if duration - pos < target_len:
            segments.append((pos, duration))
            break
        win_start = pos + target_len - win
        win_end = min(win_start + 2 * win, duration)
        silences = _detect_silence(audio_file, win_start, win_end)
    
        if silences:
            target_pos = target_len - (win_start - pos)
            split_at = next((t for t in silences if t - win_start > target_pos), None)
            if split_at:
                segments.append((pos, split_at))
                pos = split_at
                continue
        segments.append((pos, pos + target_len))
        pos += target_len
    
    print(f"🔪 Audio split into {len(segments)} segments")
    return segments

def process_transcription(result: Dict) -> pd.DataFrame:
    all_words = []
    for segment in result['segments']:
        for word in segment['words']:
            # Check word length
            if len(word["word"]) > 20:
                print(f"⚠️ Warning: Detected word longer than 20 characters, skipping: {word['word']}")
                continue
                
            # ! For French, we need to convert guillemets to empty strings
            word["word"] = word["word"].replace('»', '').replace('«', '')
            
            if 'start' not in word and 'end' not in word:
                if all_words:
                    # Assign the end time of the previous word as the start and end time of the current word
                    word_dict = {
                        'text': word["word"],
                        'start': all_words[-1]['end'],
                        'end': all_words[-1]['end'],
                    }
                    all_words.append(word_dict)
                else:
                    # If it's the first word, look next for a timestamp then assign it to the current word
                    next_word = next((w for w in segment['words'] if 'start' in w and 'end' in w), None)
                    if next_word:
                        word_dict = {
                            'text': word["word"],
                            'start': next_word["start"],
                            'end': next_word["end"],
                        }
                        all_words.append(word_dict)
                    else:
                        raise Exception(f"No next word with timestamp found for the current word : {word}")
            else:
                # Normal case, with start and end times
                word_dict = {
                    'text': f'{word["word"]}',
                    'start': word.get('start', all_words[-1]['end'] if all_words else 0),
                    'end': word['end'],
                }
                
                all_words.append(word_dict)
    
    return pd.DataFrame(all_words)

def save_results(df: pd.DataFrame):
    os.makedirs('output/log', exist_ok=True)

    # Remove rows where 'text' is empty
    initial_rows = len(df)
    df = df[df['text'].str.len() > 0]
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"ℹ️ Removed {removed_rows} row(s) with empty text.")
    
    # Check for and remove words longer than 20 characters
    long_words = df[df['text'].str.len() > 20]
    if not long_words.empty:
        print(f"⚠️ Warning: Detected {len(long_words)} word(s) longer than 20 characters. These will be removed.")
        df = df[df['text'].str.len() <= 20]
    
    df['text'] = df['text'].apply(lambda x: f'"{x}"')
    df.to_excel(CLEANED_CHUNKS_EXCEL_PATH, index=False)
    print(f"📊 Excel file saved to {CLEANED_CHUNKS_EXCEL_PATH}")

def save_language(language: str):
    update_key("whisper.detected_language", language)

# 根据对话内容总结要点
def summarize(content: str):
    system = """
    Please extract the background information, main content points, names, and explanations of key terms from the user-provided dialogue or narrative.
    """
    os.makedirs('output/log', exist_ok=True)
    if os.path.exists(SUMMARY_CACHE):
        return open(SUMMARY_CACHE).read()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content}
    ]
    res = ask_gpt(content, response_json=False, valid_def=None, log_title='summarize', system=system)
    with open(SUMMARY_CACHE, 'w', encoding='utf-8') as f:
        f.write(res)
    return res

# 使用大模型总结全篇内容要点，并以此为背景校对语句（同时更新文字）
# segments结构
# "segments": [
#         {
#             "start": 0.069,
#             "end": 25.687,
#             "text": "这一万",
#             "words": [
#                 {
#                     "word": "这",
#                     "start": 0.069,
#                     "end": 0.209,
#                     "score": 0.836
#                 },
#                 {
#                     "word": "一",
#                     "start": 0.209,
#                     "end": 0.329,
#                     "score": 0.965
#                 },
#                 {
#                     "word": "万",
#                     "start": 0.329,
#                     "end": 0.87,
#                     "score": 0.958
#                 },
#         }
# ]
def proofread_with_semantic(segments: list):
    texts = []
    for seg in segments:
        texts.append(seg['text'])
    # get_summary() # generate summary 有循环依赖问题
    # summary = json.loads(open(TERMINOLOGY_JSON_PATH, 'r', encoding='utf-8').read())
    summary = summarize("\n".join(texts))

    system = f"""用户将发给你一段视频配音文字的片段，格式为：
    [
        "第一句内容",
        "第二句内容"
    ]
    请根据背景知识，改正其中可能存在的错误文字，请直接输出修改后的内容，以同样的json格式返回，
    如果用户发来的信息中没有错误，请直接输出原文，不要添加其它任何说明，以下是背景知识：
<background>
{summary}
</background>
"""
    results = []
    # 每次提取20句
    batch_size = 20
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        texts = [item['text'] for item in batch]
        res = ask_gpt(json.dumps(texts, indent=4, ensure_ascii=False), response_json=True, valid_def=None, log_title='proofread', system=system)
        for j in range(min(len(batch), len(res))):
            rprint(f"[cyan]{batch[j]['text']}[/cyan]\n[green]{res[j]}[/green]")
            batch[j]['text'] = res[j]
        results.extend(batch)

    return results

# 将过长字幕按句拆分
def split_long_lines(segments: list):
    new_lines = []
    for sentence in segments:
        splited = split_subtitle_by_punctuation(sentence)
        new_lines.extend(splited)
    return new_lines

def split_subtitle_by_punctuation(subtitle_dict):
    """
    严格按 words 中的标点符号分割字幕，不强制限制每行字数
    
    参数:
        subtitle_dict: 原始字幕 dict，必须包含 "words" 列表，当word是标点时，没有start和end时间
        
    返回:
        拆分后的字幕 dict 列表
    """
    if "words" not in subtitle_dict:
        return [subtitle_dict]  # 如果没有 words 数据，直接返回原字幕
    
    words = subtitle_dict["words"]
    
    sentences = []
    current_sentence = ""
    current_words = []
    
    for word in words:
        current_sentence += word["word"]
        current_words.append(word)
        
        # 如果当前 word 有标点符号（逗号、句号等），则分割
        if word["word"] and word["word"] in {"，", "。", "；", "！", "？", "、", ",", ".", ";", "!", "?"} and len(current_sentence)>5:
            sentences.append((current_sentence, current_words.copy()))
            current_sentence = ""
            current_words = []
    
    # 添加最后一个句子（如果有剩余）
    if current_sentence:
        sentences.append((current_sentence, current_words.copy()))
    
    # 构建新的字幕 dict 列表
    result = []
    for sentence_text, sentence_words in sentences:
        if not sentence_words:
            continue
        # find the first word with 'start' field
        for item in sentence_words:
            if 'start' in item:
                begin_time = item["start"]
                break
        # find last word with 'end' field
        for item in reversed(sentence_words):
            if 'end' in item:
                end_time = item["end"]
                break
        
        new_dict = {
            "start": begin_time,
            "end": end_time,
            "text": sentence_text,
            "words": sentence_words
        }
        result.append(new_dict)
    
    # print("------------------------")
    # print(subtitle_dict['text'])
    # for r in result:
    #     print(f"{r['text']}\n")
    
    return result
    
if __name__ == "__main__":
    # load segments from json file
    data = json.loads(open("/app/output/audio/asr_result.json", 'r').read())
    # segments = proofread_with_semantic(data['segments'])
    # print(json.dumps(segments, indent=4, ensure_ascii=False))
    splited = split_long_lines(data['segments'])
    # segments = proofread_with_semantic(splited)
    #print(json.dumps(segments, indent=4, ensure_ascii=False))