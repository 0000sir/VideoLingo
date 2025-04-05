import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from rich import print as rprint
import subprocess

from core.config_utils import load_key
from core.all_whisper_methods.demucs_vl import demucs_main, RAW_AUDIO_FILE, VOCAL_AUDIO_FILE
from core.all_whisper_methods.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, compress_audio, CLEANED_CHUNKS_EXCEL_PATH, split_long_lines, proofread_with_semantic

from core.step1_ytdlp import find_video_files

WHISPER_FILE = "output/audio/for_whisper.mp3"
ENHANCED_VOCAL_PATH = "output/audio/enhanced_vocals.mp3"
ASR_FILE = "output/audio/asr_result.json"
PROOFREADED_ASR = "output/audio/asr_proofreaded.json"

def enhance_vocals(vocals_ratio=2.50):
    """Enhance vocals audio volume"""
    if not load_key("demucs"):
        return RAW_AUDIO_FILE

    if os.path.exists(ENHANCED_VOCAL_PATH):
        rprint(f"[cyan]🎙️ Enhancing vocals file exists: {ENHANCED_VOCAL_PATH}[/cyan]")
        return ENHANCED_VOCAL_PATH
        
    try:
        rprint(f"[cyan]🎙️ Enhancing vocals with volume ratio: {vocals_ratio}[/cyan]")
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{VOCAL_AUDIO_FILE}" '
            f'-filter:a "volume={vocals_ratio}" '
            f'"{ENHANCED_VOCAL_PATH}"'
        )
        subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)
        
        return ENHANCED_VOCAL_PATH
    except subprocess.CalledProcessError as e:
        rprint(f"[red]Error enhancing vocals: {str(e)}[/red]")
        return VOCAL_AUDIO_FILE  # Fallback to original vocals if enhancement fails
    
def transcribe():
    if os.path.exists(CLEANED_CHUNKS_EXCEL_PATH):
        rprint("[yellow]⚠️ Transcription results already exist, skipping transcription step.[/yellow]")
        return

    if not os.path.exists(ASR_FILE):
    
        # step0 Convert video to audio
        video_file = find_video_files()
        convert_video_to_audio(video_file)

        # step1 Demucs vocal separation:
        if load_key("demucs"):
            demucs_main()
        
        # step2 Compress audio
        choose_audio = enhance_vocals() if load_key("demucs") else RAW_AUDIO_FILE
        whisper_audio = compress_audio(choose_audio, WHISPER_FILE)

        if load_key("whisper.runtime") == "paraformer":
            from core.all_whisper_methods.paraformer import transcribe_audio_dashscope as ts
            rprint("[cyan]🎤 Transcribing audio with paraformer...[/cyan]")
            combined_result = ts(whisper_audio)
        else:
            # step3 Extract audio
            segments = split_audio(whisper_audio)
            
            # step4 Transcribe audio
            all_results = []
            if load_key("whisper.runtime") == "local":
                from core.all_whisper_methods.whisperX_local import transcribe_audio as ts
                rprint("[cyan]🎤 Transcribing audio with local model...[/cyan]")
            else:
                from core.all_whisper_methods.whisperX_302 import transcribe_audio_302 as ts
                rprint("[cyan]🎤 Transcribing audio with 302 API...[/cyan]")

            for start, end in segments:
                result = ts(whisper_audio, start, end)
                all_results.append(result)
            
            # step5 Combine results
            combined_result = {'segments': []}
            for result in all_results:
                combined_result['segments'].extend(result['segments'])
        
        # split sentence with punctuation marks
        
        with open(ASR_FILE, 'w', encoding='utf-8') as f:
            json.dump(combined_result, f, indent=4, ensure_ascii=False)
    else:
        combined_result = json.loads(open(ASR_FILE, 'r').read())
    
    # step6 Process df
    # 按词切分，我觉得会破坏语义，跳过
    # df = process_transcription(combined_result)
    # save_results(df)
    # 直接根据语义校对，第三步完全可以跳过
    if not os.path.exists(PROOFREADED_ASR):
        splited = split_long_lines(combined_result['segments'])
        proofreaded_segments = proofread_with_semantic(splited)
        result = {'segments': proofreaded_segments}
        with open(PROOFREADED_ASR, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
if __name__ == "__main__":
    transcribe()