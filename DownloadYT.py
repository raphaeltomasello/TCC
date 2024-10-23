import yt_dlp


def download_youtube_video(url, output_path='.'):
    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path + '/%(title)s.%(ext)s',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download concluído!")
    except Exception as e:
        print(f"Ocorreu um erro: {str(e)}")

# URL do vídeo que você deseja baixar
video_url = "https://www.youtube.com/watch?v=7uZo1SGfVAA"  # Substitua esta URL pela desejada

# Chama a função para baixar o vídeo
download_youtube_video(video_url)