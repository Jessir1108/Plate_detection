from moviepy.video.io.VideoFileClip import VideoFileClip

def split_video(input_path, output_path, segment_duration=60):
    # Cargar el vídeo
    video = VideoFileClip(input_path)
    
    # Obtener la duración del vídeo en segundos
    video_duration = int(video.duration)
    
    # Calcular el número de segmentos
    num_segments = (video_duration // segment_duration) + 1
    
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, video_duration)
        
        # Recortar el segmento
        segment = video.subclip(start_time, end_time)
        
        # Guardar el segmento
        segment.write_videofile(f"{output_path}_part{i+1}.mp4", codec="libx264")

# Ejemplo de uso
input_video_path = "videos/Raw/12raw.mp4"
output_video_path = "videos/12"
split_video(input_video_path, output_video_path)