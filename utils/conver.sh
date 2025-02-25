for f in *.mp4;
#do ffmpeg -y -i "$f" -s 1920x1080 -vcodec libx264 -preset fast -qscale 4 -r 25 "../tmp/${f%.mp4}.mp4";
#do ffmpeg -y -i "$f" -s 1920x1080 -vcodec libx264 -preset fast -qscale 4 "/home/ltp/users/9T/CODES/projects2/baokang/annotation/VIA2COCO-master/input_baokang/videos/2K/high/e2w/${f%.mp4}.mp4";
do ffmpeg -y -i "$f" -vf scale=1920:1080 "/home/ltp/users/9T/CODES/projects2/baokang/annotation/VIA2COCO-master/input_baokang/videos/2K/high/e2w/${f%.mp4}.mp4" -hide_banner
done