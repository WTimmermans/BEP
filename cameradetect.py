import platform
import cv2
import subprocess
import re
import shutil

def get_windows_cameras():
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        print("Installing pygrabber for Windows camera detection...")
        subprocess.check_call(["pip", "install", "pygrabber"])
        from pygrabber.dshow_graph import FilterGraph

    graph = FilterGraph()
    device_names = graph.get_input_devices()
    return [(i, name) for i, name in enumerate(device_names)]

def get_macos_camera_names():
    if not shutil.which("ffmpeg"):
        print("ffmpeg is not installed. Install it with: brew install ffmpeg")
        return []
    try:
        result = subprocess.run(
            ['ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore')
        output = result.stderr

        # Extract only the video section
        video_section = re.search(
            r"AVFoundation video devices:(.*?)AVFoundation audio devices:",
            output,
            re.DOTALL)
        if not video_section:
            print("No video devices section found in ffmpeg output.")
            return []

        video_lines = video_section.group(1).strip().splitlines()
        cameras = []
        for line in video_lines:
            match = re.search(r'\[(\d+)\] (.+)', line)
            if match:
                idx, name = int(match.group(1)), match.group(2).strip()
                cameras.append((idx, name))

        # Remove duplicates by name (keep first occurrence)
        seen = set()
        unique = []
        for idx, name in cameras:
            if name not in seen:
                unique.append((idx, name))
                seen.add(name)
        return unique

    except Exception as e:
        print("Error detecting macOS cameras:", e)
        return []


def detect_cameras():
    system = platform.system()
    if system == "Windows":
        return get_windows_cameras()
    elif system == "Linux":
        return get_linux_cameras()
    elif system == "Darwin":
        return get_macos_camera_names()
    else:
        print("Unsupported OS.")
        return []

def show_cameras(cameras):
    caps = [cv2.VideoCapture(index) for index, _ in cameras]
    while True:
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"{cameras[i][1]}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cameras = detect_cameras()
    if cameras:
        print("\nDetected Cameras:")
        for index, name in cameras:
            print(f"[{index}] {name}")
    else:
        print("No cameras detected.")
