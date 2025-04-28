import platform
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

#I don't know if this works and I'm not planning to install linux to test it, use at your own discretion
def get_linux_cameras():
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE)
        output = result.stdout.decode()
    except FileNotFoundError:
        print("Please install v4l-utils: sudo apt install v4l-utils")
        return []

    cameras = []
    blocks = output.strip().split('\n\n')
    for block in blocks:
        lines = block.strip().split('\n')
        name = lines[0].strip()
        for line in lines[1:]:
            match = re.search(r'/dev/video(\d+)', line)
            if match:
                index = int(match.group(1))
                cameras.append((index, name))
    return cameras

def get_macos_cameras():
    if not shutil.which("ffmpeg"):
        print("Warning: ffmpeg not installed. Attempting detection via system_profiler...")
        flag = True

        try:
            result = subprocess.run(['system_profiler', 'SPCameraDataType'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)
            output = result.stdout

            camera_matches = re.findall(r'^\s{4}(.+?):', output, re.MULTILINE)
            if camera_matches:
                cameras = [(i, name.strip()) for i, name in enumerate(camera_matches)]
                return cameras
            else:
                print("No cameras detected via system_profiler. Defaulting to 1 unknown camera.")
                return [(0, "Unknown Camera 0")]

        except Exception as e:
            print("Error detecting macOS cameras via system_profiler:", e)
            return [(0, "Unknown Camera 0")]

    else:
        # If ffmpeg is available, use the proper ffmpeg method
        try:
            result = subprocess.run(
                ['ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore')
            output = result.stderr

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
        return get_macos_cameras()
    else:
        print("Unsupported OS.")
        return []
    